"""LOSO cross-validation for a real selective Mamba/S6 autoencoder on Beta
features, Dense window (60/1/1) -- directly comparable to
train_loso_beta_v2.py (TCN AE).

Unlike the earlier train_eval_mamba_beta_v2.py (fixed-decay low-pass block),
this implements an actual selective scan per Mamba/S6 (Gu & Dao 2023),
simplified the same way as the official implementation. No custom CUDA
kernels needed (server GPU is Pascal, too old for mamba-ssm) -- sequences
are short enough for a plain PyTorch scan.

Parameter count is tuned to stay within ~10% of the TCN AE, to keep
"mechanism vs. capacity" a controlled comparison. Same protocol as
train_loso_beta_v2.py: per-fold standardization on train-safe, early
stopping on val-safe MSE, ROC/AUC per fold + pooled + mean/std."""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_DIR = "/home/fei/mik0542/beta_raw_v2"
SEQ_LEN = 60
FRAME_STRIDE = 1
WINDOW_STEP = 1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
N_FEATURES = 52

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meta():
    with open(os.path.join(RAW_DIR, "meta.json")) as f:
        return json.load(f)


def make_windows(clip_names, meta_by_clip, label_filter=None):
    windows, win_labels = [], []
    for clip_name in clip_names:
        label = meta_by_clip[clip_name]["label"]
        if label_filter is not None and label != label_filter:
            continue
        feat = np.load(os.path.join(RAW_DIR, clip_name + ".npy"))
        T = feat.shape[0]
        span = (SEQ_LEN - 1) * FRAME_STRIDE + 1
        if T < span:
            continue
        for start in range(0, T - span + 1, WINDOW_STEP):
            idx = np.arange(start, start + span, FRAME_STRIDE)
            windows.append(feat[idx])
            win_labels.append(1 if label == "anomaly" else 0)
    if not windows:
        return np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32), np.array([])
    return np.stack(windows).astype(np.float32), np.array(win_labels)


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return torch.from_numpy(self.windows[i].T)


class SelectiveSSM(nn.Module):
    """Realny (ne-komplexni) selektivni state-space scan (S6-styl).

    x: (B, T, D) -> y: (B, T, D). D = kanaly (po in_proj/conv), N = state dim.
    Delta/B/C jsou POCITANY z x_t (per-timestep), tj. skutecne selektivni -
    na rozdil od SimpleSSMBlock (pevne 'a' nezavisle na vstupu).
    """

    def __init__(self, d_model, d_state=8, dt_rank=None, chunk_len=10):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk_len = chunk_len
        dt_rank = dt_rank or max(4, d_model // 8)

        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D_skip = nn.Parameter(torch.ones(d_model))

        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        with torch.no_grad():
            self.dt_proj.bias.copy_(torch.log(torch.expm1(torch.rand(d_model) * 0.1 + 0.01)))

    def _chunked_scan(self, log_a, bx):
        """Vektorizovany (chunk-wise paralelni) scan MISTO python for-loop
        pres T. Matematicky presne stejny vysledek jako sekvencni
        h_t = exp(log_a_t)*h_{t-1} + bx_t, jen vypocitany po blocich delky
        `chunk_len` (standardni trik pro linearni/diagonalni SSM -
        S4D/Mamba/RetNet/GLA chunk scany), misto sekvencniho pythonoveho
        for-loopu pres vsech T=60 kroku (ktery byl na male GPU
        kernel-launch-overhead-bound a ~10x pomalejsi).

        DULEZITE (numericka stabilita): naivni verze tohoto triku
        (exp(S_t) * cumsum(exp(-S_s)*bx_s)) pocita exp(-S_s) SAMOSTATNE,
        coz muze prasknout do +inf, i kdyz je vysledny soucin exp(S_t)*
        exp(-S_s)=exp(S_t-S_s) vzdy v (0,1] (protoze S je kumulativni
        soucet ZAPORNYCH hodnot, tedy S_t<=S_s pro t>=s). Misto toho se
        tu pocita rozdil exponentu PRIMO (exp(S_t - S_s)), ktery nikdy
        nemuze prekrocit 0 a tedy nemuze overflowovat - jen bezpecne
        podteka k 0 pro davno rozpadle prispevky. Cena: O(L^2) misto
        O(L) uvnitr chunku, ale L (8-30) je male, takze levne.

        log_a, bx: (B,T,D,N) -> vraci h_all: (B,T,D,N) (h_1..h_T)."""
        Bsz, T, D, N = bx.shape
        L = self.chunk_len
        h_all = torch.empty_like(bx)
        h_carry = torch.zeros(Bsz, D, N, device=bx.device, dtype=bx.dtype)
        for start in range(0, T, L):
            end = min(start + L, T)
            la = log_a[:, start:end]
            bxc = bx[:, start:end]
            l = end - start
            S = torch.cumsum(la, dim=1)

            carry_term = torch.exp(S) * h_carry.unsqueeze(1)

            S_t = S.unsqueeze(2)
            S_s = S.unsqueeze(1)
            diff = S_t - S_s
            causal = torch.tril(torch.ones(l, l, dtype=torch.bool, device=bx.device))
            diff = diff.masked_fill(~causal.view(1, l, l, 1, 1), float("-inf"))
            decay = torch.exp(diff)
            pair_term = (decay * bxc.unsqueeze(1)).sum(dim=2)

            h_chunk = carry_term + pair_term
            h_all[:, start:end] = h_chunk
            h_carry = h_chunk[:, -1]
        return h_all

    def forward(self, x):
        Bsz, T, D = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log)

        x_dbl = self.x_proj(x)
        dt_low, Bt, Ct = torch.split(
            x_dbl, [x_dbl.shape[-1] - 2 * N, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(dt_low))
        delta = torch.clamp(delta, max=4.0)

        log_Abar = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        Bx = delta.unsqueeze(-1) * Bt.unsqueeze(2) * x.unsqueeze(-1)

        h_all = self._chunked_scan(log_Abar, Bx)
        y = (h_all * Ct.unsqueeze(2)).sum(dim=-1)
        y = y + x * self.D_skip
        return y


class MambaBlock(nn.Module):
    """Standardni tvar Mamba bloku: in_proj (expand + gate z), kratky
    depthwise kauzalni conv, SelectiveSSM, gating SiLU(z)*ssm_out, out_proj."""

    def __init__(self, d_in, d_out, d_inner=None, d_state=8, conv_kernel=3):
        super().__init__()
        d_inner = d_inner or d_out
        self.in_proj = nn.Linear(d_in, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel,
                               padding=conv_kernel - 1, groups=d_inner)
        self.conv_kernel = conv_kernel
        self.ssm = SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        xt = x.transpose(1, 2)
        xz = self.in_proj(xt)
        x_b, z = xz.chunk(2, dim=-1)

        x_b = x_b.transpose(1, 2)
        x_b = self.conv(x_b)[:, :, :xt.shape[1]]
        x_b = F.silu(x_b).transpose(1, 2)

        ssm_out = self.ssm(x_b)
        gated = ssm_out * F.silu(z)
        out = self.out_proj(gated)
        out = self.norm(out)
        return out.transpose(1, 2)


class MambaAutoencoder(nn.Module):
    def __init__(self, n_features, d_state=8):
        super().__init__()
        self.encoder = nn.Sequential(
            MambaBlock(n_features, 64, d_state=d_state),
            MambaBlock(64, 32, d_state=d_state),
            MambaBlock(32, 16, d_state=d_state),
        )
        self.decoder = nn.Sequential(
            MambaBlock(16, 32, d_state=d_state),
            MambaBlock(32, 64, d_state=d_state),
            nn.Conv1d(64, n_features, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_n = 0.0, 0
    criterion = nn.MSELoss()
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.shape[0]
            total_n += batch.shape[0]
    return total_loss / max(total_n, 1)


def per_window_mse(model, windows, batch_size=256):
    model.eval()
    mses = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i:i + batch_size].transpose(0, 2, 1)).to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            mses.append(mse.cpu().numpy())
    return np.concatenate(mses) if mses else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d_state", type=int, default=8)
    ap.add_argument("--only_fold", type=int, default=-1,
                     help="pokud >=0, spusti jen jeden fold (smoke test)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = load_meta()
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    print("Device:", device, "seed:", args.seed, "d_state:", args.d_state)
    print("Drivers ({}): {}".format(len(drivers), drivers))

    clips_by_driver = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}

    tmp_model = MambaAutoencoder(N_FEATURES, d_state=args.d_state)
    n_params = count_params(tmp_model)
    print("MambaAutoencoder trainable params:", n_params)
    del tmp_model

    fold_results = []
    all_fold_fprs, all_fold_tprs = [], []
    pooled_labels, pooled_scores = [], []

    fold_iter = list(enumerate(drivers))
    if args.only_fold >= 0:
        fold_iter = [fold_iter[args.only_fold]]

    for fold_idx, test_driver in fold_iter:
        remaining = [d for d in drivers if d != test_driver]
        val_driver = remaining[fold_idx % len(remaining)]
        train_drivers = [d for d in remaining if d != val_driver]

        print("\n{}".format("-" * 60))
        print("FOLD {}/{}: test={} val={} train={}".format(
            fold_idx + 1, len(drivers), test_driver, val_driver, train_drivers))

        train_clips = [c for d in train_drivers for c in clips_by_driver[d]]
        val_clips = clips_by_driver[val_driver]
        test_clips = clips_by_driver[test_driver]

        train_windows_raw, _ = make_windows(train_clips, meta_by_clip, label_filter="safe")
        val_windows_raw, _ = make_windows(val_clips, meta_by_clip, label_filter="safe")
        test_windows_raw, test_labels = make_windows(test_clips, meta_by_clip)

        if len(train_windows_raw) == 0 or len(val_windows_raw) == 0 or len(test_windows_raw) == 0:
            print("  PRESKAKUJI - chybi data")
            continue

        flat_train = train_windows_raw.reshape(-1, N_FEATURES)
        mean = flat_train.mean(axis=0)
        std = flat_train.std(axis=0)
        std[std < 1e-6] = 1e-6

        def standardize(w):
            return (w - mean) / std

        train_windows = standardize(train_windows_raw)
        val_windows = standardize(val_windows_raw)
        test_windows = standardize(test_windows_raw)

        print("  Train windows: {} | Val(safe): {} | Test: {} (anomaly frac {:.3f})".format(
            train_windows.shape[0], val_windows.shape[0], test_windows.shape[0], test_labels.mean()))

        train_loader = DataLoader(WindowDataset(train_windows), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(WindowDataset(val_windows), batch_size=BATCH_SIZE, shuffle=False)

        model = MambaAutoencoder(N_FEATURES, d_state=args.d_state).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val = float("inf")
        patience_ctr = 0
        best_state = None

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = run_epoch(model, train_loader, optimizer)
            val_loss = run_epoch(model, val_loader, optimizer=None)
            if epoch == 1 or epoch % 10 == 0:
                print("    Epoch {:3d} | train MSE {:.6f} | val MSE {:.6f}".format(epoch, train_loss, val_loss))

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_ctr = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print("    Early stopping at epoch {} (best val MSE {:.6f})".format(epoch, best_val))
                    break

        model.load_state_dict(best_state)

        test_mse = per_window_mse(model, test_windows)
        fpr, tpr, _ = roc_curve(test_labels, test_mse)
        fold_auc = auc(fpr, tpr)
        print("  >> FOLD {} test_driver={} AUC={:.4f}".format(fold_idx + 1, test_driver, fold_auc))

        fold_results.append((test_driver, fold_auc))
        all_fold_fprs.append(fpr)
        all_fold_tprs.append(tpr)
        pooled_labels.extend(test_labels.tolist())
        pooled_scores.extend(test_mse.tolist())

    if args.only_fold >= 0:
        print("\nSmoke test hotovy (jen fold {}), koncim bez agregace/grafu.".format(args.only_fold))
        return

    print("\n{}".format("=" * 60))
    print("VYSLEDKY LOSO CROSS-VALIDATION (Beta, realna selektivni Mamba AE, Dense okno)")
    print("=" * 60)
    auc_values = [r[1] for r in fold_results]
    mean_auc = float(np.mean(auc_values))
    std_auc = float(np.std(auc_values))
    print("\n{:<8} {:<10} {:<10}".format("Fold", "Ridic", "AUC"))
    for i, (drv, auc_val) in enumerate(fold_results):
        print("{:<8} {:<10} {:.4f}".format(i + 1, drv, auc_val))
    print("PRUMER: {:.4f} +/- {:.4f}".format(mean_auc, std_auc))

    pooled_labels = np.array(pooled_labels)
    pooled_scores = np.array(pooled_scores)
    pooled_fpr, pooled_tpr, _ = roc_curve(pooled_labels, pooled_scores)
    pooled_auc = auc(pooled_fpr, pooled_tpr)
    print("\nPoolovana AUC: {:.4f}".format(pooled_auc))

    results_summary = {
        "seed": args.seed,
        "d_state": args.d_state,
        "n_params": n_params,
        "fold_results": [{"driver": d, "auc": a} for d, a in fold_results],
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "pooled_auc": float(pooled_auc),
    }
    out_json = "/home/fei/mik0542/loso_mamba_real_v2_seed{}_results.json".format(args.seed)
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)

    plt.figure(figsize=(8, 7))
    plt.plot(pooled_fpr, pooled_tpr, color="darkorange", lw=2.5,
              label="Pooled ROC (AUC = {:.4f})".format(pooled_auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("LOSO Pooled ROC - Beta + real Mamba(S6) AE (YOLO26-pose)")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.savefig("/home/fei/mik0542/roc_LOSO_pooled_mamba_real_v2_seed{}.png".format(args.seed),
                dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 7))
    mean_fpr = np.linspace(0, 1, 200)
    tprs_interp = []
    for i, (fpr_i, tpr_i) in enumerate(zip(all_fold_fprs, all_fold_tprs)):
        drv, auc_i = fold_results[i]
        plt.plot(fpr_i, tpr_i, lw=1, alpha=0.4, label="Fold {} {} (AUC={:.2f})".format(i + 1, drv, auc_i))
        interp_tpr = np.interp(mean_fpr, fpr_i, tpr_i)
        interp_tpr[0] = 0.0
        tprs_interp.append(interp_tpr)
    mean_tpr = np.mean(tprs_interp, axis=0); mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, color="darkorange", lw=2.5,
              label="Mean ROC (AUC = {:.4f} +/- {:.4f})".format(mean_auc, std_auc))
    std_tpr = np.std(tprs_interp, axis=0)
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1),
                      color="darkorange", alpha=0.15, label="+/- 1 std")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("LOSO Per-Fold ROC - Beta + real Mamba(S6) AE (YOLO26-pose)")
    plt.legend(loc="lower right", fontsize=8); plt.grid(True, alpha=0.3)
    plt.savefig("/home/fei/mik0542/roc_LOSO_folds_mamba_real_v2_seed{}.png".format(args.seed),
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\nUlozeno:", out_json)
    print("Hotovo.")


if __name__ == "__main__":
    main()
