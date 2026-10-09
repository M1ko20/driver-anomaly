"""LOSO cross-validation for an improved two-stream ST-GCN autoencoder, Dense
window (60/1/1) -- comparable to train_loso_beta_v2.py (TCN AE) and
train_loso_beta_mamba_real_v2.py (Mamba/S6 AE).

Four fixes over the earlier ST-GCN version: (1) angles become node features
and long-range distances become virtual bone edges, instead of everything
being a flat vector; (2) a learnable per-edge importance mask replaces the
fixed 0/1 adjacency; (3) train-safe-only skeleton augmentation (rotation,
L/R mirroring, light jitter); (4) width/depth left unchanged unless
overfitting shows up.

Input: beta_raw_v2 (T,52), same source as the TCN/Mamba LOSO runs. Same
protocol: per-fold standardization, early stopping on val-safe MSE, ROC/AUC
per fold + pooled + mean/std."""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
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
V = 13
C = 4

PHYSICAL_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
                  (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]
VIRTUAL_EDGES = [(0, 9), (0, 10), (11, 9), (12, 10), (9, 10), (5, 9), (6, 10)]
ALL_EDGES = PHYSICAL_EDGES + VIRTUAL_EDGES
E = len(ALL_EDGES)

MIRROR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_adjacency(V, edges, self_loop=True):
    A = np.zeros((V, V), dtype=np.float32)
    for p, c in edges:
        A[p, c] = 1.0
        A[c, p] = 1.0
    if self_loop:
        A += np.eye(V, dtype=np.float32)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-6)))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


def load_meta():
    with open(os.path.join(RAW_DIR, "meta.json")) as f:
        return json.load(f)


def feat_to_node_tensor(feat):
    """(T,52) beta_raw -> (T,V,4) = (x,y,conf,angle)."""
    T = feat.shape[0]
    xyz = feat[:, :39].reshape(T, V, 3)
    angle_ch = np.zeros((T, V, 1), dtype=np.float32)
    angle_ch[:, 7, 0] = feat[:, 47]
    angle_ch[:, 8, 0] = feat[:, 48]
    angle_ch[:, 5, 0] = feat[:, 49]
    angle_ch[:, 6, 0] = feat[:, 50]
    angle_ch[:, 0, 0] = feat[:, 51]
    return np.concatenate([xyz, angle_ch], axis=-1).astype(np.float32)


def make_windows(clip_names, meta_by_clip, label_filter=None):
    windows, win_labels = [], []
    for clip_name in clip_names:
        label = meta_by_clip[clip_name]["label"]
        if label_filter is not None and label != label_filter:
            continue
        feat = np.load(os.path.join(RAW_DIR, clip_name + ".npy"))
        node_feat = feat_to_node_tensor(feat)
        T = node_feat.shape[0]
        span = (SEQ_LEN - 1) * FRAME_STRIDE + 1
        if T < span:
            continue
        for start in range(0, T - span + 1, WINDOW_STEP):
            idx = np.arange(start, start + span, FRAME_STRIDE)
            windows.append(node_feat[idx])
            win_labels.append(1 if label == "anomaly" else 0)
    if not windows:
        return np.zeros((0, SEQ_LEN, V, C), dtype=np.float32), np.array([])
    return np.stack(windows).astype(np.float32), np.array(win_labels)


def augment_window(w, rot_deg=15.0, jitter_std=0.02, rng=None):
    """w: (T,V,4) = (x,y,conf,angle). Augmentace jen pro train-safe okna."""
    rng = rng or np.random
    w = w.copy()

    if rng.rand() < 0.5:
        w = w[:, MIRROR_SWAP, :]
        w[:, :, 0] = -w[:, :, 0]

    theta = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x, y = w[:, :, 0].copy(), w[:, :, 1].copy()
    w[:, :, 0] = x * cos_t - y * sin_t
    w[:, :, 1] = x * sin_t + y * cos_t
    nose_x = w[:, 0, 0]
    nose_y = w[:, 0, 1]
    norm = np.sqrt(nose_x ** 2 + nose_y ** 2) + 1e-8
    cos_a = np.clip(-nose_y / norm, -1.0, 1.0)
    w[:, 0, 3] = np.arccos(cos_a)

    jitter = rng.normal(0, jitter_std, size=w[:, :, :2].shape).astype(np.float32)
    w[:, :, :2] = w[:, :, :2] + jitter

    return w.astype(np.float32)


class NodeWindowDataset(Dataset):
    def __init__(self, windows, augment=False, seed=0):
        self.windows = windows
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        if self.augment:
            w = augment_window(w, rng=self.rng)
        return torch.from_numpy(w.transpose(2, 0, 1))


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.register_buffer("A", torch.tensor(A))
        self.edge_importance = nn.Parameter(torch.ones_like(self.A))
        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.tcn = nn.Conv2d(out_ch, out_ch, kernel_size=(5, 1), padding=(2, 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.gcn(x)
        A_eff = self.A * self.edge_importance
        x = torch.einsum("bctv,vw->bctw", x, A_eff)
        x = self.tcn(x)
        return self.act(self.bn(x))


class TemporalBlock2D(nn.Module):
    """Bones stream - channel mixing + temporal conv, s learnable per-edge
    importance (misto grafove konvoluce - kosti uz nesou geometrii samy)."""

    def __init__(self, in_ch, out_ch, n_edges):
        super().__init__()
        self.edge_importance = nn.Parameter(torch.ones(n_edges))
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.tcn = nn.Conv2d(out_ch, out_ch, kernel_size=(5, 1), padding=(2, 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x * self.edge_importance.view(1, 1, 1, -1)
        x = self.conv1(x)
        x = self.tcn(x)
        return self.act(self.bn(x))


class ImprovedTwoStreamSTGCNAutoencoder(nn.Module):
    def __init__(self, V, physical_edges, all_edges, A_joints):
        super().__init__()
        parent_idx = [e[0] for e in all_edges]
        child_idx = [e[1] for e in all_edges]
        self.register_buffer("parent_idx", torch.tensor(parent_idx, dtype=torch.long))
        self.register_buffer("child_idx", torch.tensor(child_idx, dtype=torch.long))
        n_edges = len(all_edges)

        self.joint_encoder = nn.Sequential(
            STGCNBlock(C, 32, A_joints),
            STGCNBlock(32, 16, A_joints),
        )
        self.bone_encoder = nn.Sequential(
            TemporalBlock2D(3, 32, n_edges),
            TemporalBlock2D(32, 16, n_edges),
        )

        incidence = np.zeros((V, n_edges), dtype=np.float32)
        for e_idx, c in enumerate(child_idx):
            incidence[c, e_idx] = 1.0
        self.register_buffer("incidence", torch.tensor(incidence))

        self.fuse = nn.Conv2d(32, 16, kernel_size=1)
        self.bottleneck = STGCNBlock(16, 8, A_joints)

        self.decoder = nn.Sequential(
            STGCNBlock(8, 16, A_joints),
            STGCNBlock(16, 32, A_joints),
        )
        self.out_proj = nn.Conv2d(32, C, kernel_size=1)

    def compute_bones(self, joints):
        x = joints[:, 0]
        y = joints[:, 1]
        dx = x[..., self.child_idx] - x[..., self.parent_idx]
        dy = y[..., self.child_idx] - y[..., self.parent_idx]
        length = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        return torch.stack([dx, dy, length], dim=1)

    def forward(self, joints):
        bones = self.compute_bones(joints)

        j_lat = self.joint_encoder(joints)
        b_lat = self.bone_encoder(bones)
        b_on_j = torch.einsum("bcte,ve->bctv", b_lat, self.incidence)

        fused = torch.cat([j_lat, b_on_j], dim=1)
        fused = self.fuse(fused)
        z = self.bottleneck(fused)

        out = self.decoder(z)
        return self.out_proj(out)


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
            batch = torch.from_numpy(windows[i:i + batch_size].transpose(0, 3, 1, 2)).to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2, 3))
            mses.append(mse.cpu().numpy())
    return np.concatenate(mses) if mses else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only_fold", type=int, default=-1)
    ap.add_argument("--no_augment", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = load_meta()
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    print("Device:", device, "seed:", args.seed, "augment:", not args.no_augment)
    print("Drivers ({}): {}".format(len(drivers), drivers))
    print("Physical edges:", len(PHYSICAL_EDGES), "Virtual edges:", len(VIRTUAL_EDGES), "Total:", E)

    A_joints = build_adjacency(V, PHYSICAL_EDGES)
    tmp_model = ImprovedTwoStreamSTGCNAutoencoder(V, PHYSICAL_EDGES, ALL_EDGES, A_joints)
    print("ImprovedTwoStreamSTGCNAutoencoder trainable params:", count_params(tmp_model))
    n_params = count_params(tmp_model)
    del tmp_model

    clips_by_driver = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}

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

        flat_train = train_windows_raw.reshape(-1, V, C)
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

        train_ds = NodeWindowDataset(train_windows, augment=not args.no_augment, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(NodeWindowDataset(val_windows, augment=False), batch_size=BATCH_SIZE, shuffle=False)

        model = ImprovedTwoStreamSTGCNAutoencoder(V, PHYSICAL_EDGES, ALL_EDGES, A_joints).to(device)
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
    print("VYSLEDKY LOSO CROSS-VALIDATION (vylepseny two-stream ST-GCN AE, Dense okno)")
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
        "n_params": n_params,
        "augment": not args.no_augment,
        "fold_results": [{"driver": d, "auc": a} for d, a in fold_results],
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "pooled_auc": float(pooled_auc),
    }
    out_json = "/home/fei/mik0542/loso_stgcn_improved_v2_seed{}_results.json".format(args.seed)
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)

    plt.figure(figsize=(8, 7))
    plt.plot(pooled_fpr, pooled_tpr, color="darkorange", lw=2.5,
              label="Pooled ROC (AUC = {:.4f})".format(pooled_auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("LOSO Pooled ROC - vylepseny ST-GCN AE (YOLO26-pose)")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.savefig("/home/fei/mik0542/roc_LOSO_pooled_stgcn_improved_v2_seed{}.png".format(args.seed),
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
    plt.title("LOSO Per-Fold ROC - vylepseny ST-GCN AE (YOLO26-pose)")
    plt.legend(loc="lower right", fontsize=8); plt.grid(True, alpha=0.3)
    plt.savefig("/home/fei/mik0542/roc_LOSO_folds_stgcn_improved_v2_seed{}.png".format(args.seed),
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\nUlozeno:", out_json)
    print("Hotovo.")


if __name__ == "__main__":
    main()
