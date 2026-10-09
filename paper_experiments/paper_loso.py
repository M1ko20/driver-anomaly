"""LOSO (14 drivers) runs of every model/feature variant that is reported in the
Elsevier article, with the same protocol as train_loso_beta_v2.py:
  - test driver = all clips, val driver = safe clips only (early stopping),
    remaining 12 drivers = safe clips only (training); val_driver =
    remaining[fold_idx % 13]  (identical fold assignment to all earlier LOSO runs),
  - per-fold z-score standardization fitted on train windows only,
  - Adam lr 1e-3, batch 64, max 100 epochs, early stopping patience 10 on val MSE.
Model classes are copied verbatim from the original scripts
(train_loso_beta_v2.py, train_eval_predictive_beta_v2.py,
train_eval_mamba_beta_v2.py, train_eval_stgcn_v2.py).

Features are derived from beta_raw_v2 (first 39 columns = x, y, confidence of the
13 keypoints), so nothing has to be re-extracted. Kinematics = Savitzky-Golay
(5, 2, mode=nearest) + np.gradient, exactly as in build_gamma_features_v2.py.

Output: /home/fei/mik0542/paper_experiments/<exp>_seed<k>.json (written after every
fold, and resumed from if it already exists)."""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy.signal import savgol_filter

RAW_DIR = "/home/fei/mik0542/beta_raw_v2"
OUT_DIR = "/home/fei/mik0542/paper_experiments"
BATCH_SIZE = 64
PATIENCE = 10
LR = 1e-3
V = 13

EXPS = {
    # name: (feature set, model, SEQ_LEN, FRAME_STRIDE)
    "coords_tcn":   ("coords",   "tcn",      60, 1),
    "geom_tcn":     ("geom",     "tcn",      60, 1),
    "kin_tcn":      ("kin",      "tcn",      60, 1),
    "forecast_tcn": ("geom",     "forecast", 60, 1),
    "mambalike_lt": ("geom",     "mambalike", 30, 3),
    "stgcn_coords": ("coords",   "stgcn",    60, 1),
    "stgcn_kin":    ("kinnode",  "stgcn",    60, 1),
}
CONTEXT = 45

parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True, choices=sorted(EXPS))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--only_fold", type=int, default=-1)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--tag", default="")
args = parser.parse_args()
FEAT, MODEL, SEQ_LEN, FRAME_STRIDE = EXPS[args.exp]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ features
def build_features(b, feat):
    T = b.shape[0]
    alpha = b[:, :39]
    arr = alpha.reshape(T, V, 3)
    xy, conf = arr[:, :, :2], arr[:, :, 2:3]
    if feat == "coords":
        return alpha.astype(np.float32)
    if feat == "geom":
        return b.astype(np.float32)
    smoothed = savgol_filter(xy, window_length=5, polyorder=2, axis=0, mode="nearest") if T >= 5 else xy.copy()
    vel = np.gradient(smoothed, axis=0)
    acc = np.gradient(vel, axis=0)
    if feat == "kin":
        return np.concatenate([b, vel.reshape(T, -1), acc.reshape(T, -1)], axis=1).astype(np.float32)
    if feat == "kinnode":
        return np.concatenate([xy, conf, vel, acc], axis=2).reshape(T, -1).astype(np.float32)
    raise ValueError(feat)


def load_all():
    with open(os.path.join(RAW_DIR, "meta.json")) as f:
        meta = json.load(f)
    feats = {}
    for m in meta:
        b = np.load(os.path.join(RAW_DIR, m["clip"] + ".npy"))
        feats[m["clip"]] = build_features(b, FEAT)
    return meta, feats


def make_windows(clips, meta_by_clip, feats, label_filter=None):
    ws, labels, actions = [], [], []
    span = (SEQ_LEN - 1) * FRAME_STRIDE + 1
    for c in clips:
        m = meta_by_clip[c]
        if label_filter is not None and m["label"] != label_filter:
            continue
        f = feats[c]
        T = f.shape[0]
        if T < span:
            continue
        for start in range(0, T - span + 1):
            ws.append(f[start:start + span:FRAME_STRIDE])
            labels.append(1 if m["label"] == "anomaly" else 0)
            actions.append(m["action"])
    D = next(iter(feats.values())).shape[1]
    if not ws:
        return np.zeros((0, SEQ_LEN, D), np.float32), np.array([]), []
    return np.stack(ws).astype(np.float32), np.array(labels), actions


# -------------------------------------------------------------------- models
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        pad = dilation * 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.shape[2]] if out.shape[2] > x.shape[2] else out
        return self.act(self.bn(out))


class TCNAutoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(TCNBlock(n_features, 64, 1), TCNBlock(64, 32, 2), TCNBlock(32, 16, 4))
        self.decoder = nn.Sequential(TCNBlock(16, 32, 4), TCNBlock(32, 64, 2),
                                     nn.Conv1d(64, n_features, kernel_size=5, padding=2))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class PredictiveTCN(nn.Module):
    def __init__(self, n_features, horizon):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Sequential(TCNBlock(n_features, 64, 1), TCNBlock(64, 32, 2), TCNBlock(32, 16, 4))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Sequential(TCNBlock(16, 32, 1), TCNBlock(32, 64, 1),
                                     nn.Conv1d(64, n_features, kernel_size=1))

    def forward(self, past):
        z = self.encoder(past)
        z = self.pool(z)
        z = z.repeat(1, 1, self.horizon)
        return self.decoder(z)


class SimpleSSMBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, out_ch)
        self.a_logit = nn.Parameter(torch.zeros(out_ch))
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        xt = x.transpose(1, 2)
        u = self.in_proj(xt)
        a = torch.sigmoid(self.a_logit)
        B_, T, C = u.shape
        h = torch.zeros(B_, C, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = a * h + (1 - a) * u[:, t, :]
            outs.append(h)
        y = torch.stack(outs, dim=1).transpose(1, 2)
        return self.act(self.bn(y))


class MambaLikeAutoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(SimpleSSMBlock(n_features, 64), SimpleSSMBlock(64, 32), SimpleSSMBlock(32, 16))
        self.decoder = nn.Sequential(SimpleSSMBlock(16, 32), SimpleSSMBlock(32, 64),
                                     nn.Conv1d(64, n_features, kernel_size=1))

    def forward(self, x):
        return self.decoder(self.encoder(x))


EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
         (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]


def build_adjacency(Vn, edges):
    A = np.zeros((Vn, Vn), dtype=np.float32)
    for p, c in edges:
        A[p, c] = 1.0
        A[c, p] = 1.0
    A += np.eye(Vn, dtype=np.float32)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-6)))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.register_buffer("A", torch.tensor(A))
        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.tcn = nn.Conv2d(out_ch, out_ch, kernel_size=(5, 1), padding=(2, 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.gcn(x)
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.tcn(x)
        return self.act(self.bn(x))


class TemporalBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.tcn = nn.Conv2d(out_ch, out_ch, kernel_size=(5, 1), padding=(2, 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.tcn(x)
        return self.act(self.bn(x))


class TwoStreamSTGCNAutoencoder(nn.Module):
    def __init__(self, Vn, edges, A_joints, nc):
        super().__init__()
        parent_idx = [e[0] for e in edges]
        child_idx = [e[1] for e in edges]
        self.register_buffer("parent_idx", torch.tensor(parent_idx, dtype=torch.long))
        self.register_buffer("child_idx", torch.tensor(child_idx, dtype=torch.long))
        self.joint_encoder = nn.Sequential(STGCNBlock(nc, 32, A_joints), STGCNBlock(32, 16, A_joints))
        self.bone_encoder = nn.Sequential(TemporalBlock2D(3, 32), TemporalBlock2D(32, 16))
        incidence = np.zeros((Vn, len(edges)), dtype=np.float32)
        for e_idx, c in enumerate(child_idx):
            incidence[c, e_idx] = 1.0
        self.register_buffer("incidence", torch.tensor(incidence))
        self.fuse = nn.Conv2d(32, 16, kernel_size=1)
        self.bottleneck = STGCNBlock(16, 8, A_joints)
        self.decoder = nn.Sequential(STGCNBlock(8, 16, A_joints), STGCNBlock(16, 32, A_joints))
        self.out_proj = nn.Conv2d(32, nc, kernel_size=1)

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
        fused = self.fuse(torch.cat([j_lat, b_on_j], dim=1))
        return self.out_proj(self.decoder(self.bottleneck(fused)))


# ------------------------------------------------------------- model adapter
def make_model(D):
    if MODEL == "tcn":
        return TCNAutoencoder(D)
    if MODEL == "forecast":
        return PredictiveTCN(D, SEQ_LEN - CONTEXT)
    if MODEL == "mambalike":
        return MambaLikeAutoencoder(D)
    if MODEL == "stgcn":
        return TwoStreamSTGCNAutoencoder(V, EDGES, build_adjacency(V, EDGES), D // V)
    raise ValueError(MODEL)


def to_tensors(w_np):
    """(N, T, D) numpy -> tuple (input, target) of tensors on the device."""
    w = torch.from_numpy(np.ascontiguousarray(w_np)).to(device)
    if MODEL in ("tcn", "mambalike"):
        x = w.transpose(1, 2)
        return x, x
    if MODEL == "forecast":
        return w[:, :CONTEXT].transpose(1, 2), w[:, CONTEXT:].transpose(1, 2)
    N, T, D = w.shape
    x = w.reshape(N, T, V, D // V).permute(0, 3, 1, 2)
    return x, x


class WinDS(Dataset):
    def __init__(self, w):
        self.w = w

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        return self.w[i]


def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    crit = nn.MSELoss()
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x, y = to_tensors(batch.numpy())
            loss = crit(model(x), y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
    return tot / max(n, 1)


def window_scores(model, w, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(w), bs):
            x, y = to_tensors(w[i:i + bs])
            out.append(((model(x) - y) ** 2).mean(dim=tuple(range(1, y.dim()))).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    name = "{}{}_seed{}".format(args.exp, args.tag, args.seed)
    out_path = os.path.join(OUT_DIR, name + ".json")
    meta, feats = load_all()
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    clips_by_driver = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}
    D = next(iter(feats.values())).shape[1]
    print("exp={} feat={} model={} seq={}/{} D={} seed={} drivers={}".format(
        args.exp, FEAT, MODEL, SEQ_LEN, FRAME_STRIDE, D, args.seed, len(drivers)))
    sys.stdout.flush()

    res = {"exp": args.exp, "seed": args.seed, "feature_set": FEAT, "model": MODEL,
           "seq_len": SEQ_LEN, "frame_stride": FRAME_STRIDE, "n_features": D,
           "folds": {}}
    if os.path.exists(out_path):
        res = json.load(open(out_path))
        print("Resuming, done folds:", sorted(res["folds"]))
    pooled = {"y": [], "s": []}

    for fold_idx, test_driver in enumerate(drivers):
        if args.only_fold >= 0 and fold_idx != args.only_fold:
            continue
        if test_driver in res["folds"]:
            continue
        t0 = time.time()
        remaining = [d for d in drivers if d != test_driver]
        val_driver = remaining[fold_idx % len(remaining)]
        train_drivers = [d for d in remaining if d != val_driver]
        torch.manual_seed(args.seed * 1000 + fold_idx)
        np.random.seed(args.seed * 1000 + fold_idx)

        tr, _, _ = make_windows([c for d in train_drivers for c in clips_by_driver[d]], meta_by_clip, feats, "safe")
        va, _, _ = make_windows(clips_by_driver[val_driver], meta_by_clip, feats, "safe")
        te, y_te, act_te = make_windows(clips_by_driver[test_driver], meta_by_clip, feats)
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            print("SKIP fold", test_driver)
            continue
        flat = tr.reshape(-1, D)
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std[std < 1e-6] = 1e-6
        for w in (tr, va, te):
            w -= mean
            w /= std
        print("\nFOLD {}/{} test={} val={} | train {} val {} test {} (anomaly frac {:.3f})".format(
            fold_idx + 1, len(drivers), test_driver, val_driver, len(tr), len(va), len(te), y_te.mean()))
        sys.stdout.flush()

        model = make_model(D).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        tl = DataLoader(WinDS(tr), batch_size=BATCH_SIZE, shuffle=True)
        vl = DataLoader(WinDS(va), batch_size=BATCH_SIZE, shuffle=False)
        best, ctr, best_state, stop_ep = float("inf"), 0, None, args.max_epochs
        for ep in range(1, args.max_epochs + 1):
            trl = run_epoch(model, tl, opt)
            vll = run_epoch(model, vl, None)
            if ep == 1 or ep % 10 == 0:
                print("   ep {:3d} train {:.5f} val {:.5f}".format(ep, trl, vll))
                sys.stdout.flush()
            if vll < best - 1e-6:
                best, ctr = vll, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                ctr += 1
                if ctr >= PATIENCE:
                    stop_ep = ep
                    break
        model.load_state_dict(best_state)
        s = window_scores(model, te)
        fpr, tpr, _ = roc_curve(y_te, s)
        fold_auc = float(auc(fpr, tpr))
        fold_pr = float(average_precision_score(y_te, s))
        acts = np.array(act_te)
        safe_s = s[y_te == 0]
        ab = {}
        for a in sorted(set(acts[y_te == 1])):
            m = (acts == a) & (y_te == 1)
            yy = np.concatenate([np.zeros(len(safe_s)), np.ones(int(m.sum()))])
            ss = np.concatenate([safe_s, s[m]])
            f2, t2, _ = roc_curve(yy, ss)
            ab[a] = {"auc": float(auc(f2, t2)), "n_windows": int(m.sum())}
        res["folds"][test_driver] = {"auc": fold_auc, "pr_auc": fold_pr, "n_test_windows": int(len(y_te)),
                                     "anomaly_frac": float(y_te.mean()), "epochs": stop_ep,
                                     "n_params": n_params, "actions": ab,
                                     "scores_mean_safe": float(safe_s.mean()), "scores_std_safe": float(safe_s.std()),
                                     "seconds": round(time.time() - t0, 1)}
        print(">> FOLD {} {} AUC={:.4f} PR-AUC={:.4f} epochs={} ({:.0f}s)".format(
            fold_idx + 1, test_driver, fold_auc, fold_pr, stop_ep, time.time() - t0))
        sys.stdout.flush()
        aucs = [f["auc"] for f in res["folds"].values()]
        res["mean_auc"] = float(np.mean(aucs))
        res["std_auc"] = float(np.std(aucs))
        res["n_folds_done"] = len(aucs)
        json.dump(res, open(out_path, "w"), indent=1)

    print("\nDONE", name, "mean AUC {:.4f} +- {:.4f} over {} folds".format(
        res.get("mean_auc", float("nan")), res.get("std_auc", float("nan")), len(res["folds"])))


if __name__ == "__main__":
    main()
