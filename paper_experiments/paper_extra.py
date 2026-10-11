"""Additional LOSO (14 drivers) experiments for the article. Same protocol as
paper_loso.py / train_loso_beta_v2.py:
  - test driver = all clips, val driver = safe clips only (early stopping),
    remaining 12 drivers = safe clips only (training); val_driver =
    remaining[fold_idx % 13],
  - per-fold z-score standardization fitted on train windows only,
  - Adam lr 1e-3, batch 64, max 100 epochs, early stopping patience 10 on val MSE,
  - windows of --seq_len consecutive frames (default 60), step 1.

Models
  tcn   TCN autoencoder of the article (64-32-Z / 32-64 channels, Z = --latent, default 16)
  lstm  LSTM_extend autoencoder of the CISIM 2025 paper (Fig. 3a there): 2x time-distributed
        dense (768, 384) + batch norm, 2x bidirectional LSTM (2x192) with dropout 0.2 and
        batch norm, dense latent vector of size 96, repeat vector, 2x bidirectional LSTM,
        time-distributed dense 768 + batch norm, time-distributed dense output.
        (The activations of the dense layers are not given in the paper; ReLU is used.)

Feature sets
  coords, geom, kin   from beta_raw_v2 exactly as in paper_loso.py
  proxies             delta_raw_v2          hands   epsilon_raw_v2        hh   zeta_raw_v2
  exh                 exhaustive features as in the CISIM 2025 paper, computed from the 13
                      normalized keypoints: all 78 pairwise distances + one angle for each of
                      the 286 triplets (angle at the middle keypoint of the triplet) = 364
  vit_coords, vit_geom   Coordinates / Geometry computed from the 13 body keypoints of the
                      ViTPose-B wholebody detections (pose_hands_v2), normalized in the same
                      way as the YOLO26-pose skeleton (normalize_skeleton_core.py).

--common restricts the clips to those present in beta/epsilon/zeta (as in
train_loso_ensemble_v2.py), so that the windows of different feature sets line up.
--min_clip_len L uses only clips with at least L frames (to compare window lengths on
the same clips).

Output (after every fold, resumable):
  paper_experiments/x_<name>_seed<k>.json
  paper_experiments/scores/x_<name>_seed<k>_<driver>.npz  (test scores, labels, actions,
      clips, and mean/std/95th/99th percentile of the scores of the val-safe windows)"""
import os
import sys
import json
import time
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy.signal import savgol_filter

HOME = "/home/fei/mik0542"
OUT_DIR = os.path.join(HOME, "paper_experiments")
SCORE_DIR = os.path.join(OUT_DIR, "scores")
RAW = {"beta": os.path.join(HOME, "beta_raw_v2"), "delta": os.path.join(HOME, "delta_raw_v2"),
       "epsilon": os.path.join(HOME, "epsilon_raw_v2"), "zeta": os.path.join(HOME, "zeta_raw_v2")}
VIT_DIR = os.path.join(HOME, "pose_hands_v2")
FEAT_SRC = {"coords": "beta", "geom": "beta", "kin": "beta", "exh": "beta", "proxies": "delta",
            "hands": "epsilon", "hh": "zeta", "vit_coords": "vit", "vit_geom": "vit",
            "geom_da": "beta"}   # geom_da: only the 8 distances + 5 angles of Geometry (Selected, 13)
PATIENCE = 10
LR = 1e-3
V = 13

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["tcn", "lstm"])
parser.add_argument("--feat", required=True, choices=sorted(FEAT_SRC))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--seq_len", type=int, default=60)
parser.add_argument("--latent", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--min_clip_len", type=int, default=0)
parser.add_argument("--common", action="store_true")
parser.add_argument("--only_fold", type=int, default=-1)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--tag", default="")
parser.add_argument("--gpu_mem_frac", type=float, default=0.0)
args = parser.parse_args()
SEQ_LEN = args.seq_len
BATCH_SIZE = args.batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.gpu_mem_frac > 0 and device.type == "cuda":   # the GPUs are shared with other users
    torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)


# ------------------------------------------------------------------ features
def dist(p1, p2):
    return np.sqrt(((p1 - p2) ** 2).sum(axis=-1))


def angle_at(vertex, p1, p2, eps=1e-8):
    v1, v2 = p1 - vertex, p2 - vertex
    cos_a = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + eps)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def build_beta(arr):
    """(T, 13, 3) normalized keypoints -> Geometry (52). Copy of build_beta_raw_v2.py."""
    xy = arr[:, :, :2]
    T = arr.shape[0]
    nose, l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip = (xy[:, i] for i in (0, 5, 6, 7, 8, 9, 10, 11, 12))
    neck = np.zeros_like(nose)
    up = np.tile(np.array([0.0, -1.0]), (T, 1))
    d = np.stack([dist(nose, l_wr), dist(nose, r_wr), dist(l_sh, l_wr), dist(r_sh, r_wr),
                  dist(l_hip, l_wr), dist(r_hip, r_wr), dist(l_wr, r_wr), dist(nose, neck)], axis=1)
    a = np.stack([angle_at(l_el, l_sh, l_wr), angle_at(r_el, r_sh, r_wr), angle_at(l_sh, l_el, l_hip),
                  angle_at(r_sh, r_el, r_hip), angle_at(neck, nose, neck + up)], axis=1)
    return np.concatenate([arr.reshape(T, -1), d, a], axis=1).astype(np.float32)


PAIRS = list(itertools.combinations(range(V), 2))
TRIPLES = list(itertools.combinations(range(V), 3))


def build_exh(b):
    T = b.shape[0]
    xy = b[:, :39].reshape(T, V, 3)[:, :, :2]
    d = np.stack([dist(xy[:, i], xy[:, j]) for i, j in PAIRS], axis=1)
    a = np.stack([angle_at(xy[:, j], xy[:, i], xy[:, k]) for i, j, k in TRIPLES], axis=1)
    return np.concatenate([d, a], axis=1).astype(np.float32)


def load_vit(clip):
    """ViTPose wholebody detections (T,133,3) in (y,x,score) -> (T,13,3) normalized
    (x, y, score) as in normalize_skeleton_core.py (neck to origin, shoulder width = 1,
    fallback torso length; frames without a usable scale are filled from the last valid one)."""
    raw = np.load(os.path.join(VIT_DIR, clip + ".npy")).astype(np.float32)
    T = raw.shape[0]
    xy = raw[:, :V, [1, 0]]
    conf = raw[:, :V, 2]
    ls, rs, lh, rh = xy[:, 5], xy[:, 6], xy[:, 11], xy[:, 12]
    neck = (ls + rs) / 2.0
    width = dist(ls, rs)
    torso = dist(neck, (lh + rh) / 2.0)
    ok_sh = (conf[:, 5] >= 0.1) & (conf[:, 6] >= 0.1) & (width >= 5.0)
    scale = np.where(ok_sh, width, torso)
    valid = ok_sh | (torso >= 5.0)
    arr = np.zeros((T, V, 3), np.float32)
    arr[:, :, :2] = (xy - neck[:, None, :]) / np.maximum(scale, 1e-6)[:, None, None]
    arr[:, :, 2] = conf
    if not valid.any():
        return None
    last = None
    for t in range(T):
        if valid[t]:
            last = arr[t].copy()
        elif last is not None:
            arr[t] = last
    first = int(np.argmax(valid))
    arr[:first] = arr[first]
    return arr


def build_features(clip, feat):
    src = FEAT_SRC[feat]
    if src == "vit":
        arr = load_vit(clip)
        if arr is None:
            return None
        return arr.reshape(arr.shape[0], -1) if feat == "vit_coords" else build_beta(arr)
    b = np.load(os.path.join(RAW[src], clip + ".npy"))
    if feat in ("geom", "proxies", "hands", "hh"):
        return b.astype(np.float32)
    if feat == "coords":
        return b[:, :39].astype(np.float32)
    if feat == "exh":
        return build_exh(b)
    if feat == "geom_da":
        return np.ascontiguousarray(b[:, 39:], dtype=np.float32)
    T = b.shape[0]
    xy = b[:, :39].reshape(T, V, 3)[:, :, :2]
    smoothed = savgol_filter(xy, window_length=5, polyorder=2, axis=0, mode="nearest") if T >= 5 else xy.copy()
    vel = np.gradient(smoothed, axis=0)
    acc = np.gradient(vel, axis=0)
    return np.concatenate([b, vel.reshape(T, -1), acc.reshape(T, -1)], axis=1).astype(np.float32)


def load_all():
    meta = json.load(open(os.path.join(RAW["beta"], "meta.json")))
    need = ["beta", "epsilon", "zeta"] if args.common else [FEAT_SRC[args.feat]]
    for n in need:
        if n in RAW and n != "beta":
            s = set(m["clip"] for m in json.load(open(os.path.join(RAW[n], "meta.json"))))
            meta = [m for m in meta if m["clip"] in s]
    feats, kept = {}, []
    for m in meta:
        f = build_features(m["clip"], args.feat)
        if f is None or f.shape[0] < args.min_clip_len:
            continue
        feats[m["clip"]] = f
        kept.append(m)
    return kept, feats


def make_windows(clips, meta_by_clip, feats, label_filter=None):
    ws, labels, actions, names = [], [], [], []
    for c in clips:
        m = meta_by_clip[c]
        if label_filter is not None and m["label"] != label_filter:
            continue
        f = feats[c]
        T = f.shape[0]
        if T < SEQ_LEN:
            continue
        for start in range(0, T - SEQ_LEN + 1):
            ws.append(f[start:start + SEQ_LEN])
            labels.append(1 if m["label"] == "anomaly" else 0)
            actions.append(m["action"])
            names.append(c)
    D = next(iter(feats.values())).shape[1]
    if not ws:
        return np.zeros((0, SEQ_LEN, D), np.float32), np.array([]), [], []
    return np.stack(ws).astype(np.float32), np.array(labels), actions, names


# -------------------------------------------------------------------- models
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=dilation * 2, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.shape[2]] if out.shape[2] > x.shape[2] else out
        return self.act(self.bn(out))


class TCNAutoencoder(nn.Module):
    """Input (N, T, D). Identical to the TCN autoencoder of paper_loso.py for latent=16."""

    def __init__(self, n_features, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(TCNBlock(n_features, 64, 1), TCNBlock(64, 32, 2), TCNBlock(32, latent, 4))
        self.decoder = nn.Sequential(TCNBlock(latent, 32, 4), TCNBlock(32, 64, 2),
                                     nn.Conv1d(64, n_features, kernel_size=5, padding=2))

    def forward(self, x):
        return self.decoder(self.encoder(x.transpose(1, 2))).transpose(1, 2)


class TDBN(nn.Module):
    """Batch normalization over the feature axis of (N, T, C)."""

    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class LSTMExtendAutoencoder(nn.Module):
    def __init__(self, n_features, seq_len, hidden=192, latent=96, p=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.pre = nn.Sequential(nn.Linear(n_features, 768), nn.ReLU(), TDBN(768),
                                 nn.Linear(768, 384), nn.ReLU(), TDBN(384))
        self.enc1 = nn.LSTM(384, hidden, batch_first=True, bidirectional=True)
        self.enc1_post = nn.Sequential(nn.Dropout(p), TDBN(2 * hidden))
        self.enc2 = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.enc2_post = nn.Sequential(nn.Dropout(p), nn.BatchNorm1d(2 * hidden))
        self.to_latent = nn.Linear(2 * hidden, latent)
        self.dec1 = nn.LSTM(latent, hidden, batch_first=True, bidirectional=True)
        self.dec1_post = nn.Sequential(nn.Dropout(p), TDBN(2 * hidden))
        self.dec2 = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.dec2_post = nn.Sequential(nn.Dropout(p), TDBN(2 * hidden))
        self.post = nn.Sequential(nn.Linear(2 * hidden, 768), nn.ReLU(), TDBN(768), nn.Linear(768, n_features))

    def forward(self, x):
        h = self.pre(x)
        h, _ = self.enc1(h)
        h = self.enc1_post(h)
        _, (hn, _) = self.enc2(h)
        z = self.to_latent(self.enc2_post(torch.cat([hn[0], hn[1]], dim=1)))
        h = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        h, _ = self.dec1(h)
        h = self.dec1_post(h)
        h, _ = self.dec2(h)
        h = self.dec2_post(h)
        return self.post(h)


def make_model(D):
    if args.model == "tcn":
        return TCNAutoencoder(D, args.latent)
    return LSTMExtendAutoencoder(D, SEQ_LEN)


class WinDS(Dataset):
    def __init__(self, w):
        self.w = w

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        return torch.from_numpy(self.w[i])


def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    crit = nn.MSELoss()
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x in loader:
            x = x.to(device)
            loss = crit(model(x), x)
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
            x = torch.from_numpy(np.ascontiguousarray(w[i:i + bs])).to(device)
            out.append(((model(x) - x) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def main():
    os.makedirs(SCORE_DIR, exist_ok=True)
    base = "x_{}_{}".format(args.model, args.feat)
    if args.common:
        base += "_common"
    if SEQ_LEN != 60:
        base += "_T{}".format(SEQ_LEN)
    if args.min_clip_len:
        base += "_minL{}".format(args.min_clip_len)
    if args.model == "tcn" and args.latent != 16:
        base += "_z{}".format(args.latent)
    if BATCH_SIZE != 64:
        base += "_b{}".format(BATCH_SIZE)
    base_seed = "{}_seed{}".format(base, args.seed)
    out_path = os.path.join(OUT_DIR, "{}{}_seed{}.json".format(base, args.tag, args.seed))
    meta, feats = load_all()
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    clips_by_driver = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}
    D = next(iter(feats.values())).shape[1]
    print("{} D={} seq_len={} latent={} seed={} drivers={} clips={} device={}".format(
        base, D, SEQ_LEN, args.latent, args.seed, len(drivers), len(meta), device))
    sys.stdout.flush()

    res = {"exp": base, "seed": args.seed, "feature_set": args.feat, "model": args.model,
           "seq_len": SEQ_LEN, "latent": args.latent, "common_clips": bool(args.common),
           "min_clip_len": args.min_clip_len, "n_features": D, "n_clips": len(meta), "folds": {}}
    if os.path.exists(out_path):
        res = json.load(open(out_path))
        print("Resuming, done folds:", sorted(res["folds"]))

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

        tr, _, _, _ = make_windows([c for d in train_drivers for c in clips_by_driver[d]], meta_by_clip, feats, "safe")
        va, _, _, _ = make_windows(clips_by_driver[val_driver], meta_by_clip, feats, "safe")
        te, y_te, act_te, clip_te = make_windows(clips_by_driver[test_driver], meta_by_clip, feats)
        if len(tr) == 0 or len(va) == 0 or len(te) == 0 or y_te.min() == y_te.max():
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
                print("   ep {:3d} train {:.5f} val {:.5f} ({:.0f}s)".format(ep, trl, vll, time.time() - t0))
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
        s_val = window_scores(model, va)
        fpr, tpr, _ = roc_curve(y_te, s)
        fold_auc = float(auc(fpr, tpr))
        fold_pr = float(average_precision_score(y_te, s))
        acts = np.array(act_te)
        safe_s = s[y_te == 0]
        ab = {}
        for a in sorted(set(acts[y_te == 1])):
            m = (acts == a) & (y_te == 1)
            yy = np.concatenate([np.zeros(len(safe_s)), np.ones(int(m.sum()))])
            f2, t2, _ = roc_curve(yy, np.concatenate([safe_s, s[m]]))
            ab[a] = {"auc": float(auc(f2, t2)), "n_windows": int(m.sum())}
        vstat = {"val_mean": float(s_val.mean()), "val_std": float(max(s_val.std(), 1e-8)),
                 "val_p95": float(np.percentile(s_val, 95)), "val_p99": float(np.percentile(s_val, 99))}
        np.savez_compressed(os.path.join(SCORE_DIR, "{}_{}.npz".format(base_seed, test_driver)),
                            scores=s.astype(np.float32), labels=y_te.astype(np.int8), actions=acts,
                            clips=np.array(clip_te), **vstat)
        fold = {"auc": fold_auc, "pr_auc": fold_pr, "n_test_windows": int(len(y_te)),
                "anomaly_frac": float(y_te.mean()), "epochs": stop_ep, "n_params": n_params, "actions": ab,
                "scores_mean_safe": float(safe_s.mean()), "scores_std_safe": float(safe_s.std()),
                "seconds": round(time.time() - t0, 1)}
        fold.update(vstat)
        res["folds"][test_driver] = fold
        print(">> FOLD {} {} AUC={:.4f} PR-AUC={:.4f} epochs={} params={} ({:.0f}s)".format(
            fold_idx + 1, test_driver, fold_auc, fold_pr, stop_ep, n_params, time.time() - t0))
        sys.stdout.flush()
        aucs = [f["auc"] for f in res["folds"].values()]
        res["mean_auc"] = float(np.mean(aucs))
        res["std_auc"] = float(np.std(aucs))
        res["n_folds_done"] = len(aucs)
        json.dump(res, open(out_path, "w"), indent=1)

    print("\nDONE", base_seed, "mean AUC {:.4f} +- {:.4f} over {} folds".format(
        res.get("mean_auc", float("nan")), res.get("std_auc", float("nan")), len(res["folds"])))


if __name__ == "__main__":
    main()
