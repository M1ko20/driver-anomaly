"""Ablation LOSO (14 drivers) runs of the TCN autoencoder for the article. Same protocol as
paper_extra.py / paper_loso.py:
  - test driver = all clips, val driver = safe clips only (early stopping),
    remaining 12 drivers = safe clips only (training); val_driver = remaining[fold_idx % 13],
  - per-fold z-score standardization fitted on the train windows only,
  - Adam lr 1e-3, batch 64, max 100 epochs, early stopping patience 10 on val MSE,
  - windows of 60 consecutive frames, step 1; only clips with at least 60 frames.

Differences from paper_extra.py (they do not change the protocol):
  - the windows are not materialized; a batch is gathered from the per-frame feature
    matrix by the start indices of its windows (the standardization uses the same
    statistics as over all train windows, i.e. every frame is weighted by the number of
    windows that contain it),
  - --device cpu|cuda; on a GPU the data stay in the main memory and only the batches are
    moved, and the process is limited to --gpu_mem_frac of the GPU memory (the GPUs are
    shared with other users).

Feature sets (all from the 13 normalized keypoints in beta_raw_v2)
  geom        Geometry (52): x, y, confidence + 8 distances + 5 angles     [reference]
  exh         Exhaustive (364): 78 pairwise distances + 286 triplet angles  [reference]
  coords_nc   x, y of the keypoints, no confidences (26)
  geom_nc     Geometry without the confidences (39)
  geom_da     only the 8 distances + 5 angles of Geometry (13)
  dist78      only the 78 pairwise distances
  ang286      only the 286 triplet angles
  exh_conf    Exhaustive + the 13 confidences (377)
  exh_xy      Exhaustive + x, y of the keypoints (390)

Model: TCN autoencoder (64-32-Z / 32-64 channels), Z = --latent, kernel size = --kernel
(5 in the article; 1 gives a network that reconstructs every frame independently).

Additional test-time controls (stored for every fold):
  auc_static   every test window is replaced by its middle frame repeated 60 times
  auc_shuffle  the frames of every test window are randomly permuted

Output (after every fold, resumable):
  paper_experiments/a_<name>_seed<k>.json   (with --only_fold F: a_<name>_seed<k>.f<FF>.json,
      one file per fold, so that the folds of one run can be computed in parallel)
  paper_experiments/scores/a_<name>_seed<k>_<driver>.npz  (test scores, labels, actions, clips,
      start frame of each window in its clip, static/shuffle scores, statistics of the
      scores of the val-safe windows)
  paper_experiments/models/a_<name>_seed<k>_<driver>.pt   (--save_models: weights + scaler)"""
import os
import sys
import json
import time
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, average_precision_score

HOME = "/home/fei/mik0542"
OUT_DIR = os.path.join(HOME, "paper_experiments")
SCORE_DIR = os.path.join(OUT_DIR, "scores")
MODEL_DIR = os.path.join(OUT_DIR, "models")
RAW_DIR = os.path.join(HOME, "beta_raw_v2")
FEATS = ["geom", "exh", "coords_nc", "geom_nc", "geom_da", "dist78", "ang286", "exh_conf", "exh_xy"]
SEQ_LEN = 60
BATCH_SIZE = 64
PATIENCE = 10
LR = 1e-3
V = 13

parser = argparse.ArgumentParser()
parser.add_argument("--feat", required=True, choices=FEATS)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--latent", type=int, default=16)
parser.add_argument("--kernel", type=int, default=5)
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--gpu_mem_frac", type=float, default=0.11)
parser.add_argument("--only_fold", type=int, default=-1)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--save_models", action="store_true")
parser.add_argument("--tag", default="")
args = parser.parse_args()
torch.set_num_threads(args.threads)
device = torch.device(args.device)
if args.device == "cuda":
    torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
EVAL_BS = 1024 if args.device == "cpu" else 256


# ------------------------------------------------------------------ features
def dist(p1, p2):
    return np.sqrt(((p1 - p2) ** 2).sum(axis=-1))


def angle_at(vertex, p1, p2, eps=1e-8):
    v1, v2 = p1 - vertex, p2 - vertex
    cos_a = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + eps)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


PAIRS = list(itertools.combinations(range(V), 2))
TRIPLES = list(itertools.combinations(range(V), 3))


def build_exh(xy):
    """Identical to build_exh of paper_extra.py."""
    d = np.stack([dist(xy[:, i], xy[:, j]) for i, j in PAIRS], axis=1)
    a = np.stack([angle_at(xy[:, j], xy[:, i], xy[:, k]) for i, j, k in TRIPLES], axis=1)
    return d, a


def build_features(b, feat):
    T = b.shape[0]
    kp = b[:, :39].reshape(T, V, 3)
    xy, conf = kp[:, :, :2], kp[:, :, 2]
    if feat == "geom":
        out = b
    elif feat == "coords_nc":
        out = xy.reshape(T, -1)
    elif feat == "geom_nc":
        out = np.concatenate([xy.reshape(T, -1), b[:, 39:]], axis=1)
    elif feat == "geom_da":
        out = b[:, 39:]
    else:
        d, a = build_exh(xy)
        if feat == "exh":
            out = np.concatenate([d, a], axis=1)
        elif feat == "dist78":
            out = d
        elif feat == "ang286":
            out = a
        elif feat == "exh_conf":
            out = np.concatenate([d, a, conf], axis=1)
        else:
            out = np.concatenate([d, a, xy.reshape(T, -1)], axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


class WindowSet:
    """Frames of a set of clips in one matrix + the start index of every window."""

    def __init__(self, clips, meta_by_clip, feats, label_filter=None):
        mats, starts, labels, actions, names, offs = [], [], [], [], [], []
        pos = 0
        for c in clips:
            m = meta_by_clip[c]
            if label_filter is not None and m["label"] != label_filter:
                continue
            f = feats[c]
            T = f.shape[0]
            if T < SEQ_LEN:
                continue
            n = T - SEQ_LEN + 1
            mats.append(f)
            starts.append(pos + np.arange(n))
            offs.append(np.arange(n))
            labels += [1 if m["label"] == "anomaly" else 0] * n
            actions += [m["action"]] * n
            names += [c] * n
            pos += T
        self.n = len(labels)
        if self.n:
            self.X = np.concatenate(mats, axis=0)
            self.starts = np.concatenate(starts)
            self.offsets = np.concatenate(offs)
        self.labels = np.array(labels)
        self.actions = np.array(actions)
        self.clips = np.array(names)

    def window_stats(self):
        """Mean and std of every feature over all windows (frames weighted by the number
        of windows that contain them), as in paper_extra.py."""
        w = np.zeros(self.X.shape[0], np.float64)
        np.add.at(w, (self.starts[:, None] + np.arange(SEQ_LEN)[None, :]).ravel(), 1.0)
        X = self.X.astype(np.float64)
        mean = (X * w[:, None]).sum(axis=0) / w.sum()
        var = (((X - mean) ** 2) * w[:, None]).sum(axis=0) / w.sum()
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

    def standardize(self, mean, std):
        self.X = (self.X - mean) / std
        self.Xt = torch.from_numpy(self.X)
        self.st = torch.from_numpy(self.starts)

    def batch(self, idx):
        """idx: tensor of window indices -> (N, D, T)"""
        fr = self.st[idx][:, None] + ARANGE[None, :]
        return self.Xt[fr].transpose(1, 2).to(device)


ARANGE = torch.arange(SEQ_LEN)


# -------------------------------------------------------------------- model
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, k):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=dilation * (k - 1) // 2, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TCNAutoencoder(nn.Module):
    """Input (N, D, T). For kernel=5 identical to the TCN autoencoder of paper_extra.py."""

    def __init__(self, n_features, latent=16, k=5):
        super().__init__()
        self.encoder = nn.Sequential(TCNBlock(n_features, 64, 1, k), TCNBlock(64, 32, 2, k), TCNBlock(32, latent, 4, k))
        self.decoder = nn.Sequential(TCNBlock(latent, 32, 4, k), TCNBlock(32, 64, 2, k),
                                     nn.Conv1d(64, n_features, kernel_size=k, padding=(k - 1) // 2))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_epoch(model, ws, opt, gen):
    model.train()
    crit = nn.MSELoss()
    perm = torch.randperm(ws.n, generator=gen)
    tot = 0.0
    for i in range(0, ws.n, BATCH_SIZE):
        x = ws.batch(perm[i:i + BATCH_SIZE])
        loss = crit(model(x), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item() * x.shape[0]
    return tot / ws.n


def window_scores(model, ws, mode="normal", gen=None):
    """Reconstruction MSE of every window. mode: normal | static | shuffle."""
    model.eval()
    out, bs = [], EVAL_BS
    with torch.no_grad():
        for i in range(0, ws.n, bs):
            x = ws.batch(torch.arange(i, min(i + bs, ws.n)))
            if mode == "static":
                x = x[:, :, SEQ_LEN // 2:SEQ_LEN // 2 + 1].expand(-1, -1, SEQ_LEN).contiguous()
            elif mode == "shuffle":
                p = torch.argsort(torch.rand(x.shape[0], SEQ_LEN, generator=gen), dim=1)
                x = torch.gather(x, 2, p.to(device)[:, None, :].expand(-1, x.shape[1], -1))
            out.append(((model(x) - x) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def roc_auc(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(auc(fpr, tpr))


def main():
    os.makedirs(SCORE_DIR, exist_ok=True)
    base = "a_tcn_{}".format(args.feat)
    if args.latent != 16:
        base += "_z{}".format(args.latent)
    if args.kernel != 5:
        base += "_k{}".format(args.kernel)
    base += args.tag
    base_seed = "{}_seed{}".format(base, args.seed)
    suffix = ".f{:02d}.json".format(args.only_fold) if args.only_fold >= 0 else ".json"
    out_path = os.path.join(OUT_DIR, base_seed + suffix)

    meta = json.load(open(os.path.join(RAW_DIR, "meta.json")))
    feats = {m["clip"]: build_features(np.load(os.path.join(RAW_DIR, m["clip"] + ".npy")), args.feat) for m in meta}
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    cbd = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}
    D = next(iter(feats.values())).shape[1]
    print("{} D={} latent={} kernel={} seed={} drivers={} clips={} device={}".format(
        base, D, args.latent, args.kernel, args.seed, len(drivers), len(meta), args.device))
    sys.stdout.flush()

    res = {"exp": base, "seed": args.seed, "feature_set": args.feat, "model": "tcn", "seq_len": SEQ_LEN,
           "latent": args.latent, "kernel": args.kernel, "n_features": D, "n_clips": len(meta),
           "device": args.device, "folds": {}}
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
        gen = torch.Generator().manual_seed(args.seed * 1000 + fold_idx)

        tr = WindowSet([c for d in train_drivers for c in cbd[d]], meta_by_clip, feats, "safe")
        va = WindowSet(cbd[val_driver], meta_by_clip, feats, "safe")
        te = WindowSet(cbd[test_driver], meta_by_clip, feats)
        mean, std = tr.window_stats()
        std[std < 1e-6] = 1e-6
        for ws in (tr, va, te):
            ws.standardize(mean, std)
        print("\nFOLD {}/{} test={} val={} | train {} val {} test {} (anomaly frac {:.3f})".format(
            fold_idx + 1, len(drivers), test_driver, val_driver, tr.n, va.n, te.n, te.labels.mean()))
        sys.stdout.flush()

        model = TCNAutoencoder(D, args.latent, args.kernel).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        best, ctr, best_state, stop_ep = float("inf"), 0, None, args.max_epochs
        for ep in range(1, args.max_epochs + 1):
            trl = train_epoch(model, tr, opt, gen)
            vll = float(window_scores(model, va).mean())
            if ep == 1 or ep % 10 == 0:
                print("   ep {:3d} train {:.5f} val {:.5f} ({:.0f}s)".format(ep, trl, vll, time.time() - t0))
                sys.stdout.flush()
            if vll < best - 1e-6:
                best, ctr = vll, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                ctr += 1
                if ctr >= PATIENCE:
                    stop_ep = ep
                    break
        model.load_state_dict(best_state)
        if args.save_models:
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save({"state": best_state, "mean": mean, "std": std, "n_features": D, "latent": args.latent,
                        "kernel": args.kernel, "feat": args.feat, "val_driver": val_driver},
                       os.path.join(MODEL_DIR, "{}_{}.pt".format(base_seed, test_driver)))

        y = te.labels
        s = window_scores(model, te)
        s_static = window_scores(model, te, "static")
        s_shuf = window_scores(model, te, "shuffle", gen)
        s_val = window_scores(model, va)
        acts = te.actions
        safe_s = s[y == 0]
        ab = {}
        for a in sorted(set(acts[y == 1])):
            m = (acts == a) & (y == 1)
            yy = np.concatenate([np.zeros(len(safe_s)), np.ones(int(m.sum()))])
            ab[a] = {"auc": roc_auc(yy, np.concatenate([safe_s, s[m]])), "n_windows": int(m.sum())}
        vstat = {"val_mean": float(s_val.mean()), "val_std": float(max(s_val.std(), 1e-8)),
                 "val_p95": float(np.percentile(s_val, 95)), "val_p99": float(np.percentile(s_val, 99))}
        np.savez_compressed(os.path.join(SCORE_DIR, "{}_{}.npz".format(base_seed, test_driver)),
                            scores=s.astype(np.float32), labels=y.astype(np.int8), actions=acts, clips=te.clips,
                            offsets=te.offsets.astype(np.int32), scores_static=s_static.astype(np.float32),
                            scores_shuffle=s_shuf.astype(np.float32), **vstat)
        fold = {"auc": roc_auc(y, s), "pr_auc": float(average_precision_score(y, s)),
                "auc_static": roc_auc(y, s_static), "auc_shuffle": roc_auc(y, s_shuf),
                "macro_action_auc": float(np.mean([v["auc"] for v in ab.values()])),
                "n_test_windows": int(len(y)), "anomaly_frac": float(y.mean()), "epochs": stop_ep,
                "n_params": n_params, "actions": ab, "scores_mean_safe": float(safe_s.mean()),
                "scores_std_safe": float(safe_s.std()), "scores_mean_safe_static": float(s_static[y == 0].mean()),
                "scores_mean_safe_shuffle": float(s_shuf[y == 0].mean()), "seconds": round(time.time() - t0, 1)}
        fold.update(vstat)
        res["folds"][test_driver] = fold
        print(">> FOLD {} {} AUC={:.4f} PR={:.4f} static={:.4f} shuffle={:.4f} epochs={} params={} ({:.0f}s)".format(
            fold_idx + 1, test_driver, fold["auc"], fold["pr_auc"], fold["auc_static"], fold["auc_shuffle"],
            stop_ep, n_params, time.time() - t0))
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
