"""LOSO (14 drivers) runs of the Mamba autoencoder with the other sets of features,
with the same protocol as paper_loso.py / train_loso_beta_mamba_real_v2.py:
  - test driver = all clips, val driver = safe clips only (early stopping),
    remaining 12 drivers = safe clips only (training); val_driver =
    remaining[fold_idx % 13]  (identical fold assignment to all earlier LOSO runs),
  - per-fold z-score standardization fitted on train windows only,
  - Adam lr 1e-3, batch 64, max 100 epochs, early stopping patience 10 on val MSE,
  - Dense window (60 frames, step 1).
The model (MambaAutoencoder, d_state 8) is imported from
training/train_loso_beta_mamba_real_v2.py, i.e. it is exactly the network of the
existing Geometry runs.

Feature sets: coords / geom / kin are derived from beta_raw_v2 exactly as in
paper_loso.py; hands = epsilon_raw_v2; hh = zeta_raw_v2. The master clip list is
always beta_raw_v2/meta.json; with --common the clips are restricted to those that
exist in all of beta/epsilon/zeta (as in train_loso_ensemble_v2.py), so that the
windows of geom/hands/hh line up one to one for the score-level ensemble.

Output (written after every fold, resumed from if it exists):
  paper_experiments/mamba_<feat>[_common]_seed<k>.json            results
  paper_experiments/scores/mamba_<feat>[_common]_seed<k>_<driver>.npz
      test scores, labels, actions, clip names and the mean/std of the scores of the
      val-safe windows (needed for the ensemble and for the pooled AUC)."""
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

HOME = "/home/fei/mik0542"
sys.path.insert(0, os.path.join(HOME, "training"))
from train_loso_beta_mamba_real_v2 import MambaAutoencoder  # noqa: E402

OUT_DIR = os.path.join(HOME, "paper_experiments")
SCORE_DIR = os.path.join(OUT_DIR, "scores")
RAW = {"beta": os.path.join(HOME, "beta_raw_v2"),
       "epsilon": os.path.join(HOME, "epsilon_raw_v2"),
       "zeta": os.path.join(HOME, "zeta_raw_v2")}
FEAT_SRC = {"coords": "beta", "geom": "beta", "kin": "beta", "hands": "epsilon", "hh": "zeta"}
SEQ_LEN = 60
BATCH_SIZE = 64
PATIENCE = 10
LR = 1e-3
V = 13

parser = argparse.ArgumentParser()
parser.add_argument("--feat", required=True, choices=sorted(FEAT_SRC))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--d_state", type=int, default=8)
parser.add_argument("--only_fold", type=int, default=-1)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--common", action="store_true")
parser.add_argument("--tag", default="")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_features(b, feat):
    """Identical to paper_loso.py for coords / geom / kin."""
    if feat in ("geom", "hands", "hh"):
        return b.astype(np.float32)
    T = b.shape[0]
    alpha = b[:, :39]
    if feat == "coords":
        return alpha.astype(np.float32)
    xy = alpha.reshape(T, V, 3)[:, :, :2]
    smoothed = savgol_filter(xy, window_length=5, polyorder=2, axis=0, mode="nearest") if T >= 5 else xy.copy()
    vel = np.gradient(smoothed, axis=0)
    acc = np.gradient(vel, axis=0)
    return np.concatenate([b, vel.reshape(T, -1), acc.reshape(T, -1)], axis=1).astype(np.float32)


def load_all():
    meta = json.load(open(os.path.join(RAW["beta"], "meta.json")))
    if args.common:
        keep = None
        for d in RAW.values():
            s = set(m["clip"] for m in json.load(open(os.path.join(d, "meta.json"))))
            keep = s if keep is None else keep & s
        meta = [m for m in meta if m["clip"] in keep]
    elif FEAT_SRC[args.feat] != "beta":
        s = set(m["clip"] for m in json.load(open(os.path.join(RAW[FEAT_SRC[args.feat]], "meta.json"))))
        meta = [m for m in meta if m["clip"] in s]
    src = RAW[FEAT_SRC[args.feat]]
    feats = {m["clip"]: build_features(np.load(os.path.join(src, m["clip"] + ".npy")), args.feat) for m in meta}
    return meta, feats


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


class WinDS(Dataset):
    def __init__(self, w):
        self.w = w

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        return torch.from_numpy(self.w[i].T)


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
            x = torch.from_numpy(np.ascontiguousarray(w[i:i + bs].transpose(0, 2, 1))).to(device)
            out.append(((model(x) - x) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def main():
    os.makedirs(SCORE_DIR, exist_ok=True)
    name = "mamba_{}{}{}_seed{}".format(args.feat, "_common" if args.common else "", args.tag, args.seed)
    base = "mamba_{}{}_seed{}".format(args.feat, "_common" if args.common else "", args.seed)
    out_path = os.path.join(OUT_DIR, name + ".json")
    meta, feats = load_all()
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    clips_by_driver = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}
    D = next(iter(feats.values())).shape[1]
    print("feat={} common={} D={} seed={} d_state={} drivers={} clips={} device={}".format(
        args.feat, args.common, D, args.seed, args.d_state, len(drivers), len(meta), device))
    sys.stdout.flush()

    res = {"exp": "mamba_" + args.feat, "seed": args.seed, "feature_set": args.feat, "model": "mamba",
           "d_state": args.d_state, "common_clips": bool(args.common), "seq_len": SEQ_LEN,
           "frame_stride": 1, "n_features": D, "n_clips": len(meta), "folds": {}}
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
        flat = tr.reshape(-1, D)
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std[std < 1e-6] = 1e-6
        for w in (tr, va, te):
            w -= mean
            w /= std
        print("\nFOLD {}/{} test={} val={} | train {} val {} test {} (anomaly frac {:.3f})".format(
            fold_idx + 1, len(drivers), test_driver, val_driver, len(tr), len(va), len(te), y_te.mean()))
        sys.stdout.flush()

        model = MambaAutoencoder(D, d_state=args.d_state).to(device)
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
            ss = np.concatenate([safe_s, s[m]])
            f2, t2, _ = roc_curve(yy, ss)
            ab[a] = {"auc": float(auc(f2, t2)), "n_windows": int(m.sum())}
        np.savez_compressed(os.path.join(SCORE_DIR, "{}_{}.npz".format(base, test_driver)),
                            scores=s.astype(np.float32), labels=y_te.astype(np.int8), actions=acts,
                            clips=np.array(clip_te), val_mean=float(s_val.mean()),
                            val_std=float(max(s_val.std(), 1e-8)))
        res["folds"][test_driver] = {"auc": fold_auc, "pr_auc": fold_pr, "n_test_windows": int(len(y_te)),
                                     "anomaly_frac": float(y_te.mean()), "epochs": stop_ep,
                                     "n_params": n_params, "actions": ab,
                                     "scores_mean_safe": float(safe_s.mean()), "scores_std_safe": float(safe_s.std()),
                                     "val_mean": float(s_val.mean()), "val_std": float(s_val.std()),
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
