"""Classical one-class methods on window-level summaries of the per-frame features, under
the same 14-fold LOSO protocol as the autoencoders (paper_extra.py / paper_ablation.py):
same folds, same 60-frame windows with step 1 from the clips with at least 60 frames,
standardization fitted on the training data only, training on the safe clips of the 12
train drivers (the validation driver is not used for training; its safe windows are only
scored, to obtain the same score statistics as for the autoencoders).

  --pool mean      a window is represented by the mean of its 60 frames (D values)
  --pool meanstd   mean and standard deviation over the 60 frames (2 D values)
  --pool frame     the model is trained on single frames and applied to every frame; the
                   score of a window is the mean of the scores of its frames
  --pool flat      the window is flattened into one vector (as in paper_baselines.py)
  --win T          length of the window (default 60); --min_clip_len L uses only the clips
                   with at least L frames (to compare window lengths on the same clips)

Methods (settings as in paper_baselines.py): PCA reconstruction (20 components),
One-Class SVM (RBF kernel, nu = --nu, gamma = --gamma_mult x 'scale') and Isolation Forest
(200 trees); OC-SVM and Isolation Forest are trained on a random subset of --subsample
training samples.

Feature sets: as in paper_ablation.py, plus coords (x, y, confidence; 39).

Output: paper_experiments/c_<feat>_<pool>[_nu..][_g..][_n..]_seed<k>.json and
        paper_experiments/scores/c_<...>_seed<k>_<method>_<driver>.npz (pca and ocsvm)"""
import os
import sys
import json
import time
import argparse
import itertools
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc, average_precision_score

HOME = "/home/fei/mik0542"
RAW_DIR = os.path.join(HOME, "beta_raw_v2")
OUT_DIR = os.path.join(HOME, "paper_experiments")
SCORE_DIR = os.path.join(OUT_DIR, "scores")
FEATS = ["coords", "geom", "exh", "coords_nc", "geom_nc", "geom_da", "dist78", "ang286", "exh_conf", "exh_xy"]
V = 13

parser = argparse.ArgumentParser()
parser.add_argument("--feat", required=True, choices=FEATS)
parser.add_argument("--pool", required=True, choices=["mean", "meanstd", "frame", "flat"])
parser.add_argument("--win", type=int, default=60)
parser.add_argument("--min_clip_len", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--nu", type=float, default=0.05)
parser.add_argument("--gamma_mult", type=float, default=1.0)
parser.add_argument("--subsample", type=int, default=8000)
parser.add_argument("--methods", default="pca,ocsvm,iforest")
args = parser.parse_args()
METHODS = args.methods.split(",")
SEQ_LEN = args.win

PAIRS = list(itertools.combinations(range(V), 2))
TRIPLES = list(itertools.combinations(range(V), 3))


def dist(p1, p2):
    return np.sqrt(((p1 - p2) ** 2).sum(axis=-1))


def angle_at(vertex, p1, p2, eps=1e-8):
    v1, v2 = p1 - vertex, p2 - vertex
    cos_a = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + eps)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def build_features(b, feat):
    """Identical to build_features of paper_ablation.py (+ coords)."""
    T = b.shape[0]
    kp = b[:, :39].reshape(T, V, 3)
    xy, conf = kp[:, :, :2], kp[:, :, 2]
    if feat == "geom":
        out = b
    elif feat == "coords":
        out = b[:, :39]
    elif feat == "coords_nc":
        out = xy.reshape(T, -1)
    elif feat == "geom_nc":
        out = np.concatenate([xy.reshape(T, -1), b[:, 39:]], axis=1)
    elif feat == "geom_da":
        out = b[:, 39:]
    else:
        d = np.stack([dist(xy[:, i], xy[:, j]) for i, j in PAIRS], axis=1)
        a = np.stack([angle_at(xy[:, j], xy[:, i], xy[:, k]) for i, j, k in TRIPLES], axis=1)
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
    return np.ascontiguousarray(out, dtype=np.float64)


def win_mean(x):
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    return (cs[SEQ_LEN:] - cs[:-SEQ_LEN]) / SEQ_LEN


def summarize(f):
    """Per-frame features of one clip (T, D) -> window-level samples (T-59, D')."""
    if args.pool == "flat":
        return np.stack([f[s:s + SEQ_LEN].reshape(-1) for s in range(f.shape[0] - SEQ_LEN + 1)])
    m = win_mean(f)
    if args.pool == "mean":
        return m
    var = np.maximum(win_mean(f * f) - m * m, 0.0)
    return np.concatenate([m, np.sqrt(var)], axis=1)


def collect(clips, meta_by_clip, feats, label_filter=None):
    """Returns samples (windows, or frames for --pool frame) and the window bookkeeping."""
    xs, labels, actions, names, offs = [], [], [], [], []
    for c in clips:
        m = meta_by_clip[c]
        if label_filter is not None and m["label"] != label_filter:
            continue
        f = feats[c]
        T = f.shape[0]
        if T < max(SEQ_LEN, args.min_clip_len):
            continue
        n = T - SEQ_LEN + 1
        xs.append(f if args.pool == "frame" else summarize(f))
        labels += [1 if m["label"] == "anomaly" else 0] * n
        actions += [m["action"]] * n
        names += [c] * n
        offs.append(np.arange(n))
    return xs, np.array(labels), np.array(actions), np.array(names), np.concatenate(offs)


def to_window_scores(frame_scores, xs):
    """Per-frame scores of the concatenated clips -> mean score of every window."""
    out, pos = [], 0
    for x in xs:
        s = frame_scores[pos:pos + len(x)]
        out.append(win_mean(s[:, None])[:, 0])
        pos += len(x)
    return np.concatenate(out)


def roc_auc(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(auc(fpr, tpr))


def metrics(y, s, acts):
    safe = s[y == 0]
    ab = {}
    for act in sorted(set(acts[y == 1])):
        m = (acts == act) & (y == 1)
        yy = np.concatenate([np.zeros(len(safe)), np.ones(int(m.sum()))])
        ab[act] = {"auc": roc_auc(yy, np.concatenate([safe, s[m]])), "n_windows": int(m.sum())}
    return {"auc": roc_auc(y, s), "pr_auc": float(average_precision_score(y, s)),
            "macro_action_auc": float(np.mean([v["auc"] for v in ab.values()])), "actions": ab}


def main():
    os.makedirs(SCORE_DIR, exist_ok=True)
    base = "c_{}_{}".format(args.feat, args.pool)
    if args.nu != 0.05:
        base += "_nu{:g}".format(args.nu)
    if args.gamma_mult != 1.0:
        base += "_g{:g}".format(args.gamma_mult)
    if args.subsample != 8000:
        base += "_n{}".format(args.subsample)
    if args.win != 60:
        base += "_T{}".format(args.win)
    if args.min_clip_len:
        base += "_minL{}".format(args.min_clip_len)
    base_seed = "{}_seed{}".format(base, args.seed)
    out_path = os.path.join(OUT_DIR, base_seed + ".json")
    meta = json.load(open(os.path.join(RAW_DIR, "meta.json")))
    meta_by_clip = {m["clip"]: m for m in meta}
    feats = {m["clip"]: build_features(np.load(os.path.join(RAW_DIR, m["clip"] + ".npy")), args.feat) for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    cbd = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}
    res = {"exp": base, "feat": args.feat, "pool": args.pool, "seed": args.seed, "nu": args.nu,
           "gamma_mult": args.gamma_mult, "subsample": args.subsample, "win": args.win,
           "min_clip_len": args.min_clip_len, "folds": {}}
    if os.path.exists(out_path):
        res = json.load(open(out_path))
    for fold_idx, test_driver in enumerate(drivers):
        if test_driver in res["folds"]:
            continue
        t0 = time.time()
        remaining = [d for d in drivers if d != test_driver]
        val_driver = remaining[fold_idx % len(remaining)]
        train_drivers = [d for d in remaining if d != val_driver]
        xtr, _, _, _, _ = collect([c for d in train_drivers for c in cbd[d]], meta_by_clip, feats, "safe")
        xva, _, _, _, _ = collect(cbd[val_driver], meta_by_clip, feats, "safe")
        xte, y, acts, names, offs = collect(cbd[test_driver], meta_by_clip, feats)
        tr = np.concatenate(xtr)
        mean, std = tr.mean(axis=0), tr.std(axis=0)
        std[std < 1e-6] = 1e-6
        tr = ((tr - mean) / std).astype(np.float32)
        te = ((np.concatenate(xte) - mean) / std).astype(np.float32)
        va = ((np.concatenate(xva) - mean) / std).astype(np.float32)
        rng = np.random.RandomState(args.seed * 1000 + fold_idx)
        sub = tr[rng.choice(len(tr), min(args.subsample, len(tr)), replace=False)]

        def finish(s_te, s_va):
            if args.pool == "frame":
                return to_window_scores(s_te, xte), to_window_scores(s_va, xva)
            return s_te, s_va

        fold = {"n_test_windows": int(len(y)), "anomaly_frac": float(y.mean()), "dim": int(tr.shape[1]),
                "n_train": int(len(tr))}
        scores = {}
        if "pca" in METHODS:
            pca = PCA(n_components=min(20, tr.shape[1] - 1), random_state=args.seed).fit(tr)
            rec = lambda x: ((x - pca.inverse_transform(pca.transform(x))) ** 2).mean(axis=1)
            scores["pca"] = finish(rec(te), rec(va))
        if "ocsvm" in METHODS:
            gamma = args.gamma_mult / (sub.shape[1] * sub.var())
            oc = OneClassSVM(kernel="rbf", nu=args.nu, gamma=gamma).fit(sub)
            scores["ocsvm"] = finish(-oc.decision_function(te), -oc.decision_function(va))
            fold["n_sv"] = int(len(oc.support_))
        if "iforest" in METHODS:
            iso = IsolationForest(n_estimators=200, random_state=args.seed, n_jobs=1).fit(sub)
            scores["iforest"] = finish(-iso.score_samples(te), -iso.score_samples(va))
        for mname, (s, sv) in scores.items():
            fold[mname] = metrics(y, s, acts)
            vstat = {"val_mean": float(sv.mean()), "val_std": float(max(sv.std(), 1e-8)),
                     "val_p95": float(np.percentile(sv, 95)), "val_p99": float(np.percentile(sv, 99))}
            fold[mname].update(vstat)
            if mname != "iforest":
                np.savez_compressed(os.path.join(SCORE_DIR, "{}_{}_{}.npz".format(base_seed, mname, test_driver)),
                                    scores=s.astype(np.float32), labels=y.astype(np.int8), actions=acts,
                                    clips=names, offsets=offs.astype(np.int32), **vstat)
        fold["seconds"] = round(time.time() - t0, 1)
        res["folds"][test_driver] = fold
        print(">> {} {} dim {} | {} ({:.0f}s)".format(base_seed, test_driver, tr.shape[1], "  ".join(
            "{} {:.4f}".format(k, fold[k]["auc"]) for k in scores), fold["seconds"]))
        sys.stdout.flush()
        for k in scores:
            res["mean_auc_" + k] = float(np.mean([f[k]["auc"] for f in res["folds"].values() if k in f]))
        json.dump(res, open(out_path, "w"), indent=1)
    print("DONE", base_seed, {k: round(v, 4) for k, v in res.items() if k.startswith("mean_auc")})


if __name__ == "__main__":
    main()
