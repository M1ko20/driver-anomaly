"""Classical baselines (PCA reconstruction, One-Class SVM, Isolation Forest) with the
Exhaustive set and with window-mean features, under the same 14-fold LOSO protocol as
paper_baselines.py (same folds, 60-frame windows with step 1, per-fold z-score
standardization from the train windows, training on the safe windows of the 12 train
drivers, PCA 20 components, OC-SVM RBF nu=0.05 gamma='scale' and Isolation Forest 200
trees, the last two on a random subset of 8000 train windows).

  --feat geom|exh     Geometry (52) or Exhaustive (364) per frame
  --pool flat|mean    flat: the window is flattened into one vector (as in paper_baselines.py);
                      mean: the window is represented by the mean of its frames (no temporal
                      information at all)

Output: paper_experiments/b_<feat>_<pool>_seed<k>.f<FF>.json (one file per fold)"""
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

RAW_DIR = "/home/fei/mik0542/beta_raw_v2"
OUT_DIR = "/home/fei/mik0542/paper_experiments"
SEQ_LEN = 60
V = 13
SUBSAMPLE = 8000

parser = argparse.ArgumentParser()
parser.add_argument("--feat", required=True, choices=["geom", "exh"])
parser.add_argument("--pool", required=True, choices=["flat", "mean"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--only_fold", type=int, required=True)
args = parser.parse_args()

PAIRS = list(itertools.combinations(range(V), 2))
TRIPLES = list(itertools.combinations(range(V), 3))


def dist(p1, p2):
    return np.sqrt(((p1 - p2) ** 2).sum(axis=-1))


def angle_at(vertex, p1, p2, eps=1e-8):
    v1, v2 = p1 - vertex, p2 - vertex
    cos_a = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + eps)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def build_features(b):
    if args.feat == "geom":
        return b.astype(np.float32)
    T = b.shape[0]
    xy = b[:, :39].reshape(T, V, 3)[:, :, :2]
    d = np.stack([dist(xy[:, i], xy[:, j]) for i, j in PAIRS], axis=1)
    a = np.stack([angle_at(xy[:, j], xy[:, i], xy[:, k]) for i, j, k in TRIPLES], axis=1)
    return np.concatenate([d, a], axis=1).astype(np.float32)


def make_windows(clips, meta_by_clip, feats, label_filter=None):
    ws, labels, actions = [], [], []
    for c in clips:
        m = meta_by_clip[c]
        if label_filter is not None and m["label"] != label_filter:
            continue
        f = feats[c]
        T = f.shape[0]
        if T < SEQ_LEN:
            continue
        if args.pool == "mean":
            cs = np.concatenate([np.zeros((1, f.shape[1])), np.cumsum(f.astype(np.float64), axis=0)])
            ws.append(((cs[SEQ_LEN:] - cs[:-SEQ_LEN]) / SEQ_LEN).astype(np.float32))
        else:
            ws.append(np.stack([f[s:s + SEQ_LEN].reshape(-1) for s in range(0, T - SEQ_LEN + 1)]))
        n = T - SEQ_LEN + 1
        labels += [1 if m["label"] == "anomaly" else 0] * n
        actions += [m["action"]] * n
    return np.concatenate(ws).astype(np.float32), np.array(labels), actions


def metrics(y, s, acts):
    fpr, tpr, _ = roc_curve(y, s)
    a = np.array(acts)
    safe = s[y == 0]
    ab = {}
    for act in sorted(set(a[y == 1])):
        m = (a == act) & (y == 1)
        yy = np.concatenate([np.zeros(len(safe)), np.ones(int(m.sum()))])
        f2, t2, _ = roc_curve(yy, np.concatenate([safe, s[m]]))
        ab[act] = {"auc": float(auc(f2, t2)), "n_windows": int(m.sum())}
    return {"auc": float(auc(fpr, tpr)), "pr_auc": float(average_precision_score(y, s)), "actions": ab}


def main():
    out_path = os.path.join(OUT_DIR, "b_{}_{}_seed{}.f{:02d}.json".format(args.feat, args.pool, args.seed, args.only_fold))
    if os.path.exists(out_path):
        print("exists", out_path)
        return
    meta = json.load(open(os.path.join(RAW_DIR, "meta.json")))
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    cbd = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}
    fold_idx = args.only_fold
    test_driver = drivers[fold_idx]
    remaining = [d for d in drivers if d != test_driver]
    val_driver = remaining[fold_idx % len(remaining)]
    train_drivers = [d for d in remaining if d != val_driver]
    need = set(c for d in train_drivers + [test_driver] for c in cbd[d])
    feats = {c: build_features(np.load(os.path.join(RAW_DIR, c + ".npy"))) for c in need
             if meta_by_clip[c]["label"] == "safe" or meta_by_clip[c]["driver_id"] == test_driver}
    t0 = time.time()
    tr, _, _ = make_windows([c for d in train_drivers for c in cbd[d]], meta_by_clip, feats, "safe")
    te, y, acts = make_windows(cbd[test_driver], meta_by_clip, feats)
    mean, std = tr.mean(axis=0, dtype=np.float64).astype(np.float32), tr.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1e-6
    tr -= mean
    tr /= std
    te -= mean
    te /= std
    rng = np.random.RandomState(args.seed * 1000 + fold_idx)
    sub = tr[rng.choice(len(tr), min(SUBSAMPLE, len(tr)), replace=False)]
    print("FOLD {} {} | train {} test {} dim {}".format(fold_idx + 1, test_driver, len(tr), len(te), tr.shape[1]))
    sys.stdout.flush()
    fold = {"n_test_windows": int(len(y)), "anomaly_frac": float(y.mean()), "dim": int(tr.shape[1])}
    pca = PCA(n_components=min(20, tr.shape[1] - 1), random_state=args.seed).fit(tr)
    rec = pca.inverse_transform(pca.transform(te))
    fold["pca"] = metrics(y, ((te - rec) ** 2).mean(axis=1), acts)
    fold["pca_explained"] = float(pca.explained_variance_ratio_.sum())
    del rec
    oc = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(sub)
    fold["ocsvm"] = metrics(y, -oc.decision_function(te), acts)
    iso = IsolationForest(n_estimators=200, random_state=args.seed, n_jobs=1).fit(sub)
    fold["iforest"] = metrics(y, -iso.score_samples(te), acts)
    fold["seconds"] = round(time.time() - t0, 1)
    res = {"feat": args.feat, "pool": args.pool, "seed": args.seed, "folds": {test_driver: fold}}
    json.dump(res, open(out_path, "w"), indent=1)
    print(">> {} PCA {:.4f} OCSVM {:.4f} IF {:.4f} ({:.0f}s)".format(
        test_driver, fold["pca"]["auc"], fold["ocsvm"]["auc"], fold["iforest"]["auc"], fold["seconds"]))


if __name__ == "__main__":
    main()
