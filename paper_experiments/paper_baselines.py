"""Classical baselines (PCA reconstruction, One-Class SVM, Isolation Forest) under the
same 14-fold LOSO protocol as the autoencoders (paper_loso.py): same folds, same
Dense windows (60 frames, step 1) flattened into one vector, per-fold z-score
standardization from train windows, training on the 12 train drivers' safe windows
(the validation driver is not used), scores on all windows of the test driver.
Settings as in the earlier baseline scripts: PCA 20 components, OC-SVM RBF
nu=0.05 gamma='scale' and Isolation Forest 200 trees, both on a random subset
of 8000 training windows.
Output: /home/fei/mik0542/paper_experiments/baselines_<feat>_seed<k>.json"""
import os
import sys
import json
import time
import argparse
import numpy as np
from scipy.signal import savgol_filter
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
parser.add_argument("--feat", required=True, choices=["coords", "geom", "kin"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--only_fold", type=int, default=-1)
parser.add_argument("--tag", default="")
args = parser.parse_args()


def build_features(b, feat):
    T = b.shape[0]
    alpha = b[:, :39]
    if feat == "coords":
        return alpha.astype(np.float32)
    if feat == "geom":
        return b.astype(np.float32)
    xy = alpha.reshape(T, V, 3)[:, :, :2]
    sm = savgol_filter(xy, window_length=5, polyorder=2, axis=0, mode="nearest") if T >= 5 else xy.copy()
    vel = np.gradient(sm, axis=0)
    acc = np.gradient(vel, axis=0)
    return np.concatenate([b, vel.reshape(T, -1), acc.reshape(T, -1)], axis=1).astype(np.float32)


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
        for s in range(0, T - SEQ_LEN + 1):
            ws.append(f[s:s + SEQ_LEN].reshape(-1))
            labels.append(1 if m["label"] == "anomaly" else 0)
            actions.append(m["action"])
    return np.stack(ws).astype(np.float32), np.array(labels), actions


def metrics(y, s, acts):
    fpr, tpr, _ = roc_curve(y, s)
    a = np.array(acts)
    safe = s[y == 0]
    ab = {}
    for act in sorted(set(a[y == 1])):
        m = (a == act) & (y == 1)
        yy = np.concatenate([np.zeros(len(safe)), np.ones(int(m.sum()))])
        ss = np.concatenate([safe, s[m]])
        f2, t2, _ = roc_curve(yy, ss)
        ab[act] = {"auc": float(auc(f2, t2)), "n_windows": int(m.sum())}
    return {"auc": float(auc(fpr, tpr)), "pr_auc": float(average_precision_score(y, s)), "actions": ab}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "baselines_{}{}_seed{}.json".format(args.feat, args.tag, args.seed))
    meta = json.load(open(os.path.join(RAW_DIR, "meta.json")))
    meta_by_clip = {m["clip"]: m for m in meta}
    feats = {m["clip"]: build_features(np.load(os.path.join(RAW_DIR, m["clip"] + ".npy")), args.feat) for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    cbd = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}
    res = {"feat": args.feat, "seed": args.seed, "folds": {}}
    if os.path.exists(out_path):
        res = json.load(open(out_path))
    for fold_idx, test_driver in enumerate(drivers):
        if args.only_fold >= 0 and fold_idx != args.only_fold:
            continue
        if test_driver in res["folds"]:
            continue
        t0 = time.time()
        remaining = [d for d in drivers if d != test_driver]
        val_driver = remaining[fold_idx % len(remaining)]
        train_drivers = [d for d in remaining if d != val_driver]
        tr, _, _ = make_windows([c for d in train_drivers for c in cbd[d]], meta_by_clip, feats, "safe")
        te, y, acts = make_windows(cbd[test_driver], meta_by_clip, feats)
        mean, std = tr.mean(axis=0), tr.std(axis=0)
        std[std < 1e-6] = 1e-6
        tr -= mean; tr /= std
        te -= mean; te /= std
        rng = np.random.RandomState(args.seed * 1000 + fold_idx)
        sub = tr[rng.choice(len(tr), min(SUBSAMPLE, len(tr)), replace=False)]
        print("FOLD {}/{} {} | train {} test {} dim {}".format(fold_idx + 1, len(drivers), test_driver,
                                                              len(tr), len(te), tr.shape[1]))
        sys.stdout.flush()
        fold = {"n_test_windows": int(len(y)), "anomaly_frac": float(y.mean())}
        # PCA reconstruction (all train windows)
        pca = PCA(n_components=20, random_state=args.seed).fit(tr)
        rec = pca.inverse_transform(pca.transform(te))
        fold["pca"] = metrics(y, ((te - rec) ** 2).mean(axis=1), acts)
        del rec
        # One-Class SVM
        oc = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(sub)
        fold["ocsvm"] = metrics(y, -oc.decision_function(te), acts)
        # Isolation Forest
        iso = IsolationForest(n_estimators=200, random_state=args.seed, n_jobs=4).fit(sub)
        fold["iforest"] = metrics(y, -iso.score_samples(te), acts)
        fold["seconds"] = round(time.time() - t0, 1)
        res["folds"][test_driver] = fold
        print(">> {} PCA {:.4f} OCSVM {:.4f} IF {:.4f} ({:.0f}s)".format(
            test_driver, fold["pca"]["auc"], fold["ocsvm"]["auc"], fold["iforest"]["auc"], fold["seconds"]))
        sys.stdout.flush()
        for k in ("pca", "ocsvm", "iforest"):
            a = [f[k]["auc"] for f in res["folds"].values()]
            res["mean_auc_" + k] = float(np.mean(a))
        json.dump(res, open(out_path, "w"), indent=1)
    print("DONE", out_path)


if __name__ == "__main__":
    main()
