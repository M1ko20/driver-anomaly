"""Results added on 2026-10-09: classical one-class methods on window-level summaries
(paper_classical.py), flattened-window baselines with the Exhaustive set
(paper_baselines2.py) and the ablations of the TCN autoencoder (paper_ablation.py).

  python ablation.py            # summary, paired tests, per-action results
  python ablation.py thr        # + thresholds: validation driver vs calibration on the test driver
  python ablation.py comb       # + score-level combinations of OC-SVM and the TCN autoencoder

NEW[name] = list of runs; run = {"seed", "auc": {driver: auc}, "pr", "macro", "actions", "scores": prefix}"""
import os
import re
import sys
import glob
import json
import collections
import numpy as np
from sklearn.metrics import roc_auc_score
from load import RUNS, DRIVERS, PE, S
from analyze import wilcoxon_exact
from tables import pd_ as old_pd

np.set_printoptions(linewidth=220, precision=4, suppress=True)
NEW = collections.defaultdict(list)


def macro_of(actions):
    return float(np.mean([v["auc"] for v in actions.values()]))


# classical, window summaries
for p in sorted(glob.glob(os.path.join(PE, "c_*_seed*.json"))):
    d = json.load(open(p))
    if len(d["folds"]) != 14:
        continue
    for m in ("pca", "ocsvm", "iforest"):
        if m not in next(iter(d["folds"].values())):
            continue
        f = d["folds"]
        NEW["{}|{}".format(d["exp"][2:], m)].append({
            "seed": d["seed"], "auc": {k: v[m]["auc"] for k, v in f.items()}, "pr": {k: v[m]["pr_auc"] for k, v in f.items()},
            "macro": {k: v[m]["macro_action_auc"] for k, v in f.items()},
            "actions": {k: {a: b["auc"] for a, b in v[m]["actions"].items()} for k, v in f.items()},
            "val_p95": {k: v[m]["val_p95"] for k, v in f.items()},
            "scores": "{}_seed{}_{}".format(d["exp"], d["seed"], m) if m != "iforest" else None})


def merge_folds(pattern):
    out = collections.defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(PE, pattern))):
        d = json.load(open(p))
        key = re.sub(r"\.f\d\d\.json$", "", os.path.basename(p))
        out[key].update(d["folds"])
        out[key + "#meta"] = d
    return out


# flattened-window baselines with the Exhaustive set
mf = merge_folds("b_*_seed*.f??.json")
for key in sorted(k for k in mf if "#" not in k):
    f = mf[key]
    if len(f) != 14:
        continue
    d = mf[key + "#meta"]
    for m in ("pca", "ocsvm", "iforest"):
        NEW["{}_{}(b)|{}".format(d["feat"], d["pool"], m)].append({
            "seed": d["seed"], "auc": {k: v[m]["auc"] for k, v in f.items()}, "pr": {k: v[m]["pr_auc"] for k, v in f.items()},
            "macro": {k: macro_of(v[m]["actions"]) for k, v in f.items()},
            "actions": {k: {a: b["auc"] for a, b in v[m]["actions"].items()} for k, v in f.items()}, "scores": None})

# TCN ablations
mf = merge_folds("a_tcn_*_seed*.f??.json")
INCOMPLETE = {}
for key in sorted(k for k in mf if "#" not in k):
    f = mf[key]
    d = mf[key + "#meta"]
    if len(f) != 14:
        INCOMPLETE[key] = len(f)
        continue
    NEW[d["exp"][2:]].append({
        "seed": d["seed"], "auc": {k: v["auc"] for k, v in f.items()}, "pr": {k: v["pr_auc"] for k, v in f.items()},
        "macro": {k: v["macro_action_auc"] for k, v in f.items()},
        "static": {k: v["auc_static"] for k, v in f.items()}, "shuffle": {k: v["auc_shuffle"] for k, v in f.items()},
        "actions": {k: {a: b["auc"] for a, b in v["actions"].items()} for k, v in f.items()},
        "epochs": {k: v["epochs"] for k, v in f.items()}, "n_params": next(iter(f.values()))["n_params"],
        "val_p95": {k: v["val_p95"] for k, v in f.items()}, "scores": key})


def pd(name, key="auc"):
    return np.mean([[r[key][d] for d in DRIVERS] for r in NEW[name]], axis=0)


def load(prefix, d):
    return np.load(os.path.join(PE, "scores", "{}_{}.npz".format(prefix, d)), allow_pickle=False)


def paired(a, b, la, lb):
    da = np.asarray(a) - np.asarray(b)
    _, _, p = wilcoxon_exact(a, b)
    print("  {:34s} vs {:26s} diff {:+.4f}  wins {:2d}/14  p={:.4f}  [min {:+.4f} max {:+.4f}]".format(
        la, lb, da.mean(), int((da > 0).sum()), p, da.min(), da.max()))


ACTS = ["reach_backseat", "reach_side", "hair_and_makeup", "phonecall_left", "phonecall_right", "drinking",
        "texting_right", "talking_to_passenger", "radio", "texting_left"]


def per_action(name):
    out = []
    for a in ACTS:
        out.append(np.mean([np.mean([r["actions"][d][a] for d in DRIVERS if a in r["actions"][d]]) for r in NEW[name]]))
    return np.array(out)


if __name__ == "__main__":
    what = sys.argv[1:]
    refs = {"TCN-G": old_pd("TCN-G"), "TCN-E": old_pd("TCN-E"), "TCN-C": old_pd("TCN-C"), "SVM-G(flat)": old_pd("SVM-G"),
            "PCA-G(flat)": old_pd("PCA-G"), "TCN-G-z8": old_pd("TCN-G-z8")}
    print("=== reference (article of 2026-10-08): " + "  ".join("{} {:.4f}".format(k, v.mean()) for k, v in refs.items()))
    if INCOMPLETE:
        print("=== incomplete TCN ablation runs (folds done):", INCOMPLETE)
    print("\n=== NEW RUNS: n runs, avg AUC, sd over drivers, min (driver), PR-AUC, class-balanced AUC, run means")
    for n in sorted(NEW):
        a = pd(n)
        rm = [np.mean([r["auc"][d] for d in DRIVERS]) for r in NEW[n]]
        extra = ""
        if "static" in NEW[n][0]:
            extra = " | static {:.4f} shuffle {:.4f} params {}".format(pd(n, "static").mean(), pd(n, "shuffle").mean(), NEW[n][0]["n_params"])
        print("  {:30s} n={} avg {:.4f} sd {:.4f} min {:.4f} (D{:<2d}) PR {:.4f} macro {:.4f} | runs {}{}".format(
            n, len(rm), a.mean(), a.std(ddof=1), a.min(), a.argmin() + 1, pd(n, "pr").mean(), pd(n, "macro").mean(),
            " ".join("%.4f" % x for x in rm), extra))
    print("\n=== PER DRIVER")
    for k, v in refs.items():
        print("  {:30s}".format(k), " ".join("%.4f" % x for x in v), "| %.4f" % v.mean())
    for n in sorted(NEW):
        a = pd(n)
        print("  {:30s}".format(n), " ".join("%.4f" % x for x in a), "| %.4f" % a.mean())

    print("\n=== PAIRED TESTS (exact two-sided Wilcoxon over the 14 drivers)")
    T = [("exh_mean|ocsvm", "TCN-E"), ("exh_mean|ocsvm", "TCN-G"), ("exh_mean|pca", "TCN-E"), ("geom_mean|ocsvm", "TCN-G"),
         ("geom_mean|pca", "TCN-G"), ("geom_mean|ocsvm", "SVM-G(flat)"), ("geom_da_mean|ocsvm", "TCN-E"),
         ("geom_da_mean|ocsvm", "exh_mean|ocsvm"), ("geom_da_mean|ocsvm", "geom_mean|ocsvm"),
         ("geom_nc_mean|ocsvm", "geom_mean|ocsvm"), ("coords_nc_mean|ocsvm", "coords_mean|ocsvm"),
         ("geom_da_mean|ocsvm", "geom_nc_mean|ocsvm"), ("exh_mean|ocsvm", "exh_conf_mean|ocsvm"),
         ("exh_mean|ocsvm", "exh_xy_mean|ocsvm"), ("exh_mean|ocsvm", "dist78_mean|ocsvm"), ("exh_mean|ocsvm", "ang286_mean|ocsvm"),
         ("exh_mean|ocsvm", "exh_meanstd|ocsvm"), ("exh_mean|ocsvm", "exh_frame|ocsvm"), ("geom_mean|ocsvm", "geom_frame|ocsvm"),
         ("geom_mean|ocsvm", "geom_meanstd|ocsvm"), ("exh_mean|ocsvm", "exh_flat(b)|ocsvm"), ("exh_flat(b)|ocsvm", "TCN-E"),
         ("exh_flat(b)|pca", "TCN-E"),
         ("tcn_geom", "TCN-G"), ("tcn_exh", "TCN-E"), ("tcn_geom_nc", "TCN-G"), ("tcn_coords_nc", "TCN-C"),
         ("tcn_geom_da", "TCN-G"), ("tcn_geom_da", "TCN-E"), ("tcn_geom_k1", "TCN-G"), ("tcn_exh_k1", "TCN-E"),
         ("tcn_geom_da_k1", "tcn_geom_da"), ("tcn_exh_z8", "TCN-E"), ("tcn_exh_conf", "TCN-E"),
         ("geom_da_mean|ocsvm", "tcn_geom_da"), ("exh_mean|ocsvm", "tcn_exh_z8")]
    for a, b in T:
        xa = pd(a) if a in NEW else refs.get(a)
        xb = pd(b) if b in NEW else refs.get(b)
        if xa is None or xb is None or (a not in NEW and a not in refs) or (b not in NEW and b not in refs):
            continue
        paired(xa, xb, a, b)

    print("\n=== PER ACTION (avg over the drivers in which the type occurs, then over runs)")
    print("  {:30s}".format("") + " ".join("{:>9s}".format(a[:9]) for a in ACTS))
    for n in sorted(NEW):
        print("  {:30s}".format(n) + " ".join("{:9.4f}".format(x) for x in per_action(n)))

    if "thr" in what:
        man = {m["clip"]: m for m in json.load(open(os.path.join(S, "clips_v2_manifest.json")))}
        print("\n=== THRESHOLDS. val: 95th percentile of the scores of the validation driver (as in the article).")
        print("    cal K: 95th percentile of the scores of the first K seconds of normal driving of the TEST driver")
        print("    (clips in chronological order, sessions s1, s2, s3); FPR on the remaining normal windows, TPR on all anomalous windows.")
        cfgs = [("TCN-G", ["x_tcn_geom_common_seed%d" % s for s in (0, 1, 2)]), ("TCN-E", ["x_tcn_exh_seed0", "x_tcn_exh_seed1"])]
        for n in ("exh_mean|ocsvm", "geom_da_mean|ocsvm", "geom_mean|ocsvm", "exh_mean|pca", "tcn_geom_da", "tcn_exh", "tcn_geom"):
            if n in NEW and NEW[n][0]["scores"]:
                cfgs.append((n, [r["scores"] for r in NEW[n]]))
        for name, prefixes in cfgs:
            rows = collections.defaultdict(list)
            for p in prefixes:
                for d in DRIVERS:
                    z = load(p, d)
                    s, y, c = z["scores"], z["labels"], z["clips"]
                    rows["val"].append(((s[y == 0] > float(z["val_p95"])).mean(), (s[y == 1] > float(z["val_p95"])).mean()))
                    nc = sorted(set(c[y == 0].tolist()), key=lambda x: (man[x]["session"], man[x]["start_frame"]))
                    nwin = {x: int(((c == x)).sum()) for x in nc}
                    for K in (30, 60, 120):
                        cal, tot = [], 0
                        for x in nc:
                            if tot >= K * 30:
                                break
                            cal.append(x)
                            tot += nwin[x] + 59
                        mc = np.isin(c, cal)
                        th = np.percentile(s[(y == 0) & mc], 95)
                        rest = (y == 0) & ~mc
                        rows["cal%d" % K].append(((s[rest] > th).mean() if rest.any() else np.nan, (s[y == 1] > th).mean()))
                    # calibration in the first session only (s1), test on the other sessions
                    ses = np.array([man[x]["session"] for x in c])
                    c1 = (y == 0) & (ses == "s1")
                    oth = ses != "s1"
                    if c1.sum() > 100 and ((y == 0) & oth).any():
                        th = np.percentile(s[c1], 95)
                        rows["s1->rest"].append(((s[(y == 0) & oth] > th).mean(), (s[(y == 1) & oth] > th).mean()))
            for k, v in rows.items():
                v = np.array(v).reshape(len(prefixes), -1, 2).mean(axis=0)
                print("  {:20s} {:9s} FPR mean {:.3f} median {:.3f} max {:.3f} | TPR mean {:.3f} median {:.3f} min {:.3f}".format(
                    name, k, np.nanmean(v[:, 0]), np.nanmedian(v[:, 0]), np.nanmax(v[:, 0]), v[:, 1].mean(), np.median(v[:, 1]), v[:, 1].min()))

    if "comb" in what:
        print("\n=== SCORE-LEVEL COMBINATIONS (z-normalization by the validation driver, mean), avg AUC over drivers")

        def zs(p, d):
            z = load(p, d)
            return (z["scores"] - float(z["val_mean"])) / float(z["val_std"]), z["labels"], z["clips"]

        def comb(prefs):
            out = []
            for d in DRIVERS:
                parts = [zs(p, d) for p in prefs]
                assert all(len(x[0]) == len(parts[0][0]) and (x[2] == parts[0][2]).all() for x in parts)
                out.append(roc_auc_score(parts[0][1], np.mean([x[0] for x in parts], axis=0)))
            return np.array(out)

        def single(p):
            return np.array([roc_auc_score(load(p, d)["labels"], load(p, d)["scores"]) for d in DRIVERS])

        P = {"OCSVM-E": "c_exh_mean_seed0_ocsvm", "OCSVM-DA": "c_geom_da_mean_seed0_ocsvm", "PCA-E": "c_exh_mean_seed0_pca",
             "TCN-E": "x_tcn_exh_seed0", "TCN-E'": "x_tcn_exh_seed1"}
        for k, p in P.items():
            print("  {:28s} {:.4f}".format(k, single(p).mean()))
        for ks in (("OCSVM-E", "TCN-E"), ("OCSVM-E", "OCSVM-DA"), ("OCSVM-E", "PCA-E"), ("OCSVM-DA", "TCN-E"),
                   ("OCSVM-E", "OCSVM-DA", "TCN-E"), ("TCN-E", "TCN-E'")):
            a = comb([P[k] for k in ks])
            print("  {:28s} {:.4f}  min {:.4f}   vs best single: wins {}".format(
                " + ".join(ks), a.mean(), a.min(), int((a > np.max([single(P[k]) for k in ks], axis=0)).sum())))
