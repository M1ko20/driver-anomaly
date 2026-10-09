"""LaTeX bodies of the tables added on 2026-10-09 (window-level classical methods and
ablations of the TCN autoencoder). Prints the rows; the numbers are pasted into the
manuscript.  python gen2.py [thr]"""
import os
import sys
import json
import collections
import numpy as np
from load import DRIVERS, S
from analyze import wilcoxon_exact
from tables import pd_ as old_pd
from ablation import NEW, pd, per_action, load, ACTS, INCOMPLETE

F4 = lambda x: "{:.4f}".format(x)
SETS = [("coords", "Coordinates", 39), ("coords_nc", "Coordinates without confidences", 26), ("geom", "Geometry", 52),
        ("geom_nc", "Geometry without confidences", 39), ("geom_da", "Selected", 13), ("dist78", "All distances", 78),
        ("ang286", "All angles", 286), ("exh", "Exhaustive", 364), ("exh_conf", "Exhaustive + confidences", 377),
        ("exh_xy", "Exhaustive + coordinates", 390)]


def has(n):
    return n in NEW and len(NEW[n]) > 0


def stat(n):
    a = pd(n)
    return a.mean(), a.std(ddof=1), a.min(), int(a.argmin()) + 1, pd(n, "pr").mean(), pd(n, "macro").mean(), len(NEW[n])


print("%% --- table:16:Classical (window mean): set, dim | OC-SVM avg, s.d., min (driver), PR-AUC, class-bal. | PCA avg | Forest avg")
for k, name, dim in SETS:
    n = k + "_mean|ocsvm"
    if not has(n):
        continue
    m, sd, mn, dr, pr, mac, runs = stat(n)
    print("{} & {} & {} & {} & {} ({}) & {} & {} & {} & {} \\\\ \\hline   % runs {}".format(
        name, dim, F4(m), F4(sd), F4(mn), dr, F4(pr), F4(mac), F4(pd(k + "_mean|pca").mean()), F4(pd(k + "_mean|iforest").mean()), runs))

print("\n%% --- per-driver table: TCN-G, TCN-E | OC-SVM G, OC-SVM Selected, OC-SVM E, PCA E")
cols = [old_pd("TCN-G"), old_pd("TCN-E"), pd("geom_mean|ocsvm"), pd("geom_da_mean|ocsvm"), pd("exh_mean|ocsvm"), pd("exh_mean|pca")]
for i in range(14):
    print("Driver-{} & ".format(i + 1) + " & ".join(F4(c[i]) for c in cols) + " \\\\ \\hline")
print("avg. & " + " & ".join(F4(c.mean()) for c in cols) + " \\\\ \\hline")
print("s.d. & " + " & ".join(F4(c.std(ddof=1)) for c in cols) + " \\\\ \\hline")

print("\n%% --- representation of the window (OC-SVM avg AUC / class-balanced): flattened, per-frame, mean, mean+std")
flat = {"geom": ("SVM-G(old)", old_pd("SVM-G").mean()), "exh": None, "geom_da": None}
for k, name in (("geom", "Geometry"), ("geom_da", "Selected"), ("exh", "Exhaustive")):
    row = [name]
    for pool in ("flat", "frame", "mean", "meanstd"):
        cand = ["{}_{}|ocsvm".format(k, pool), "{}_{}(b)|ocsvm".format(k, pool)]
        n = next((c for c in cand if has(c)), None)
        if n is None and pool == "flat" and k == "geom":
            row.append(F4(old_pd("SVM-G").mean()) + " (3)")
        elif n is None:
            row.append("--")
        else:
            row.append("{} ({})".format(F4(pd(n).mean()), len(NEW[n])))
    print(" & ".join(row) + " \\\\ \\hline")
for m in ("pca", "iforest"):
    for k in ("geom", "geom_da", "exh"):
        print("   %% {} {}: ".format(m, k) + "  ".join("{} {}".format(p, F4(pd("{}_{}|{}".format(k, p, m)).mean())) for p in ("frame", "mean", "meanstd", "flat") if has("{}_{}|{}".format(k, p, m))))

print("\n%% --- window length (clips >= 90 frames), OC-SVM, window mean")
for k in ("geom_da", "exh"):
    print("  " + k + ": " + "  ".join("T{} {} ({})".format(T, F4(pd(n).mean()), len(NEW[n])) for T in (15, 30, 60, 90)
                                       for n in ["{}_mean_T{}_minL90|ocsvm".format(k, T) if T != 60 else "{}_mean_minL90|ocsvm".format(k)] if has(n)))

print("\n%% --- sensitivity of OC-SVM (window mean, one run)")
for k in ("geom_da", "exh", "geom"):
    base = pd(k + "_mean|ocsvm").mean()
    out = ["default {}".format(F4(base))]
    for tag in ("nu0.01", "nu0.1", "nu0.2", "g0.25", "g0.5", "g2", "g4", "n2000", "n32000"):
        n = "{}_mean_{}|ocsvm".format(k, tag)
        if has(n):
            out.append("{} {}".format(tag, F4(pd(n).mean())))
    print("  " + k + ": " + "  ".join(out))

print("\n%% --- paired tests (A, B, diff, drivers, p)")
refs = {"TCN-G": old_pd("TCN-G"), "TCN-E": old_pd("TCN-E"), "TCN-C": old_pd("TCN-C"), "SVM-G": old_pd("SVM-G"), "TCN-G-z8": old_pd("TCN-G-z8")}
get = lambda n: pd(n) if n in NEW else refs[n]
T = [("geom_da_mean|ocsvm", "TCN-E"), ("exh_mean|ocsvm", "TCN-E"), ("exh_mean|pca", "TCN-E"), ("geom_mean|ocsvm", "TCN-G"),
     ("geom_mean|ocsvm", "SVM-G"), ("geom_da_mean|ocsvm", "exh_mean|ocsvm"), ("geom_da_mean|ocsvm", "geom_mean|ocsvm"),
     ("geom_da_mean|ocsvm", "geom_nc_mean|ocsvm"), ("geom_nc_mean|ocsvm", "geom_mean|ocsvm"), ("coords_nc_mean|ocsvm", "coords_mean|ocsvm"),
     ("exh_mean|ocsvm", "exh_conf_mean|ocsvm"), ("exh_mean|ocsvm", "exh_xy_mean|ocsvm"), ("exh_mean|ocsvm", "dist78_mean|ocsvm"),
     ("exh_mean|ocsvm", "ang286_mean|ocsvm"), ("exh_mean|ocsvm", "exh_frame|ocsvm"), ("exh_mean|ocsvm", "exh_meanstd|ocsvm"),
     ("geom_da_mean|ocsvm", "geom_da_frame|ocsvm"), ("geom_da_mean|ocsvm", "geom_da_meanstd|ocsvm"), ("geom_da_mean|ocsvm", "geom_da_flat|ocsvm"),
     ("exh_mean|ocsvm", "exh_flat(b)|ocsvm"), ("exh_flat(b)|ocsvm", "TCN-E"),
     ("tcn_geom_da", "TCN-G"), ("tcn_geom_da", "TCN-E"), ("geom_da_mean|ocsvm", "tcn_geom_da"), ("tcn_geom_nc", "TCN-G"),
     ("tcn_coords_nc", "TCN-C"), ("tcn_geom_k1", "tcn_geom"), ("tcn_geom_da_k1", "tcn_geom_da"), ("tcn_exh_k1", "tcn_exh"),
     ("tcn_exh_z8", "TCN-E"), ("tcn_exh_conf", "TCN-E"), ("tcn_geom", "TCN-G"), ("tcn_exh", "TCN-E")]
for a, b in T:
    if (a not in NEW and a not in refs) or (b not in NEW and b not in refs):
        continue
    xa, xb = get(a), get(b)
    d = xa - xb
    p = wilcoxon_exact(xa, xb)[2]
    print("  {:28s} {:24s} {:+.4f} & {:2d} & {}".format(a, b, d.mean(), int((d > 0).sum()), "$<$0.001" if p < 0.001 else "{:.3f}".format(p)))

print("\n%% --- per action: OC-SVM Selected, OC-SVM Exhaustive (order of table:7:Categories)")
for n in ("geom_da_mean|ocsvm", "exh_mean|ocsvm", "tcn_geom_da", "tcn_exh"):
    if has(n):
        print("  {:22s} ".format(n) + " ".join(F4(x) for x in per_action(n)))

print("\n%% --- leave-one-driver-out selection of the feature set (OC-SVM, window mean)")
M = {k: pd(k + "_mean|ocsvm") for k, _, _ in SETS if has(k + "_mean|ocsvm")}
sel, ch = [], collections.Counter()
for i in range(14):
    oth = [j for j in range(14) if j != i]
    best = max(M, key=lambda k: M[k][oth].mean())
    ch[best] += 1
    sel.append(M[best][i])
print("  mean AUC {:.4f}; chosen: {}".format(np.mean(sel), dict(ch)))

print("\n%% --- TCN ablations (finished runs); incomplete: {}".format(INCOMPLETE))
for n in sorted(k for k in NEW if k.startswith("tcn_")):
    m, sd, mn, dr, pr, mac, runs = stat(n)
    print("  {:16s} runs {} avg {} sd {} min {} ({}) PR {} macro {} | static {} shuffle {} params {}".format(
        n, runs, F4(m), F4(sd), F4(mn), dr, F4(pr), F4(mac), F4(pd(n, "static").mean()), F4(pd(n, "shuffle").mean()), NEW[n][0]["n_params"]))

if "thr" in sys.argv:
    from sklearn.metrics import roc_auc_score
    man = {m["clip"]: m for m in json.load(open(os.path.join(S, "clips_v2_manifest.json")))}
    print("\n%% --- thresholds: method | pooled AUC raw, normalized | val: FPR mean, median, max, TPR mean, median, min | cal 60 s: the same")
    cfgs = [("TCN, Geometry", ["x_tcn_geom_common_seed%d" % s for s in (0, 1, 2)]), ("TCN, Exhaustive", ["x_tcn_exh_seed0", "x_tcn_exh_seed1"])]
    for n, lab in (("geom_da_mean|ocsvm", "OC-SVM, Selected"), ("exh_mean|ocsvm", "OC-SVM, Exhaustive")):
        cfgs.append((lab, [r["scores"] for r in NEW[n]]))
    for name, prefixes in cfgs:
        rows = collections.defaultdict(list)
        pooled = []
        for p in prefixes:
            S_, Z_, Y_ = [], [], []
            for d in DRIVERS:
                z = load(p, d)
                s, y, c = z["scores"], z["labels"], z["clips"]
                S_.append(s); Y_.append(y); Z_.append((s - float(z["val_mean"])) / float(z["val_std"]))
                rows["val"].append(((s[y == 0] > float(z["val_p95"])).mean(), (s[y == 1] > float(z["val_p95"])).mean()))
                nc = sorted(set(c[y == 0].tolist()), key=lambda x: (man[x]["session"], man[x]["start_frame"]))
                for K in (30, 60, 120):
                    cal, tot = [], 0
                    for x in nc:
                        if tot >= K * 30:
                            break
                        cal.append(x)
                        tot += int((c == x).sum()) + 59
                    mc = np.isin(c, cal)
                    th = np.percentile(s[(y == 0) & mc], 95)
                    rest = (y == 0) & ~mc
                    rows["cal%d" % K].append(((s[rest] > th).mean(), (s[y == 1] > th).mean(), tot / 30.0, rest.sum()))
            pooled.append((roc_auc_score(np.concatenate(Y_), np.concatenate(S_)), roc_auc_score(np.concatenate(Y_), np.concatenate(Z_))))
        pm = np.mean(pooled, axis=0)
        for k, v in rows.items():
            v = np.array(v).reshape(len(prefixes), 14, -1).mean(axis=0)
            extra = "" if k == "val" else "  [cal. length mean {:.0f} s, drivers with FPR<=0.10: {}]".format(v[:, 2].mean(), int((v[:, 0] <= 0.10).sum()))
            if k == "val":
                extra = "  [drivers with FPR<=0.10: {}; argmax FPR D{}, argmin TPR D{}]".format(int((v[:, 0] <= 0.10).sum()), v[:, 0].argmax() + 1, v[:, 1].argmin() + 1)
            print("{} & {:6s} & {} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \\hline{}".format(
                name, k, F4(pm[0]), F4(pm[1]), v[:, 0].mean(), np.median(v[:, 0]), v[:, 0].max(), v[:, 1].mean(), np.median(v[:, 1]), v[:, 1].min(), extra))
