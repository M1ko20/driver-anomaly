"""Score-level analysis: ensembles, pooled AUC, thresholds, exact Wilcoxon tests."""
import sys
import json
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from load import RUNS, DRIVERS, per_driver, run_means, load_scores

np.set_printoptions(linewidth=200, precision=4, suppress=True)
_cache = {}


def sc(prefix, d):
    k = (prefix, d)
    if k not in _cache:
        z = load_scores(prefix, d)
        _cache[k] = {f: z[f] for f in z.files}
    return _cache[k]


def wilcoxon_exact(x, y):
    """Two-sided exact Wilcoxon signed-rank test (no zero differences expected), by
    enumeration of all 2^n sign patterns. Returns (W+, W-, p)."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    n = len(d)
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    a = np.abs(d)[order]
    i = 0
    while i < n:  # average ranks for ties
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    wp = ranks[d > 0].sum()
    wm = ranks[d < 0].sum()
    stat = min(wp, wm)
    cnt = 0
    tot = 2 ** n
    for mask in range(tot):
        s = sum(ranks[k] for k in range(n) if mask >> k & 1)
        if min(s, ranks.sum() - s) <= stat + 1e-9:
            cnt += 1
    return wp, wm, cnt / tot


def paired(a, b, la, lb):
    da = np.asarray(a) - np.asarray(b)
    wp, wm, p = wilcoxon_exact(a, b)
    print("  {:34s} vs {:30s} diff {:+.4f}  sd(diff) {:.4f}  wins {:2d}/14  p={:.4f}  [min {:+.4f} max {:+.4f}]".format(
        la, lb, da.mean(), da.std(ddof=1), int((da > 0).sum()), p, da.min(), da.max()))
    return da.mean(), int((da > 0).sum()), p


def ens_scores(prefixes, d, mode="mean", clips=None):
    zs = []
    for p in prefixes:
        z = sc(p, d)
        s = (z["scores"] - float(z["val_mean"])) / float(z["val_std"])
        if clips is not None:
            s = s[np.isin(z["clips"], clips)]
        zs.append(s)
    zs = np.stack(zs)
    return zs.mean(axis=0) if mode == "mean" else zs.max(axis=0)


def ens_eval(prefixes, mode="mean"):
    """per-driver AUC array, PR-AUC array, pooled AUC of the combined z-scores"""
    aucs, prs, allS, allY = [], [], [], []
    common = None
    for d in DRIVERS:
        clipsets = [set(sc(p, d)["clips"].tolist()) for p in prefixes]
        inter = set.intersection(*clipsets)
        ref = sc(prefixes[0], d)
        keep = np.isin(ref["clips"], list(inter))
        y = ref["labels"][keep]
        s = ens_scores(prefixes, d, mode, clips=list(inter))
        assert len(s) == len(y)
        aucs.append(roc_auc_score(y, s))
        prs.append(average_precision_score(y, s))
        allS.append(s)
        allY.append(y)
    return np.array(aucs), np.array(prs), roc_auc_score(np.concatenate(allY), np.concatenate(allS))


def pooled_single(prefix):
    raw_s, z_s, ys = [], [], []
    for d in DRIVERS:
        z = sc(prefix, d)
        raw_s.append(z["scores"])
        z_s.append((z["scores"] - float(z["val_mean"])) / float(z["val_std"]))
        ys.append(z["labels"])
    y = np.concatenate(ys)
    return roc_auc_score(y, np.concatenate(raw_s)), roc_auc_score(y, np.concatenate(z_s))


def action_auc_from_scores(score_fn):
    """score_fn(d) -> (scores, labels, actions). Returns {action: [auc per driver in which it occurs]}"""
    out = {}
    for d in DRIVERS:
        s, y, a = score_fn(d)
        safe = s[y == 0]
        for act in sorted(set(a[y == 1].tolist())):
            m = (a == act) & (y == 1)
            yy = np.concatenate([np.zeros(len(safe)), np.ones(int(m.sum()))])
            out.setdefault(act, {})[d] = roc_auc_score(yy, np.concatenate([safe, s[m]]))
    return out


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    T = lambda k: "all" in what or k in what

    if T("pooled"):
        print("\n=== POOLED AUC of single models (raw scores / val-normalized scores), per run")
        for n in RUNS:
            if RUNS[n][0].get("scores"):
                ps = [pooled_single(r["scores"]) for r in RUNS[n]]
                print("  {:26s} raw {}  mean {:.4f} | znorm {}  mean {:.4f}".format(
                    n, " ".join("{:.4f}".format(p[0]) for p in ps), np.mean([p[0] for p in ps]),
                    " ".join("{:.4f}".format(p[1]) for p in ps), np.mean([p[1] for p in ps])))

    ENS = {}
    if T("ens"):
        print("\n=== ENSEMBLES (z-normalized by val mean/std), per seed: avg AUC | avg PR-AUC | pooled")
        defs = {}
        for s in (0, 1):
            g, h, hh = ("x_tcn_{}_common_seed{}".format(f, s) for f in ("geom", "hands", "hh"))
            mg, mh, mhh = ("mamba_{}_common_seed{}".format(f, s) for f in ("geom", "hands", "hh"))
            defs.setdefault("TCN G+H+HH", []).append([g, h, hh])
            defs.setdefault("TCN G+H", []).append([g, h])
            defs.setdefault("TCN G+HH", []).append([g, hh])
            defs.setdefault("TCN H+HH", []).append([h, hh])
            defs.setdefault("Mamba G+H+HH", []).append([mg, mh, mhh])
            defs.setdefault("Mamba G+H", []).append([mg, mh])
            defs.setdefault("Mamba G+HH", []).append([mg, mhh])
            defs.setdefault("Mamba H+HH", []).append([mh, mhh])
            defs.setdefault("TCN G + Mamba G", []).append([g, mg])
            defs.setdefault("TCN+Mamba x G,H,HH (6)", []).append([g, h, hh, mg, mh, mhh])
            defs.setdefault("TCN G + TCN exh", []).append([g, "x_tcn_exh_seed{}".format(s)])
        g0, g1, g2 = ("x_tcn_geom_common_seed{}".format(s) for s in (0, 1, 2))
        defs["TCN 3xG seeds"] = [[g0, g1, g2]]
        defs["TCN 2xG seeds"] = [[g0, g1], [g0, g2], [g1, g2]]
        defs["TCN 2x exh seeds"] = [["x_tcn_exh_seed0", "x_tcn_exh_seed1"]]
        defs["Mamba 2xG seeds"] = [["mamba_geom_common_seed0", "mamba_geom_common_seed1"]]
        for mode in ("mean", "max"):
            for name, lists in defs.items():
                res = [ens_eval(l, mode) for l in lists]
                ENS[(name, mode)] = res
                print("  {:24s} {:4s} avg {}  => {:.4f} | PR {:.4f} | pooled {}  => {:.4f}".format(
                    name, mode, " ".join("{:.4f}".format(r[0].mean()) for r in res),
                    np.mean([r[0].mean() for r in res]), np.mean([r[1].mean() for r in res]),
                    " ".join("{:.4f}".format(r[2]) for r in res), np.mean([r[2] for r in res])))
        print("\n  per-driver (average of seeds):")
        for name in ["TCN G+H+HH", "Mamba G+H+HH", "TCN 3xG seeds", "TCN G + Mamba G", "TCN+Mamba x G,H,HH (6)"]:
            print("  {:24s}".format(name), np.mean([r[0] for r in ENS[(name, "mean")]], axis=0))
        json.dump({"{}|{}".format(*k): {"auc": np.mean([r[0] for r in v], axis=0).tolist(),
                                        "auc_runs": [r[0].tolist() for r in v],
                                        "pr": float(np.mean([r[1].mean() for r in v])),
                                        "pooled": [float(r[2]) for r in v]} for k, v in ENS.items()},
                  open("ens.json", "w"), indent=1)

    if T("thr"):
        print("\n=== THRESHOLD = 95th / 99th percentile of the validation-driver normal scores (x_ runs)")
        print("    mean over drivers (and runs) of FPR, TPR, F1; [min..max over drivers]")
        for n in RUNS:
            r0 = RUNS[n][0]
            if not r0.get("scores") or not n.startswith("x_"):
                continue
            for q in ("val_p95", "val_p99"):
                F, Tp, F1 = [], [], []
                for r in RUNS[n]:
                    f_, t_, f1_ = [], [], []
                    for d in DRIVERS:
                        z = sc(r["scores"], d)
                        pred = z["scores"] > float(z[q])
                        y = z["labels"] == 1
                        tp, fp, fn, tn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum()
                        f_.append(fp / (fp + tn))
                        t_.append(tp / (tp + fn))
                        f1_.append(2 * tp / (2 * tp + fp + fn))
                    F.append(f_); Tp.append(t_); F1.append(f1_)
                F, Tp, F1 = (np.mean(x, axis=0) for x in (F, Tp, F1))
                print("  {:24s} {} FPR {:.3f} [{:.3f}..{:.3f}]  TPR {:.3f} [{:.3f}..{:.3f}]  F1 {:.3f} [{:.3f}..{:.3f}]".format(
                    n, q[-3:], F.mean(), F.min(), F.max(), Tp.mean(), Tp.min(), Tp.max(), F1.mean(), F1.min(), F1.max()))
                if n in ("x_tcn_geom_common", "x_tcn_exh") and q == "val_p95":
                    print("       FPR per driver", F)
                    print("       TPR per driver", Tp)
