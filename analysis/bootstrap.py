"""Bootstrap confidence intervals of the average AUC over the test drivers (LOSO, 14 drivers).
The clips are resampled with replacement inside every test driver, separately for the normal
and for the anomalous clips (all windows of a clip stay together, since the windows of one
clip are strongly dependent). In every replicate, the AUC of every driver is computed from
the resampled windows, averaged over the runs of the method and then over the drivers. The
same resampled clips are used for all methods, so that the intervals of the differences
between two methods are paired. The drivers themselves are not resampled, i.e. the intervals
describe the uncertainty that comes from the limited number of clips of the 14 drivers and
not the uncertainty of the selection of the drivers.
  python bootstrap.py [number of replicates, default 2000]"""
import sys
import numpy as np
from scipy.stats import rankdata
from load import DRIVERS
from ablation import NEW, load

B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
METHODS = [("OC-SVM, Selected", "geom_da_mean|ocsvm"), ("OC-SVM, Exhaustive", "exh_mean|ocsvm"),
           ("OC-SVM, Geometry", "geom_mean|ocsvm"), ("TCN, Selected", "tcn_geom_da"),
           ("TCN, Exhaustive", "tcn_exh"), ("TCN, Geometry", "tcn_geom")]
PAIRS = [("OC-SVM, Selected", "TCN, Selected"), ("OC-SVM, Exhaustive", "TCN, Exhaustive"),
         ("OC-SVM, Selected", "TCN, Exhaustive"), ("OC-SVM, Selected", "OC-SVM, Exhaustive"),
         ("TCN, Selected", "TCN, Geometry"), ("OC-SVM, Selected", "OC-SVM, Geometry")]


def auc_w(s, y, w):
    """AUC with integer weights of the windows (the multiplicity of their clip)."""
    keep = w > 0
    s, y, w = s[keep], y[keep], w[keep].astype(np.float64)
    r = rankdata(np.repeat(s, w.astype(int)))            # ranks with the multiplicities expanded
    yy = np.repeat(y, w.astype(int))
    n1, n0 = yy.sum(), (1 - yy).sum()
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


rng = np.random.default_rng(0)
rep = {name: np.zeros((B, len(DRIVERS))) for name, _ in METHODS}
point = {name: np.zeros(len(DRIVERS)) for name, _ in METHODS}
for di, d in enumerate(DRIVERS):
    data = {}
    ref = None
    for name, key in METHODS:
        runs = []
        for r in NEW[key]:
            z = load(r["scores"], d)
            clips, y = z["clips"].astype(str), z["labels"].astype(int)
            if ref is None:
                ref = (clips, y)
            assert len(clips) == len(ref[0]) and (clips == ref[0]).all() and (y == ref[1]).all(), (name, d)
            runs.append(z["scores"].astype(np.float64))
        data[name] = runs
    clips, y = ref
    names, inv = np.unique(clips, return_inverse=True)
    lab = np.array([y[inv == i][0] for i in range(len(names))])
    idx0, idx1 = np.where(lab == 0)[0], np.where(lab == 1)[0]
    ones = np.ones(len(y))
    for name, _ in METHODS:
        point[name][di] = np.mean([auc_w(s, y, ones) for s in data[name]])
    for b in range(B):
        cnt = np.zeros(len(names), int)
        np.add.at(cnt, rng.choice(idx0, len(idx0)), 1)
        np.add.at(cnt, rng.choice(idx1, len(idx1)), 1)
        w = cnt[inv]
        for name, _ in METHODS:
            rep[name][b, di] = np.mean([auc_w(s, y, w) for s in data[name]])
    print("driver", di + 1, "clips", len(names), "windows", len(y), flush=True)

print("\n%% bootstrap over clips, {} replicates: average AUC, 95 % interval (percentiles 2.5 and 97.5)".format(B))
for name, _ in METHODS:
    m = rep[name].mean(axis=1)
    print("  {:20s} {:.4f}  [{:.4f}, {:.4f}]".format(name, point[name].mean(), *np.percentile(m, [2.5, 97.5])))
print("%% paired differences A - B")
for a, b in PAIRS:
    dm = rep[a].mean(axis=1) - rep[b].mean(axis=1)
    print("  {:20s} - {:20s} {:+.4f}  [{:+.4f}, {:+.4f}]  share of replicates > 0: {:.3f}".format(
        a, b, point[a].mean() - point[b].mean(), *np.percentile(dm, [2.5, 97.5]), (dm > 0).mean()))
