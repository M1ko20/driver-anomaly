"""Remaining numbers: same-feature ensembles, merged ensemble (old + new runs), per-action
results of ensembles, clip / window counts, thresholds (median), proxies per action."""
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from load import RUNS, DRIVERS, per_driver, TOP, PE
from analyze import ens_eval, ens_scores, sc, paired, action_auc_from_scores
from tables import pd_, runs_of

np.set_printoptions(linewidth=220, precision=4, suppress=True)
X = lambda f, s: "x_tcn_{}_common_seed{}".format(f, s)
M = lambda f, s: "mamba_{}_common_seed{}".format(f, s)

print("=== same-feature (seed) ensembles, mean")
for name, l in [("TCN 2xH", [X("hands", 0), X("hands", 1)]), ("TCN 2xHH", [X("hh", 0), X("hh", 1)]),
                ("Mamba 2xG", [M("geom", 0), M("geom", 1)]), ("Mamba 2xH", [M("hands", 0), M("hands", 1)]),
                ("Mamba 2xHH", [M("hh", 0), M("hh", 1)]),
                ("Mamba 2xC", ["mamba_coords_seed0", "mamba_coords_seed1"]),
                ("TCN 2x z8", ["x_tcn_geom_z8_seed0", "x_tcn_geom_z8_seed1"]),
                ("TCN exh + z8 (s0)", ["x_tcn_exh_seed0", "x_tcn_geom_z8_seed0"]),
                ("TCN exh + z8 (s1)", ["x_tcn_exh_seed1", "x_tcn_geom_z8_seed1"])]:
    a, p, pooled = ens_eval(l, "mean")
    print("  {:20s} avg {:.4f} PR {:.4f} pooled {:.4f}".format(name, a.mean(), p.mean(), pooled))

print("\n=== cross-feature 2-model ensembles with different seeds (TCN): G(s) + H(s'), etc.")
for fa, fb in [("geom", "hands"), ("geom", "hh"), ("hands", "hh")]:
    vals = []
    for sa in (0, 1):
        for sb in (0, 1):
            vals.append(ens_eval([X(fa, sa), X(fb, sb)], "mean")[0].mean())
    print("  TCN {}+{}: {}  mean {:.4f}".format(fa, fb, " ".join("%.4f" % v for v in vals), np.mean(vals)))
for fa, fb in [("geom", "hands"), ("geom", "hh"), ("hands", "hh")]:
    vals = []
    for sa in (0, 1):
        for sb in (0, 1):
            vals.append(ens_eval([M(fa, sa), M(fb, sb)], "mean")[0].mean())
    print("  Mamba {}+{}: {}  mean {:.4f}".format(fa, fb, " ".join("%.4f" % v for v in vals), np.mean(vals)))

print("\n=== 3-model TCN ensembles over all seed combinations (G: 3 seeds, H: 2, HH: 2)")
vals = [ens_eval([X("geom", a), X("hands", b), X("hh", c)], "mean")[0].mean() for a in (0, 1, 2) for b in (0, 1) for c in (0, 1)]
print("  G+H+HH, 12 combinations: mean {:.4f} sd {:.4f} min {:.4f} max {:.4f}".format(np.mean(vals), np.std(vals, ddof=1), min(vals), max(vals)))

print("\n=== merged TCN ensemble G+H+HH: 2 old runs (August) + 2 new runs")
ENS = json.load(open("ens.json"))
new = np.array(ENS["TCN G+H+HH|mean"]["auc_runs"])
old = np.array([[r["auc"][d] for d in DRIVERS] for r in RUNS["oldens_mean"]])
ens4 = np.concatenate([old, new]).mean(axis=0)
g4 = np.concatenate([[[r["auc"][d] for d in DRIVERS] for r in RUNS["oldens_geom"]],
                     [[r["auc"][d] for d in DRIVERS] for r in RUNS["x_tcn_geom_common"][:2]]]).mean(axis=0)
h4 = np.concatenate([[[r["auc"][d] for d in DRIVERS] for r in RUNS["oldens_hands"]],
                     [[r["auc"][d] for d in DRIVERS] for r in RUNS["x_tcn_hands_common"]]]).mean(axis=0)
hh4 = np.concatenate([[[r["auc"][d] for d in DRIVERS] for r in RUNS["oldens_hh"]],
                      [[r["auc"][d] for d in DRIVERS] for r in RUNS["x_tcn_hh_common"]]]).mean(axis=0)
print("  per run means: old", old.mean(axis=1), "new", new.mean(axis=1), " => avg {:.4f}".format(ens4.mean()))
print("  ens4 per driver", ens4)
print("  components (same 4 runs): G {:.4f}  H {:.4f}  HH {:.4f}".format(g4.mean(), h4.mean(), hh4.mean()))
paired(ens4, g4, "Ens4", "G (same 4 runs)")
paired(ens4, h4, "Ens4", "H (same 4 runs)")
paired(ens4, hh4, "Ens4", "HH (same 4 runs)")
paired(ens4, pd_("TCN-G"), "Ens4", "TCN-G (8 runs)")
paired(pd_("TCN-E"), ens4, "TCN-E", "Ens4")
paired(pd_("TCN-G-z8"), ens4, "TCN-G-z8", "Ens4")
newmax = np.array(ENS["TCN G+H+HH|max"]["auc_runs"])
oldmax = np.array([[r["auc"][d] for d in DRIVERS] for r in RUNS["oldens_max"]])
print("  max variant, 4 runs: {:.4f}".format(np.concatenate([oldmax, newmax]).mean()))
json.dump({"ens4": ens4.tolist(), "g4": g4.tolist(), "h4": h4.tolist(), "hh4": hh4.tolist(),
           "ens4_max": float(np.concatenate([oldmax, newmax]).mean()),
           "ens4_runs": np.concatenate([old, new]).mean(axis=1).tolist()}, open("ens4.json", "w"))

print("\n=== per-action AUC of ensembles from the new score files (avg over drivers, then seeds)")
def ens_fn(prefixes):
    def f(d):
        z = sc(prefixes[0], d)
        return ens_scores(prefixes, d, "mean"), z["labels"], z["actions"]
    return f
res = {}
for name, lists in [("TCN G+H+HH", [[X("geom", s), X("hands", s), X("hh", s)] for s in (0, 1)]),
                    ("TCN 3xG", [[X("geom", 0), X("geom", 1), X("geom", 2)]]),
                    ("Mamba G+H+HH", [[M("geom", s), M("hands", s), M("hh", s)] for s in (0, 1)])]:
    per = [action_auc_from_scores(ens_fn(l)) for l in lists]
    res[name] = {a: float(np.mean([np.mean(list(p[a].values())) for p in per])) for a in per[0]}
    print("  " + name, {a: round(v, 4) for a, v in res[name].items()})
# old ensemble per action
oldact = {}
for s in (0, 1):
    d = json.load(open(os.path.join(TOP, "loso_ensemble_v2_seed{}_results.json".format(s))))
    for drv, v in d["per_driver_diagnostic"].items():
        for a, b in v["action_breakdown"].items():
            oldact.setdefault(a, {}).setdefault(s, []).append(b)
k = next(iter(oldact))
print("  old ensemble action entry example:", k, oldact[k][0][0])
json.dump(res, open("ens_actions.json", "w"), indent=1)

print("\n=== proxies per action (old runs, action_summary)")
for s in (0, 1):
    d = json.load(open(os.path.join(TOP, "loso_delta_v2_seed{}_results.json".format(s))))
    print("  seed", s, json.dumps(d["action_summary"])[:700])

print("\n=== clips and windows")
allc = {}
for d in DRIVERS:
    a = sc("x_tcn_exh_seed0", d); b = sc(X("geom", 0), d)
    ca, cb = set(a["clips"].tolist()), set(b["clips"].tolist())
    if ca != cb:
        miss = sorted(ca - cb)
        print("  driver", d, "(Driver-%d)" % (DRIVERS.index(d) + 1), "clips not in the common set:", miss,
              "windows", int(np.isin(a["clips"], miss).sum()), "labels", set(a["labels"][np.isin(a["clips"], miss)].tolist()))
for pre, lab in [("x_tcn_geom_minL90_seed0", "T60 minL90"), ("x_tcn_geom_T30_minL90_seed0", "T30 minL90"),
                 ("x_tcn_geom_T90_minL90_seed0", "T90 minL90"), ("x_lstm_geom_T12_b128_seed0", "LSTM T12"),
                 ("x_tcn_exh_seed0", "full T60"), ("x_tcn_vit_geom_seed0", "vit")]:
    n0 = n1 = c0 = c1 = 0
    for d in DRIVERS:
        z = sc(pre, d)
        n0 += int((z["labels"] == 0).sum()); n1 += int((z["labels"] == 1).sum())
        cl = {}
        for c, y in zip(z["clips"].tolist(), z["labels"].tolist()):
            cl[c] = y
        c0 += sum(1 for v in cl.values() if v == 0); c1 += sum(1 for v in cl.values() if v == 1)
    print("  {:12s} normal windows {} anomalous {} | clips normal {} anomalous {}".format(lab, n0, n1, c0, c1))
r = RUNS["old_geom_tcn_long"][0]
print("  long window (30 frames, step 3): test windows", sum(r["nwin"].values()))

print("\n=== threshold (val p95): per-driver FPR/TPR, mean / median / number of drivers with FPR<=0.10")
for n in ["x_tcn_geom_common", "x_tcn_exh", "x_tcn_geom_z8", "x_tcn_geom_z4", "x_lstm_geom_b128", "x_tcn_hands_common", "x_tcn_hh_common", "x_tcn_vit_geom"]:
    F, T = [], []
    for r in RUNS[n]:
        f_, t_ = [], []
        for d in DRIVERS:
            z = sc(r["scores"], d)
            pred = z["scores"] > float(z["val_p95"]); y = z["labels"] == 1
            f_.append((pred & ~y).sum() / (~y).sum()); t_.append((pred & y).sum() / y.sum())
        F.append(f_); T.append(t_)
    F, T = np.mean(F, axis=0), np.mean(T, axis=0)
    print("  {:20s} FPR mean {:.3f} median {:.3f} max {:.3f} (D{}) n<=0.10: {} | TPR mean {:.3f} median {:.3f} min {:.3f} (D{})".format(
        n, F.mean(), np.median(F), F.max(), F.argmax() + 1, int((F <= 0.10).sum()), T.mean(), np.median(T), T.min(), T.argmin() + 1))
    print("      FPR", F); print("      TPR", T)
