"""Generates the bodies of all result tables of the manuscript (tex/<name>.tex) and a
JSON with the scalars that are quoted in the text (scalars.json)."""
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from load import RUNS, DRIVERS, per_driver, TOP
from analyze import ens_eval, ens_scores, sc, wilcoxon_exact, pooled_single, action_auc_from_scores
from tables import GROUPS, pd_, runs_of, rm

OUT = "tex"
os.makedirs(OUT, exist_ok=True)
SC = {}
f4 = lambda x: "{:.4f}".format(x)
f3 = lambda x: "{:.3f}".format(x)
sg = lambda x: ("+" if x >= 0 else "$-$") + "{:.4f}".format(abs(x))
sd = lambda a: float(np.std(a, ddof=1))
X = lambda f, s: "x_tcn_{}_common_seed{}".format(f, s)
M = lambda f, s: "mamba_{}_common_seed{}".format(f, s)
E4 = json.load(open("ens4.json"))
ENS = json.load(open("ens.json"))


def write(name, lines):
    open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(lines) + "\n")
    print("---", name)
    print("\n".join(lines))


def pr_of(g):
    rr = [r for r in runs_of(g) if r.get("pr")]
    return (float(np.mean([[r["pr"][d] for d in DRIVERS] for r in rr])), len(rr)) if rr else (None, 0)


def bold(s, cond):
    return "\\textbf{" + s + "}" if cond else s


# ----------------------------------------------------------------- grid C / G / K
rows = [("TCN", "TCN-C", "TCN-G", "TCN-K"), ("Mamba", "Mamba-C", "Mamba-G", "Mamba-K"),
        ("TCN (forecast.)", None, "TCN-forecast", None), ("ST-GCN (first)", "STGCN1-C", None, "STGCN1-K"), "sep",
        ("PCA", "PCA-C", "PCA-G", "PCA-K"), ("OC-SVM", "SVM-C", "SVM-G", "SVM-K"), ("Forest", "IF-C", "IF-G", "IF-K")]
best_auc = max(pd_(g).mean() for r in rows if r != "sep" for g in r[1:] if g)
best_pr = max(pr_of(g)[0] for r in rows if r != "sep" for g in r[1:] if g)
L = []
for i, r in enumerate(rows):
    if r == "sep":
        continue
    a = [bold(f4(pd_(g).mean()), abs(pd_(g).mean() - best_auc) < 1e-12) if g else "--" for g in r[1:]]
    p = [bold(f4(pr_of(g)[0]), abs(pr_of(g)[0] - best_pr) < 1e-12) if g else "--" for g in r[1:]]
    end = "\\hline\\hline" if i + 1 < len(rows) and rows[i + 1] == "sep" else "\\hline"
    L.append("{} & {} & {} \\\\ {}".format(r[0], " & ".join(a), " & ".join(p), end))
write("grid", L)

# ----------------------------------------------------------------- master summary
ens_rows = {}
a3, p3, pool3 = ens_eval([X("geom", 0), X("geom", 1), X("geom", 2)])
am = [ens_eval([M("geom", s), M("hands", s), M("hh", s)]) for s in (0, 1)]
atm = [ens_eval([X("geom", s), M("geom", s)]) for s in (0, 1)]
a6 = [ens_eval([X("geom", s), X("hands", s), X("hh", s), M("geom", s), M("hands", s), M("hh", s)]) for s in (0, 1)]
anew = [ens_eval([X("geom", s), X("hands", s), X("hh", s)]) for s in (0, 1)]
ae = ens_eval(["x_tcn_exh_seed0", "x_tcn_exh_seed1"])
ENSPD = {"ens4": np.array(E4["ens4"]), "ens3g": a3, "ensM": np.mean([x[0] for x in am], axis=0),
         "ensTM": np.mean([x[0] for x in atm], axis=0), "ens6": np.mean([x[0] for x in a6], axis=0)}
ENSRUNS = {"ens4": E4["ens4_runs"], "ens3g": [a3.mean()], "ensM": [x[0].mean() for x in am],
           "ensTM": [x[0].mean() for x in atm], "ens6": [x[0].mean() for x in a6]}
ENSPR = {"ens4": np.mean([x[1].mean() for x in anew]), "ens3g": p3.mean(), "ensM": np.mean([x[1].mean() for x in am]),
         "ensTM": np.mean([x[1].mean() for x in atm]), "ens6": np.mean([x[1].mean() for x in a6])}

master = [
    ("TCN", "Coordinates", "TCN-C"), ("TCN", "Geometry", "TCN-G"), ("TCN", "Kinematics", "TCN-K"),
    ("TCN", "Proxies", "TCN-P"), ("TCN", "Hands", "TCN-H"), ("TCN", "Hands+Head", "TCN-HH"),
    ("TCN", "Exhaustive", "TCN-E"), "sep",
    ("TCN (4 channels)", "Geometry", "TCN-G-z4"), ("TCN (8 channels)", "Geometry", "TCN-G-z8"),
    ("TCN (32 channels)", "Geometry", "TCN-G-z32"), ("TCN (long window)", "Geometry", "TCN-G-long"),
    ("TCN (forecasting)", "Geometry", "TCN-forecast"), "sep",
    ("Mamba", "Coordinates", "Mamba-C"), ("Mamba", "Geometry", "Mamba-G"), ("Mamba", "Kinematics", "Mamba-K"),
    ("Mamba", "Hands", "Mamba-H"), ("Mamba", "Hands+Head", "Mamba-HH"), "sep",
    ("ST-GCN", "Geometry (graph)", "STGCN"), ("ST-GCN (first)", "Coordinates", "STGCN1-C"),
    ("ST-GCN (first)", "Kinematics", "STGCN1-K"), "sep",
    ("$\\text{LSTM}_{\\text{extend}}(12t)$", "Geometry", "LSTM-G-T12"),
    ("$\\text{LSTM}_{\\text{extend}}(60t)$", "Geometry", "LSTM-G-T60"),
    ("$\\text{LSTM}_{\\text{extend}}(12t)$", "Exhaustive", "LSTM-E-T12"), "sep",
    ("TCN", "Coordinates (ViTPose)", "TCN-vitC"), ("TCN", "Geometry (ViTPose)", "TCN-vitG"), "sep",
    ("Ens. (3$\\times$ TCN)", "G, H, HH", "ens4"),
    ("Ens. (3$\\times$ TCN)", "Geometry (three runs)", "ens3g"),
    ("Ens. (3$\\times$ Mamba)", "G, H, HH", "ensM"),
    ("Ens. (TCN + Mamba)", "Geometry", "ensTM"),
    ("Ens. (3$\\times$ TCN + 3$\\times$ Mamba)", "G, H, HH", "ens6"),
]
L = []
vals = {}
for i, r in enumerate(master):
    if r == "sep":
        continue
    net, feat, g = r
    if g in ENSPD:
        a = ENSPD[g]; runs = ENSRUNS[g]; pr = ENSPR[g]
        nruns = {"ens4": 4, "ens3g": 1, "ensM": 2, "ensTM": 2, "ens6": 2}[g]
    else:
        a = pd_(g); runs = rm(g); pr = pr_of(g)[0]; nruns = len(runs)
    vals[g] = a
    rng = "{}--{}".format(f4(min(runs)), f4(max(runs))) if nruns > 1 else "--"
    end = "\\hline\\hline" if i + 1 < len(master) and master[i + 1] == "sep" else "\\hline"
    L.append("{} & {} & {} & {} & {} & {} ({}) & {} & {} \\\\ {}".format(
        net, feat, nruns, "@AUC{}@".format(len(L)), f4(sd(a)), f4(a.min()), int(a.argmin()) + 1,
        f4(pr) if pr is not None else "--", rng, end))
best = max(v.mean() for v in vals.values())
k = 0
for i, r in enumerate([r for r in master if r != "sep"]):
    a = vals[r[2]]
    L[i] = L[i].replace("@AUC{}@".format(i), bold(f4(a.mean()), abs(a.mean() - best) < 1e-12))
write("master", L)

# ----------------------------------------------------------------- per-driver: networks
def per_driver_table(name, cols, boldbest=True):
    arrs = [c if isinstance(c, np.ndarray) else pd_(c) for c in cols]
    L = []
    for i in range(14):
        L.append("Driver-{} & {} \\\\ \\hline{}".format(i + 1, " & ".join(f4(a[i]) for a in arrs), "\\hline" if i == 13 else ""))
    best = max(a.mean() for a in arrs)
    L.append("avg. & {} \\\\ \\hline".format(" & ".join(bold(f4(a.mean()), boldbest and abs(a.mean() - best) < 1e-12) for a in arrs)))
    L.append("s.d. & {} \\\\ \\hline".format(" & ".join(f4(sd(a)) for a in arrs)))
    write(name, L)


per_driver_table("networks", ["TCN-G", "Mamba-G", "STGCN", "LSTM-G-T12", "LSTM-G-T60"])
per_driver_table("features", ["TCN-C", "TCN-G", "TCN-K", "TCN-P", "TCN-H", "TCN-HH", "TCN-E", "LSTM-E-T12"])

# ----------------------------------------------------------------- ablation (bottleneck, window)
L = []
for lab, g in [("4 channels", "TCN-G-z4"), ("8 channels", "TCN-G-z8"), ("16 channels", "TCN-G"), ("32 channels", "TCN-G-z32")]:
    a = pd_(g)
    L.append("{} & {} & {} & {} & {} ({}) & {} \\\\ \\hline".format(lab, len(rm(g)), f4(a.mean()), f4(sd(a)), f4(a.min()), a.argmin() + 1, f4(pr_of(g)[0])))
L[-1] += "\\hline"
for lab, g in [("30 frames", "TCN-G-T30(L90)"), ("60 frames", "TCN-G-T60(L90)"), ("90 frames", "TCN-G-T90(L90)")]:
    a = pd_(g)
    L.append("{} & {} & {} & {} & {} ({}) & {} \\\\ \\hline".format(lab, len(rm(g)), f4(a.mean()), f4(sd(a)), f4(a.min()), a.argmin() + 1, f4(pr_of(g)[0])))
write("ablation", L)
SC["params"] = {g: RUNS[GROUPS[g][-1]][0]["n_params"] for g in ["TCN-G-z4", "TCN-G-z8", "TCN-G-z32", "TCN-E", "LSTM-G-T60", "LSTM-G-T12", "LSTM-E-T12", "Mamba-C", "Mamba-K", "Mamba-H", "Mamba-HH", "Mamba-G"]}

# ----------------------------------------------------------------- ensembles
def comp_mean(prefixes):
    out = []
    for p in prefixes:
        for n in RUNS:
            for r in RUNS[n]:
                if r.get("scores") == p:
                    out.append(np.mean([r["auc"][d] for d in DRIVERS]))
    assert len(out) == len(prefixes), prefixes
    return float(np.mean(out))


def ens_row(net, feats, n, lists, mode="mean", sep=False):
    res = [ens_eval(l, mode) for l in lists]
    e = float(np.mean([r[0].mean() for r in res]))
    c = float(np.mean([comp_mean(l) for l in lists]))
    pooled = float(np.mean([r[2] for r in res]))
    return "{} & {} & {} & {} & {} & {} & {} & {} \\\\ \\hline{}".format(net, feats, n, len(lists), f4(c), f4(e), sg(e - c), f4(pooled), "\\hline" if sep else ""), e, c


G = lambda s: X("geom", s); H = lambda s: X("hands", s); HH = lambda s: X("hh", s)
MG = lambda s: M("geom", s); MH = lambda s: M("hands", s); MHH = lambda s: M("hh", s)
L = []
# merged 4-run ensemble (old + new)
c4 = float(np.mean([np.mean(E4["g4"]), np.mean(E4["h4"]), np.mean(E4["hh4"])]))
e4 = float(np.mean(E4["ens4"]))
old_pooled = [r["pooled"] for r in RUNS["oldens_mean"]]
p4 = float(np.mean(old_pooled + [x[2] for x in anew]))
L.append("TCN & Geometry, Hands, Hands+Head & 3 & 4 & {} & {} & {} & {} \\\\ \\hline".format(f4(c4), f4(e4), sg(e4 - c4), f4(p4)))
oldmax_pooled = [r["pooled"] for r in RUNS["oldens_max"]]
newmax = [ens_eval([G(s), H(s), HH(s)], "max") for s in (0, 1)]
L.append("TCN & Geometry, Hands, Hands+Head (max.) & 3 & 4 & {} & {} & {} & {} \\\\ \\hline".format(
    f4(c4), f4(E4["ens4_max"]), sg(E4["ens4_max"] - c4), f4(float(np.mean(oldmax_pooled + [x[2] for x in newmax])))))
SC["ens4"] = {"avg": e4, "comp": c4, "pooled": p4, "runs": E4["ens4_runs"], "max": E4["ens4_max"],
              "g4": float(np.mean(E4["g4"])), "h4": float(np.mean(E4["h4"])), "hh4": float(np.mean(E4["hh4"]))}
for net, feats, n, lists, sep in [
    ("TCN", "Geometry, Hands", 2, [[G(s), H(s)] for s in (0, 1)], False),
    ("TCN", "Geometry, Hands+Head", 2, [[G(s), HH(s)] for s in (0, 1)], False),
    ("TCN", "Hands, Hands+Head", 2, [[H(s), HH(s)] for s in (0, 1)], True),
    ("TCN", "Geometry (different runs)", 3, [[G(0), G(1), G(2)]], False),
    ("TCN", "Geometry (different runs)", 2, [[G(0), G(1)], [G(0), G(2)], [G(1), G(2)]], False),
    ("TCN", "Hands (different runs)", 2, [[H(0), H(1)]], False),
    ("TCN", "Hands+Head (different runs)", 2, [[HH(0), HH(1)]], False),
    ("TCN", "Exhaustive (different runs)", 2, [["x_tcn_exh_seed0", "x_tcn_exh_seed1"]], True),
    ("Mamba", "Geometry, Hands, Hands+Head", 3, [[MG(s), MH(s), MHH(s)] for s in (0, 1)], False),
    ("Mamba", "Geometry, Hands", 2, [[MG(s), MH(s)] for s in (0, 1)], False),
    ("Mamba", "Geometry, Hands+Head", 2, [[MG(s), MHH(s)] for s in (0, 1)], False),
    ("Mamba", "Hands, Hands+Head", 2, [[MH(s), MHH(s)] for s in (0, 1)], False),
    ("Mamba", "Geometry (different runs)", 2, [[MG(0), MG(1)]], False),
    ("Mamba", "Hands (different runs)", 2, [[MH(0), MH(1)]], False),
    ("Mamba", "Hands+Head (different runs)", 2, [[MHH(0), MHH(1)]], True),
    ("TCN + Mamba", "Geometry", 2, [[G(s), MG(s)] for s in (0, 1)], False),
    ("TCN + Mamba", "Geometry, Hands, Hands+Head", 6, [[G(s), H(s), HH(s), MG(s), MH(s), MHH(s)] for s in (0, 1)], False),
]:
    row, e, c = ens_row(net, feats, n, lists, sep=sep)
    L.append(row)
    SC.setdefault("ens", {})["{}|{}|{}".format(net, feats, n)] = {"ens": e, "comp": c}
write("ensembles", L)

# ----------------------------------------------------------------- paired tests
def test_row(A, B, a, b):
    d = np.asarray(a) - np.asarray(b)
    _, _, p = wilcoxon_exact(a, b)
    ps = "$<$0.001" if p < 0.001 else f3(p)
    return "{} & {} & {} & {} & {} \\\\ \\hline".format(A, B, sg(d.mean()), int((d > 0).sum()), ps), p


xg2 = np.mean([[r["auc"][d] for d in DRIVERS] for r in RUNS["x_tcn_geom_common"][:2]], axis=0)
xg3 = per_driver("x_tcn_geom_common")
mg = per_driver("mamba_geom_common")
mens = ENSPD["ensM"]
m2g = ens_eval([MG(0), MG(1)])[0]
T1 = [("TCN, E", "TCN, G", pd_("TCN-E"), pd_("TCN-G")),
      ("TCN, E", "TCN, C", pd_("TCN-E"), pd_("TCN-C")),
      ("TCN, G", "TCN, C", pd_("TCN-G"), pd_("TCN-C")),
      ("TCN, G", "TCN, K", pd_("TCN-G"), pd_("TCN-K")),
      ("TCN, G", "TCN, P", pd_("TCN-G"), pd_("TCN-P")),
      ("TCN, G", "TCN, H", pd_("TCN-G"), pd_("TCN-H")),
      ("TCN, G", "TCN, HH", pd_("TCN-G"), pd_("TCN-HH")),
      ("OC-SVM, G", "OC-SVM, C", pd_("SVM-G"), pd_("SVM-C")),
      ("TCN, G", "OC-SVM, G", pd_("TCN-G"), pd_("SVM-G")),
      ("TCN, G", "TCN, G (ViTPose)", pd_("TCN-G"), pd_("TCN-vitG")),
      ("TCN, C", "TCN, C (ViTPose)", pd_("TCN-C"), pd_("TCN-vitC")),
      "sep",
      ("TCN (8 channels)", "TCN (16 channels)", pd_("TCN-G-z8"), pd_("TCN-G")),
      ("TCN (4 channels)", "TCN (16 channels)", pd_("TCN-G-z4"), pd_("TCN-G")),
      ("TCN (32 channels)", "TCN (16 channels)", pd_("TCN-G-z32"), pd_("TCN-G")),
      ("TCN, E", "TCN (8 channels)", pd_("TCN-E"), pd_("TCN-G-z8")),
      ("TCN, 90 frames", "TCN, 60 frames", pd_("TCN-G-T90(L90)"), pd_("TCN-G-T60(L90)")),
      ("TCN, 30 frames", "TCN, 60 frames", pd_("TCN-G-T30(L90)"), pd_("TCN-G-T60(L90)"))]
T2 = [("TCN, C", "Mamba, C", pd_("TCN-C"), pd_("Mamba-C")),
      ("TCN, G", "Mamba, G", pd_("TCN-G"), pd_("Mamba-G")),
      ("TCN, K", "Mamba, K", pd_("TCN-K"), pd_("Mamba-K")),
      ("TCN, H", "Mamba, H", pd_("TCN-H"), pd_("Mamba-H")),
      ("TCN, HH", "Mamba, HH", pd_("TCN-HH"), pd_("Mamba-HH")),
      ("TCN, G", "ST-GCN", pd_("TCN-G"), pd_("STGCN")),
      ("Mamba, G", "ST-GCN", pd_("Mamba-G"), pd_("STGCN")),
      ("TCN, G", "LSTM$(60t)$, G", pd_("TCN-G"), pd_("LSTM-G-T60")),
      ("TCN, G", "LSTM$(12t)$, G", pd_("TCN-G"), pd_("LSTM-G-T12")),
      ("TCN, E", "LSTM$(12t)$, E", pd_("TCN-E"), pd_("LSTM-E-T12")),
      ("LSTM$(12t)$, E", "LSTM$(12t)$, G", pd_("LSTM-E-T12"), pd_("LSTM-G-T12")),
      "sep",
      ("Ensemble", "TCN, G", np.array(E4["ens4"]), np.array(E4["g4"])),
      ("Ensemble", "TCN, H", np.array(E4["ens4"]), np.array(E4["h4"])),
      ("Ensemble", "TCN, HH", np.array(E4["ens4"]), np.array(E4["hh4"])),
      ("Ens. (3$\\times$ G)", "TCN, G", a3, xg3),
      ("Ensemble", "Ens. (3$\\times$ G)", anew_pd := np.mean([x[0] for x in anew], axis=0), a3),
      ("Ens. (3$\\times$ Mamba)", "Mamba, G", mens, mg),
      ("Ens. (3$\\times$ Mamba)", "Ens. (2$\\times$ Mamba, G)", mens, m2g),
      ("TCN, E", "Ensemble", pd_("TCN-E"), np.array(E4["ens4"]))]
SC["tests"] = {}
for nm, T in [("tests_a", T1), ("tests_b", T2)]:
    L = []
    for i, t in enumerate(T):
        if t == "sep":
            L[-1] += "\\hline"
            continue
        row, p = test_row(*t)
        L.append(row)
        d = t[2] - t[3]
        SC["tests"]["{} | {}".format(t[0], t[1])] = {"diff": float(d.mean()), "wins": int((d > 0).sum()), "p": p}
    write(nm, L)
SC["n_tests"] = len(SC["tests"])

# ----------------------------------------------------------------- individual types of anomalies
INFO = {"reach_backseat": ("reaching to the back seat", "32 / 32"), "reach_side": ("reaching to the side", "514 / 269"),
        "phonecall_left": ("phone call (left)", "32 / 32"), "hair_and_makeup": ("hair and makeup", "71 / 55"),
        "phonecall_right": ("phone call (right)", "33 / 33"), "drinking": ("drinking", "76 / 73"),
        "texting_right": ("texting (right)", "44 / 32"), "radio": ("operating the radio", "78 / 45"),
        "talking_to_passenger": ("talking to the passenger", "242 / 20"), "texting_left": ("texting (left)", "43 / 32")}


def act_of(g):
    rr = [r for r in runs_of(g) if r.get("actions")]
    return {a: float(np.mean([np.mean([r["actions"][d][a][0] for d in DRIVERS if a in r["actions"][d]]) for r in rr])) for a in INFO}, len(rr)


# ensemble: 2 old runs (stored breakdown) + 2 new runs (from the scores)
ens_act_runs = []
for s in (0, 1):
    d = json.load(open(os.path.join(TOP, "loso_ensemble_v2_seed{}_results.json".format(s))))["per_driver_diagnostic"]
    ens_act_runs.append({a: float(np.mean([d[k]["action_breakdown"][a]["auc"] for k in DRIVERS if a in d[k]["action_breakdown"]])) for a in INFO})
for s in (0, 1):
    def fn(drv, s=s):
        z = sc(G(s), drv)
        return ens_scores([G(s), H(s), HH(s)], drv), z["labels"], z["actions"]
    per = action_auc_from_scores(fn)
    ens_act_runs.append({a: float(np.mean(list(per[a].values()))) for a in INFO})
ens_act = {a: float(np.mean([r[a] for r in ens_act_runs])) for a in INFO}
cols = ["TCN-G", "TCN-E", "TCN-G-z8", "TCN-H", "TCN-HH", "Mamba-G"]
acts = {g: act_of(g) for g in cols}
SC["actions_runs"] = {g: acts[g][1] for g in cols}
r0 = RUNS["geom_tcn"][0]
nwin = {a: sum(r0["actions"][d][a][1] for d in DRIVERS if a in r0["actions"][d]) for a in INFO}
ndrv = {a: sum(1 for d in DRIVERS if a in r0["actions"][d]) for a in INFO}
order = sorted(INFO, key=lambda a: -acts["TCN-G"][0][a])
L = []
for a in order:
    v = [acts[g][0][a] for g in cols] + [ens_act[a]]
    b = max(v)
    L.append("{} & {} & {} & {} & {} \\\\ \\hline".format(INFO[a][0], INFO[a][1], nwin[a], ndrv[a],
                                                        " & ".join(bold(f4(x), abs(x - b) < 1e-12) for x in v)))
write("categories", L)
SC["actions"] = {g: acts[g][0] for g in cols}
SC["actions"]["Ensemble"] = ens_act

# ----------------------------------------------------------------- fixed threshold
def thr(name):
    F, T = [], []
    for r in RUNS[name]:
        f_, t_ = [], []
        for d in DRIVERS:
            z = sc(r["scores"], d)
            pred = z["scores"] > float(z["val_p95"]); y = z["labels"] == 1
            f_.append((pred & ~y).sum() / (~y).sum()); t_.append((pred & y).sum() / y.sum())
        F.append(f_); T.append(t_)
    return np.mean(F, axis=0), np.mean(T, axis=0)


L = []
SC["thr"] = {}
for lab, n in [("TCN, Geometry", "x_tcn_geom_common"), ("TCN, Hands+Head", "x_tcn_hh_common"), ("TCN (8 channels), Geometry", "x_tcn_geom_z8"),
               ("TCN, Exhaustive", "x_tcn_exh"), ("$\\text{LSTM}_{\\text{extend}}(60t)$, Geometry", "x_lstm_geom_b128")]:
    F, T = thr(n)
    pooled = float(np.mean([pooled_single(r["scores"])[0] for r in RUNS[n]]))
    pooledz = float(np.mean([pooled_single(r["scores"])[1] for r in RUNS[n]]))
    L.append("{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \\hline".format(
        lab, f4(pooled), f4(pooledz), f3(F.mean()), f3(np.median(F)), f3(F.max()), f3(T.mean()), f3(np.median(T)), f3(T.min())))
    SC["thr"][n] = {"pooled": pooled, "pooled_norm": pooledz, "F": F.tolist(), "T": T.tolist(), "n_fpr_le_010": int((F <= 0.10).sum())}
write("threshold", L)
SC["pooled"] = {n: float(np.mean([pooled_single(r["scores"])[0] for r in RUNS[n]])) for n in
                ["mamba_geom_common", "mamba_coords", "x_tcn_hands_common", "x_tcn_vit_geom", "x_lstm_geom_T12_b128", "x_lstm_exh_T12_b128"]}
SC["pooled"]["old_geom_tcn"] = RUNS["old_geom_tcn"][0]["pooled"]

# ratio of the mean score of the normal windows of the test driver to that of the validation driver
ratio = []
for r in RUNS["x_tcn_geom_common"]:
    row = []
    for d in DRIVERS:
        z = sc(r["scores"], d)
        row.append(float(z["scores"][z["labels"] == 0].mean() / float(z["val_mean"])))
    ratio.append(row)
SC["normal_ratio_geom"] = np.mean(ratio, axis=0).tolist()
ratio = []
for r in RUNS["x_tcn_exh"]:
    row = []
    for d in DRIVERS:
        z = sc(r["scores"], d)
        row.append(float(z["scores"][z["labels"] == 0].mean() / float(z["val_mean"])))
    ratio.append(row)
SC["normal_ratio_exh"] = np.mean(ratio, axis=0).tolist()

# ----------------------------------------------------------------- misc scalars
Mx = np.array([[r["auc"][d] for d in DRIVERS] for r in runs_of("TCN-G")])
SC["tcn_g"] = {"runs": Mx.mean(axis=1).tolist(), "sd_runs": float(Mx.mean(axis=1).std(ddof=1)),
               "sd_per_driver": Mx.std(axis=0, ddof=1).tolist(), "mean_sd_per_driver": float(Mx.std(axis=0, ddof=1).mean())}
SC["avg"] = {g: float(pd_(g).mean()) for g in GROUPS}
SC["sd"] = {g: sd(pd_(g)) for g in GROUPS}
SC["pr"] = {g: pr_of(g)[0] for g in GROUPS}
SC["pd"] = {g: pd_(g).tolist() for g in GROUPS}
json.dump(SC, open("scalars.json", "w"), indent=1)
print("\nn tests", SC["n_tests"], " bonferroni 0.05/n =", 0.05 / SC["n_tests"])
print("normal ratio geom", np.round(SC["normal_ratio_geom"], 2))
print("normal ratio exh ", np.round(SC["normal_ratio_exh"], 2))
print("params", SC["params"])
