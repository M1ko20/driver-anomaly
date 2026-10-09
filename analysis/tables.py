"""Per-driver tables, pooled groups of runs, paired tests, per-action results."""
import sys
import json
import numpy as np
from load import RUNS, DRIVERS, per_driver, run_means
from analyze import paired, wilcoxon_exact

np.set_printoptions(linewidth=220, precision=4, suppress=True)
ENS = json.load(open("ens.json"))

# groups of runs of the same network + features (different scripts / clip sets)
GROUPS = {
    "TCN-C": ["coords_tcn"],
    "TCN-G": ["geom_tcn", "old_geom_tcn", "oldens_geom", "x_tcn_geom_common"],
    "TCN-K": ["kin_tcn"],
    "TCN-P": ["old_proxies_tcn"],
    "TCN-H": ["old_hands_tcn", "oldens_hands", "x_tcn_hands_common"],
    "TCN-HH": ["old_hh_tcn", "oldens_hh", "x_tcn_hh_common"],
    "TCN-E": ["x_tcn_exh"],
    "TCN-G-z4": ["x_tcn_geom_z4"], "TCN-G-z8": ["x_tcn_geom_z8"], "TCN-G-z32": ["x_tcn_geom_z32"],
    "TCN-vitC": ["x_tcn_vit_coords"], "TCN-vitG": ["x_tcn_vit_geom"],
    "TCN-G-T30(L90)": ["x_tcn_geom_T30_minL90"], "TCN-G-T60(L90)": ["x_tcn_geom_minL90"],
    "TCN-G-T90(L90)": ["x_tcn_geom_T90_minL90"],
    "TCN-G-long": ["old_geom_tcn_long"],
    "TCN-forecast": ["forecast_tcn"],
    "Mamba-C": ["mamba_coords"], "Mamba-G": ["old_mamba_geom", "mamba_geom_common"], "Mamba-K": ["mamba_kin"],
    "Mamba-H": ["mamba_hands_common"], "Mamba-HH": ["mamba_hh_common"],
    "STGCN": ["old_stgcn"], "STGCN1-C": ["stgcn_coords"], "STGCN1-K": ["stgcn_kin"],
    "LSTM-G-T12": ["x_lstm_geom_T12_b128"], "LSTM-G-T60": ["x_lstm_geom_b128"], "LSTM-E-T12": ["x_lstm_exh_T12_b128"],
    "PCA-C": ["pca_coords"], "PCA-G": ["pca_geom"], "PCA-K": ["pca_kin"],
    "SVM-C": ["ocsvm_coords"], "SVM-G": ["ocsvm_geom"], "SVM-K": ["ocsvm_kin"],
    "IF-C": ["iforest_coords"], "IF-G": ["iforest_geom"], "IF-K": ["iforest_kin"],
    "Ens-old": ["oldens_mean"], "Ens-old-max": ["oldens_max"],
}


def runs_of(g):
    return [r for n in GROUPS[g] for r in RUNS[n]]


def pd_(g, key="auc"):
    rr = [r for r in runs_of(g) if r.get(key)]
    return np.mean([[r[key][d] for d in DRIVERS] for r in rr], axis=0)


def rm(g):
    return np.array([np.mean([r["auc"][d] for d in DRIVERS]) for r in runs_of(g)])


if __name__ == "__main__":
    print("=== GROUPS: n runs, avg AUC (mean of per-driver means), sd over drivers, min driver, PR-AUC (runs that have it), run means: mean, sd, min, max")
    for g in GROUPS:
        a = pd_(g)
        m = rm(g)
        has_pr = any(r.get("pr") for r in runs_of(g))
        pr = pd_(g, "pr").mean() if has_pr else float("nan")
        print("  {:16s} n={} avg {:.4f} sdDrv {:.4f} min {:.4f} ({}) PR {:.4f} | runs: sd {:.4f} [{:.4f}..{:.4f}]  {}".format(
            g, len(m), a.mean(), a.std(ddof=1), a.min(), "D%d" % (a.argmin() + 1), pr,
            m.std(ddof=1) if len(m) > 1 else float("nan"), m.min(), m.max(), " ".join("%.4f" % x for x in m)))

    print("\n=== PER DRIVER (avg of runs); columns D1..D14, then avg")
    for g in GROUPS:
        a = pd_(g)
        print("  {:16s}".format(g), " ".join("%.4f" % x for x in a), "| %.4f" % a.mean())
    for k in ["TCN G+H+HH|mean", "TCN G+H+HH|max", "TCN 3xG seeds|mean", "Mamba G+H+HH|mean", "TCN G + Mamba G|mean",
              "TCN+Mamba x G,H,HH (6)|mean", "TCN 2x exh seeds|mean", "TCN G + TCN exh|mean"]:
        a = np.array(ENS[k]["auc"])
        print("  {:16s}".format(k[:16]), " ".join("%.4f" % x for x in a), "| %.4f" % a.mean(), k)

    print("\n=== seed variability of the per-driver AUC (TCN-G, 8 runs): sd over runs for each driver")
    M = np.array([[r["auc"][d] for d in DRIVERS] for r in runs_of("TCN-G")])
    print("  sd per driver", M.std(axis=0, ddof=1))
    print("  mean sd over drivers {:.4f}; sd of run means {:.4f}; range of run means {:.4f}..{:.4f}".format(
        M.std(axis=0, ddof=1).mean(), M.mean(axis=1).std(ddof=1), M.mean(axis=1).min(), M.mean(axis=1).max()))
    for g in ["Mamba-G", "TCN-H", "TCN-HH", "TCN-E", "TCN-G-z8", "STGCN"]:
        M2 = np.array([[r["auc"][d] for d in DRIVERS] for r in runs_of(g)])
        print("  {:10s} n={} mean per-driver sd {:.4f}  max per-driver sd {:.4f} (D{})".format(
            g, len(M2), M2.std(axis=0, ddof=1).mean(), M2.std(axis=0, ddof=1).max(), M2.std(axis=0, ddof=1).argmax() + 1))

    print("\n=== PAIRED TESTS (per-driver averages over the available runs; exact two-sided Wilcoxon)")
    P = lambda a, b: paired(pd_(a), pd_(b), a, b)
    P("TCN-G", "Mamba-G"); P("TCN-C", "Mamba-C"); P("TCN-K", "Mamba-K"); P("TCN-H", "Mamba-H"); P("TCN-HH", "Mamba-HH")
    P("TCN-G", "STGCN"); P("Mamba-G", "STGCN"); P("TCN-C", "STGCN1-C")
    P("TCN-G", "LSTM-G-T60"); P("TCN-G", "LSTM-G-T12"); P("LSTM-G-T60", "LSTM-G-T12"); P("TCN-E", "LSTM-E-T12"); P("LSTM-E-T12", "LSTM-G-T12")
    P("TCN-G", "TCN-C"); P("TCN-G", "TCN-K"); P("TCN-C", "TCN-K"); P("TCN-E", "TCN-G"); P("TCN-E", "TCN-C")
    P("TCN-G", "TCN-P"); P("TCN-G", "TCN-H"); P("TCN-G", "TCN-HH")
    P("TCN-G-z8", "TCN-G"); P("TCN-G-z4", "TCN-G"); P("TCN-G-z32", "TCN-G"); P("TCN-G-z8", "TCN-G-z4"); P("TCN-E", "TCN-G-z8")
    P("TCN-G", "TCN-vitG"); P("TCN-C", "TCN-vitC"); P("TCN-vitG", "TCN-vitC")
    P("TCN-G-T90(L90)", "TCN-G-T60(L90)"); P("TCN-G-T30(L90)", "TCN-G-T60(L90)"); P("TCN-G-T90(L90)", "TCN-G-T30(L90)")
    P("TCN-G-long", "old_geom_tcn") if False else None
    P("TCN-G", "SVM-G"); P("TCN-G", "PCA-G"); P("TCN-C", "SVM-C"); P("SVM-G", "SVM-C"); P("PCA-G", "PCA-C"); P("IF-G", "IF-C")
    P("TCN-G", "TCN-forecast")
    print("  -- ensembles (new score files; components = the x_tcn_*_common / mamba_*_common runs, seeds 0 and 1)")
    comp = {"G": pd_("TCN-G")}
    xg = np.mean([[r["auc"][d] for d in DRIVERS] for r in RUNS["x_tcn_geom_common"][:2]], axis=0)
    xg3 = per_driver("x_tcn_geom_common")
    xh = per_driver("x_tcn_hands_common"); xhh = per_driver("x_tcn_hh_common")
    mg = per_driver("mamba_geom_common"); mh = per_driver("mamba_hands_common"); mhh = per_driver("mamba_hh_common")
    E = lambda k: np.array(ENS[k]["auc"])
    paired(E("TCN G+H+HH|mean"), xg, "Ens TCN G+H+HH", "TCN G (same runs)")
    paired(E("TCN G+H+HH|mean"), xh, "Ens TCN G+H+HH", "TCN H (same runs)")
    paired(E("TCN G+H+HH|mean"), xhh, "Ens TCN G+H+HH", "TCN HH (same runs)")
    paired(E("TCN 3xG seeds|mean"), xg3, "Ens TCN 3xG seeds", "TCN G (avg of the 3)")
    paired(E("TCN G+H+HH|mean"), E("TCN 3xG seeds|mean"), "Ens TCN G+H+HH", "Ens TCN 3xG seeds")
    paired(E("TCN G+H+HH|mean"), E("TCN G+H+HH|max"), "Ens TCN mean", "Ens TCN max")
    paired(E("Mamba G+H+HH|mean"), mg, "Ens Mamba G+H+HH", "Mamba G (same runs)")
    paired(E("Mamba G+H+HH|mean"), mh, "Ens Mamba G+H+HH", "Mamba H (same runs)")
    paired(E("Mamba G+H+HH|mean"), mhh, "Ens Mamba G+H+HH", "Mamba HH (same runs)")
    paired(E("Mamba G+H+HH|mean"), E("Mamba 2xG seeds|mean"), "Ens Mamba G+H+HH", "Ens Mamba 2xG seeds")
    paired(E("Mamba 2xG seeds|mean"), mg, "Ens Mamba 2xG seeds", "Mamba G")
    paired(E("TCN G+H+HH|mean"), E("Mamba G+H+HH|mean"), "Ens TCN G+H+HH", "Ens Mamba G+H+HH")
    paired(E("TCN G + Mamba G|mean"), xg, "Ens TCN G + Mamba G", "TCN G (same runs)")
    paired(E("TCN+Mamba x G,H,HH (6)|mean"), E("TCN G+H+HH|mean"), "Ens 6 models", "Ens TCN G+H+HH")
    paired(pd_("TCN-E"), E("TCN G+H+HH|mean"), "TCN-E (single)", "Ens TCN G+H+HH")
    paired(E("TCN 2x exh seeds|mean"), pd_("TCN-E"), "Ens 2x exh seeds", "TCN-E")
    print("  -- old ensemble runs (August)")
    paired(pd_("Ens-old"), per_driver("oldens_geom"), "Ens-old", "oldens_geom")

    print("\n=== EPOCHS (mean over folds and runs; number of folds that reached 100 epochs) and training time per fold [min]")
    for n in RUNS:
        rr = [r for r in RUNS[n] if r.get("epochs")]
        if not rr:
            continue
        ep = np.array([[r["epochs"][d] for d in DRIVERS] for r in rr])
        sec = np.mean([r.get("seconds", 0) for r in rr]) / 14 / 60
        print("  {:26s} mean epochs {:5.1f}  min {:3d} max {:3d}  folds at 100: {}/{}   {:.0f} min/fold".format(
            n, ep.mean(), ep.min(), ep.max(), int((ep >= 100).sum()), ep.size, sec))

    print("\n=== PER ACTION (avg over drivers that have it, then over runs)")
    acts = sorted({a for r in RUNS["geom_tcn"] for d in DRIVERS for a in r["actions"][d]})
    def act_table(g):
        rr = [r for r in runs_of(g) if r.get("actions")]
        out = {}
        for a in acts:
            out[a] = np.mean([np.mean([r["actions"][d][a][0] for d in DRIVERS if a in r["actions"][d]]) for r in rr])
        return out, len(rr)
    hdr = ["TCN-C", "TCN-G", "TCN-K", "TCN-P", "TCN-H", "TCN-HH", "TCN-E", "TCN-G-z8", "TCN-vitG", "Mamba-C", "Mamba-G", "Mamba-H", "Mamba-HH",
           "LSTM-G-T60", "LSTM-E-T12", "STGCN1-C", "Ens-old"]
    tabs = {g: act_table(g) for g in hdr}
    print("  {:22s}".format("action") + " ".join("{:>10s}".format(h[:10]) for h in hdr))
    print("  {:22s}".format("(runs with actions)") + " ".join("{:>10d}".format(tabs[h][1]) for h in hdr))
    for a in acts:
        print("  {:22s}".format(a) + " ".join("{:10.4f}".format(tabs[h][0][a]) for h in hdr))
    r = RUNS["geom_tcn"][0]
    print("  windows per action:", {a: sum(r["actions"][d][a][1] for d in DRIVERS if a in r["actions"][d]) for a in acts})
    print("  drivers per action:", {a: sum(1 for d in DRIVERS if a in r["actions"][d]) for a in acts})
