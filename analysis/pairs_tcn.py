"""Paired comparisons (exact Wilcoxon test over the 14 drivers) among the TCN ablation runs of
paper_ablation.py and the classical methods, and the summary of the autoencoder trained on the
averaged vectors (--pool mean). Used for Section 4.4 of the article.  python pairs_tcn.py"""
import numpy as np
from ablation import NEW, pd, paired, per_action
from tables import pd_ as old_pd
P = [("tcn_geom_da","tcn_geom"),("tcn_geom_da","tcn_exh"),("tcn_geom_nc","tcn_geom"),("tcn_geom_da","tcn_geom_nc"),
     ("tcn_geom_k1","tcn_geom"),("tcn_exh_k1","tcn_exh"),("tcn_geom_da_k1","tcn_geom_da"),
     ("tcn_exh_z4","tcn_exh"),("tcn_exh_z8","tcn_exh"),("tcn_exh_z32","tcn_exh"),("tcn_exh_conf","tcn_exh"),("tcn_exh_xy","tcn_exh"),
     ("tcn_ang286","tcn_exh"),("tcn_dist78","tcn_exh"),
     ("geom_da_mean|ocsvm","tcn_geom_da"),("exh_mean|ocsvm","tcn_exh"),("exh_mean|ocsvm","tcn_exh_k1"),("geom_da_mean|ocsvm","tcn_exh_z8"),
     ("geom_nc_mean|ocsvm","tcn_geom_nc"),("coords_nc_mean|ocsvm","tcn_coords_nc"),("ang286_mean|ocsvm","tcn_ang286"),("dist78_mean|ocsvm","tcn_dist78")]
print("== paired (new implementation, 2 runs each)")
for a,b in P: paired(pd(a),pd(b),a,b)
print("== old vs new reference")
paired(pd("tcn_coords_nc"),old_pd("TCN-C"),"tcn_coords_nc","TCN-C(old)")
print("== static vs normal, same models")
for n in ["tcn_geom","tcn_geom_nc","tcn_geom_da","tcn_exh","tcn_exh_k1","tcn_geom_da_k1","tcn_exh_z8","tcn_coords_nc"]:
    paired(pd(n,"static"),pd(n),n+" static",n)
print("== all pairs of TCN-new models, all 15 variants combined: mean of (static - normal):",
      np.mean([pd(n,"static").mean()-pd(n).mean() for n in NEW if n.startswith("tcn_")]))
print("== epochs (mean), run-to-run")
for n in sorted(k for k in NEW if k.startswith("tcn_")):
    ep = np.mean([[r["epochs"][d] for d in r["epochs"]] for r in NEW[n]])
    print("  {:16s} epochs {:5.1f}  runs {}".format(n, ep, " ".join("{:.4f}".format(np.mean(list(r["auc"].values()))) for r in NEW[n])))
print("== D9 and D5 for key variants")
for n in ["tcn_geom","tcn_geom_nc","tcn_geom_da","tcn_exh","tcn_exh_k1","geom_da_mean|ocsvm","exh_mean|ocsvm"]:
    a=pd(n); print("  {:22s} D5 {:.4f} D9 {:.4f} D14 {:.4f}".format(n,a[4],a[8],a[13]))
for n in sorted(k for k in NEW if k.endswith("_mean") and k.startswith("tcn_")):
    a = pd(n)
    print("{:22s} runs {} avg {:.4f} sd {:.4f} min {:.4f} (D{}) PR {:.4f} macro {:.4f} | runs {}".format(
        n, len(NEW[n]), a.mean(), a.std(ddof=1), a.min(), a.argmin() + 1, pd(n, "pr").mean(), pd(n, "macro").mean(),
        " ".join("{:.4f}".format(np.mean(list(r["auc"].values()))) for r in NEW[n])))
    print("   per driver", " ".join("{:.4f}".format(x) for x in a))
    print("   per action", " ".join("{:.4f}".format(x) for x in per_action(n)))
    f = n[4:].replace("_k1_mean", "")
    paired(pd(f + "_mean|ocsvm"), a, f + "_mean|ocsvm", n)
    paired(a, pd("tcn_" + f), n, "tcn_" + f)
    paired(a, pd("tcn_" + f + "_k1"), n, "tcn_" + f + "_k1")
    if f + "_mean|pca" in NEW:
        paired(a, pd(f + "_mean|pca"), n, f + "_mean|pca")
print("== extra pairs")
paired(pd("geom_da_mean|ocsvm"), pd("tcn_exh_k1_mean"), "ocsvm Selected", "meanAE Exhaustive")
paired(pd("tcn_exh_k1_mean"), pd("tcn_geom_da_k1_mean"), "meanAE Exhaustive", "meanAE Selected")
paired(pd("tcn_geom_da_k1_mean"), pd("tcn_geom_k1_mean"), "meanAE Selected", "meanAE Geometry")
paired(pd("tcn_geom_da_k1_mean"), pd("tcn_exh"), "meanAE Selected", "tcn_exh")
for n in ["tcn_geom_k1_mean", "tcn_geom_da_k1_mean", "tcn_exh_k1_mean"]:
    print(n, "params", NEW[n][0]["n_params"], "epochs", np.mean([[r["epochs"][d] for d in r["epochs"]] for r in NEW[n]]))
