"""Analysis scripts behind the result tables and figures of the manuscript (2026-10-08).

Data: a local copy of the results from sumo2, pointed to by the PAPER_RESULTS variable:
  rsync -az --exclude __pycache__ --exclude slots sumo2.cs.vsb.cz:/home/fei/mik0542/paper_experiments/ $PAPER_RESULTS/pe/
  rsync -az 'sumo2.cs.vsb.cz:/home/fei/mik0542/loso_*.json' $PAPER_RESULTS/top/
Order (needs numpy, scipy, scikit-learn, matplotlib; run from this directory):
  python analyze.py pooled ens thr   # pooled AUC, ensembles (writes ens.json), thresholds
  python tables.py                   # per-driver tables, paired tests, per-action results
  python extra.py                    # merged ensemble (writes ens4.json), clips, thresholds
  python gen.py                      # LaTeX bodies of the tables (tex/*.tex) + scalars.json
  python fig.py ../figs              # fig_roc_loso.pdf, fig_drivers.pdf

Loads every LOSO result of the article into one structure.
RUNS[name] = list of runs; run = {"seed", "auc": {driver: auc}, "pr": {driver: pr} or None,
"actions": {driver: {action: (auc, n)}}, "epochs": {driver: n}, "n_params", "pooled" (if stored),
"scores": prefix of the score files or None}"""
import os
import glob
import json
import numpy as np

S = os.environ.get("PAPER_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sumo"))
PE = os.path.join(S, "pe")
TOP = os.path.join(S, "top")
DRIVERS = ["gA_1", "gA_5", "gB_10", "gB_6", "gB_7", "gB_9", "gC_13", "gC_14",
           "gE_28", "gE_29", "gF_23", "gZ_33", "gZ_36", "gZ_37"]
RUNS = {}


def add(name, run):
    assert sorted(run["auc"]) == sorted(DRIVERS), (name, sorted(run["auc"]))
    RUNS.setdefault(name, []).append(run)


def from_folds(d, scores=None):
    f = d["folds"]
    return {"seed": d.get("seed"), "auc": {k: v["auc"] for k, v in f.items()},
            "pr": {k: v["pr_auc"] for k, v in f.items()},
            "actions": {k: {a: (b["auc"], b["n_windows"]) for a, b in v["actions"].items()} for k, v in f.items()},
            "epochs": {k: v.get("epochs") for k, v in f.items()},
            "nwin": {k: v["n_test_windows"] for k, v in f.items()},
            "n_params": next(iter(f.values())).get("n_params"), "n_clips": d.get("n_clips"),
            "seconds": sum(v.get("seconds", 0) for v in f.values()), "scores": scores}


# --- queue 1 (paper_loso.py)
for exp in ["coords_tcn", "geom_tcn", "kin_tcn", "forecast_tcn", "stgcn_coords", "stgcn_kin"]:
    for s in (0, 1):
        p = os.path.join(PE, "{}_seed{}.json".format(exp, s))
        if os.path.exists(p):
            add(exp, from_folds(json.load(open(p))))

# --- queue A (paper_extra.py)
for p in sorted(glob.glob(os.path.join(PE, "x_*_seed*.json"))):
    d = json.load(open(p))
    base = os.path.basename(p)[:-5]
    add(d["exp"], from_folds(d, scores=base))

# --- queue B (paper_mamba.py), one file per fold
for feat in ["coords", "kin", "geom_common", "hands_common", "hh_common", "geom_da"]:   # geom_da = Selected
    for s in (0, 1):
        files = sorted(glob.glob(os.path.join(PE, "mamba_{}_f*_seed{}.json".format(feat, s))))
        if not files:
            continue
        d = json.load(open(files[0]))
        for p in files[1:]:
            d["folds"].update(json.load(open(p))["folds"])
        add("mamba_" + feat, from_folds(d, scores="mamba_{}_seed{}".format(feat, s)))

# --- mambalike_lt (first queue, one seed; not used in the article)
files = sorted(glob.glob(os.path.join(PE, "mambalike_lt_f*_seed0.json")))
d = json.load(open(files[0]))
for p in files[1:]:
    d["folds"].update(json.load(open(p))["folds"])
add("mambalike_lt", from_folds(d))

# --- classical baselines
BASE = {}
for feat in ["coords", "geom", "kin"]:
    for s in (0, 1, 2):
        d = json.load(open(os.path.join(PE, "baselines_{}_seed{}.json".format(feat, s))))
        for m in ["pca", "ocsvm", "iforest"]:
            f = d["folds"]
            RUNS.setdefault("{}_{}".format(m, feat), []).append(
                {"seed": s, "auc": {k: v[m]["auc"] for k, v in f.items()},
                 "pr": {k: v[m]["pr_auc"] for k, v in f.items()}, "actions": None, "epochs": None,
                 "nwin": {k: v["n_test_windows"] for k, v in f.items()}, "n_params": None, "scores": None})


# --- older runs in the home directory (August / September)
def old(d, key="fold_results"):
    fr = d[key]
    diag = d.get("per_driver_diagnostic") or {}
    acts = {k: {a: (b["auc"], b["n_windows"]) for a, b in v["action_breakdown"].items()}
            for k, v in diag.items()} if diag else None
    return {"seed": d.get("seed"), "auc": {x["driver"]: x["auc"] for x in fr}, "pr": None, "actions": acts,
            "epochs": None, "nwin": None, "n_params": d.get("n_params"), "pooled": d.get("pooled_auc"), "scores": None}


add("old_geom_tcn", old(json.load(open(os.path.join(TOP, "loso_beta_v2_results.json")))))
d = json.load(open(os.path.join(TOP, "loso_longterm_diagnostic.json")))
add("old_geom_tcn_long", {"seed": None, "auc": {k: v["fold_auc"] for k, v in d["per_driver"].items()}, "pr": None,
                          "actions": {k: {a: (b["auc"], b["n_windows"]) for a, b in v["action_breakdown"].items()}
                                      for k, v in d["per_driver"].items()},
                          "epochs": None, "nwin": {k: v["n_test_windows"] for k, v in d["per_driver"].items()},
                          "n_params": None, "pooled": d["pooled_auc"], "scores": None})
for nm, key in [("mamba_real", "old_mamba_geom"), ("stgcn_improved", "old_stgcn"), ("delta", "old_proxies_tcn"),
                ("epsilon", "old_hands_tcn"), ("zeta", "old_hh_tcn")]:
    for s in (0, 1):
        add(key, old(json.load(open(os.path.join(TOP, "loso_{}_v2_seed{}_results.json".format(nm, s))))))
for s in (0, 1):
    d = json.load(open(os.path.join(TOP, "loso_ensemble_v2_seed{}_results.json".format(s))))
    for nm, key in [("beta", "oldens_geom"), ("epsilon", "oldens_hands"), ("zeta", "oldens_hh")]:
        add(key, {"seed": s, "auc": {x["driver"]: x["auc"] for x in d["per_model_fold_results"][nm]}, "pr": None,
                  "actions": None, "epochs": None, "nwin": None, "n_params": None, "scores": None})
    diag = d.get("per_driver_diagnostic") or {}
    acts = {k: {a: (b["auc"], b["n_windows"]) for a, b in v["action_breakdown"].items()} for k, v in diag.items()} \
        if diag and "action_breakdown" in next(iter(diag.values())) else None
    add("oldens_mean", {"seed": s, "auc": {x["driver"]: x["auc"] for x in d["ensemble_mean_fold_results"]}, "pr": None,
                        "actions": acts, "epochs": None, "nwin": None, "n_params": None,
                        "pooled": d["pooled_auc_mean"], "scores": None})
    add("oldens_max", {"seed": s, "auc": {x["driver"]: x["auc"] for x in d["ensemble_max_fold_results"]}, "pr": None,
                       "actions": None, "epochs": None, "nwin": None, "n_params": None,
                       "pooled": d["pooled_auc_max"], "scores": None})


def per_driver(name, key="auc"):
    """Average over the runs, per driver (array in the order of DRIVERS)."""
    return np.mean([[r[key][d] for d in DRIVERS] for r in RUNS[name]], axis=0)


def run_means(name, key="auc"):
    return [float(np.mean([r[key][d] for d in DRIVERS])) for r in RUNS[name]]


def load_scores(prefix, driver):
    z = np.load(os.path.join(PE, "scores", "{}_{}.npz".format(prefix, driver)), allow_pickle=False)
    return z


if __name__ == "__main__":
    print("{:28s} {:>4s} {:>8s} {:>8s} {:>7s} {:>8s} {:>8s} {:>7s}  per-run means".format(
        "name", "runs", "avgAUC", "sd(drv)", "minDrv", "PR-AUC", "params", "clips"))
    for n in RUNS:
        a = per_driver(n)
        rm = run_means(n)
        pr = per_driver(n, "pr").mean() if RUNS[n][0]["pr"] else float("nan")
        r0 = RUNS[n][0]
        print("{:28s} {:4d} {:8.4f} {:8.4f} {:7.4f} {:8.4f} {:>8s} {:>7s}  {}".format(
            n, len(RUNS[n]), a.mean(), a.std(), a.min(), pr, str(r0.get("n_params")), str(r0.get("n_clips")),
            " ".join("{:.4f}".format(x) for x in rm)))
