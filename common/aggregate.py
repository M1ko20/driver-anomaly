"""Shared seed/model result aggregation, used by the aggregate_seeds_*.py and
aggregate_baselines_*.py scripts (each just picks a results-file pattern,
seed list, and which bits of output it wants)."""
import json
import numpy as np

DEFAULT_METRICS = [
    "val_roc_auc", "val_pr_auc", "test_roc_auc", "test_pr_auc",
    "test_precision", "test_recall", "test_f1",
]


def aggregate_seeds(result_path_pattern, seeds, out_path, metrics=DEFAULT_METRICS,
                     show_table=True, show_saved_message=True):
    """Loads one results JSON per seed (each must contain "seed" plus every
    metric), prints a per-seed table and mean/std summary, and saves the
    summary to out_path."""
    rows = []
    for s in seeds:
        with open(result_path_pattern.format(s)) as f:
            rows.append(json.load(f))

    if show_table:
        header = "{:<6}".format("seed") + "".join("{:>16}".format(m) for m in metrics)
        print(header)
        for r in rows:
            line = "{:<6}".format(r["seed"]) + "".join("{:>16.4f}".format(r[m]) for m in metrics)
            print(line)
        print("---")

    summary = {}
    for m in metrics:
        vals = np.array([r[m] for r in rows])
        summary[m] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print("{:<16} mean={:.4f} std={:.4f}".format(m, vals.mean(), vals.std()))

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    if show_saved_message:
        print("\nSaved summary.")
    return summary


def aggregate_baselines(result_path_pattern, seeds, models, out_path, metrics=DEFAULT_METRICS):
    """Same idea as aggregate_seeds, but each results JSON holds one dict per
    baseline model (PCA/OCSVM/IsolationForest/...) instead of a flat set of
    metrics."""
    rows = {m: [] for m in models}
    for s in seeds:
        with open(result_path_pattern.format(s)) as f:
            d = json.load(f)
        for m in models:
            rows[m].append(d[m])

    summary = {}
    for m in models:
        summary[m] = {}
        print("=== {} ===".format(m))
        for metric in metrics:
            vals = np.array([r[metric] for r in rows[m]])
            summary[m][metric] = {"mean": float(vals.mean()), "std": float(vals.std())}
            print("  {:<16} mean={:.4f} std={:.4f}".format(metric, vals.mean(), vals.std()))

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary.")
    return summary
