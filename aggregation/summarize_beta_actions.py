import json

d = json.load(open("/home/fei/mik0542/loso_fold6_diagnostic.json"))
print("mean_auc={:.4f} std={:.4f} pooled={:.4f}".format(d["mean_auc"], d["std_auc"], d["pooled_auc"]))

action_totals = {}
for drv, info in d["per_driver"].items():
    for act, ai in info.get("action_breakdown", {}).items():
        action_totals.setdefault(act, []).append((ai["auc"], ai["n_windows"]))

for act, vals in sorted(action_totals.items(), key=lambda kv: sum(v[0] for v in kv[1]) / len(kv[1])):
    aucs = [v[0] for v in vals]
    weights = [v[1] for v in vals]
    mean_auc = sum(aucs) / len(aucs)
    weighted_auc = sum(a * w for a, w in zip(aucs, weights)) / sum(weights)
    print("  {:22s} mean={:.3f} weighted={:.3f} n_folds={}".format(act, mean_auc, weighted_auc, len(vals)))
