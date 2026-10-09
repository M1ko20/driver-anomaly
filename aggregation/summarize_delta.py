import json

for seed in [0, 1]:
    d = json.load(open("/home/fei/mik0542/loso_delta_v2_seed{}_results.json".format(seed)))
    print("=== seed {} ===".format(seed))
    print("mean_auc={:.4f} std={:.4f} pooled={:.4f}".format(d["mean_auc"], d["std_auc"], d["pooled_auc"]))
    for f in d["fold_results"]:
        print("  {:8s} {:.4f}".format(f["driver"], f["auc"]))
    print("  -- action_summary --")
    for act, s in sorted(d["action_summary"].items(), key=lambda kv: kv[1]["mean_auc"]):
        print("  {:22s} mean={:.3f} weighted={:.3f} n_folds={}".format(
            act, s["mean_auc"], s["weighted_auc"], s["n_folds"]))
    print()
