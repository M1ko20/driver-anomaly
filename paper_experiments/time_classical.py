"""Time of the OC-SVM method with the averaged features and of the TCN autoencoder on a CPU
(one thread), for the time table of the article. Fold 1 of the LOSO protocol (test driver
gA_1), the same training data and settings as paper_classical.py --pool mean.
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 time_classical.py
Output: paper_experiments/timing_cpu.json"""
import os
import sys
import json
import time
import platform
import importlib
import numpy as np

REPS, WARM = 300, 30
out = {"cpu": platform.processor() or open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
       "threads": 1, "ocsvm": {}, "tcn_cpu": {}}


def timeit(fn):
    for _ in range(WARM):
        fn()
    t = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(t)), float(np.std(t))


for feat in ("geom_da", "exh"):
    sys.argv = ["paper_classical.py", "--feat", feat, "--pool", "mean"]
    pc = importlib.reload(sys.modules["paper_classical"]) if "paper_classical" in sys.modules else importlib.import_module("paper_classical")
    from sklearn.svm import OneClassSVM
    meta = json.load(open(os.path.join(pc.RAW_DIR, "meta.json")))
    mbc = {m["clip"]: m for m in meta}
    raw = {m["clip"]: np.load(os.path.join(pc.RAW_DIR, m["clip"] + ".npy")) for m in meta}
    t0 = time.perf_counter()
    feats = {c: pc.build_features(b, feat) for c, b in raw.items()}
    n_frames = sum(b.shape[0] for b in raw.values())
    t_feat = (time.perf_counter() - t0) * 1000.0 / n_frames
    one = next(iter(raw.values()))[:1]
    t_feat1 = timeit(lambda: pc.build_features(one, feat))
    drivers = sorted(set(m["driver_id"] for m in meta))
    cbd = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}
    test, rem = drivers[0], drivers[1:]
    val = rem[0]
    xtr = pc.collect([c for d in rem if d != val for c in cbd[d]], mbc, feats, "safe")[0]
    xte = pc.collect(cbd[test], mbc, feats)[0]
    tr = np.concatenate(xtr)
    mean, std = tr.mean(axis=0), tr.std(axis=0)
    std[std < 1e-6] = 1e-6
    tr = ((tr - mean) / std).astype(np.float32)
    te = ((np.concatenate(xte) - mean) / std).astype(np.float32)
    sub = tr[np.random.RandomState(0).choice(len(tr), 8000, replace=False)]
    gamma = 1.0 / (sub.shape[1] * sub.var())
    t0 = time.perf_counter()
    oc = OneClassSVM(kernel="rbf", nu=0.05, gamma=gamma).fit(sub)
    t_fit = time.perf_counter() - t0
    x1 = te[:1]
    t_one = timeit(lambda: oc.decision_function(x1))
    win = np.random.RandomState(1).randn(60, sub.shape[1]).astype(np.float32)
    t_avg = timeit(lambda: ((win.mean(axis=0) - mean) / std))
    t0 = time.perf_counter()
    oc.decision_function(te)
    t_batch = time.perf_counter() - t0
    out["ocsvm"][feat] = {"dim": int(sub.shape[1]), "n_train": 8000, "n_sv": int(len(oc.support_)), "fit_s": t_fit,
                          "one_window_ms": t_one, "average_and_standardize_ms": t_avg,
                          "windows_per_s_batch": len(te) / t_batch, "features_ms_per_frame_vectorized": t_feat,
                          "features_ms_one_frame": t_feat1}
    print(feat, out["ocsvm"][feat], flush=True)

import torch
torch.set_num_threads(1)
sys.argv = ["paper_ablation.py", "--feat", "geom", "--threads", "1"]
pa = importlib.import_module("paper_ablation")
for name, D, k in (("Selected", 13, 5), ("Geometry", 52, 5), ("Exhaustive", 364, 5), ("Selected, kernel 1", 13, 1),
                   ("Exhaustive, kernel 1", 364, 1)):
    model = pa.TCNAutoencoder(D, 16, k).eval()
    x = torch.randn(1, D, 60)
    xb = torch.randn(64, D, 60)
    with torch.no_grad():
        t_one = timeit(lambda: ((model(x) - x) ** 2).mean().item())
        t0 = time.perf_counter()
        for _ in range(20):
            ((model(xb) - xb) ** 2).mean(dim=(1, 2))
        t_b = time.perf_counter() - t0
    out["tcn_cpu"][name] = {"dim": D, "kernel": k, "params": sum(p.numel() for p in model.parameters()),
                            "one_window_ms": t_one, "windows_per_s_batch64": 20 * 64 / t_b}
    print(name, out["tcn_cpu"][name], flush=True)
json.dump(out, open(os.path.join(pc.OUT_DIR, "timing_cpu.json"), "w"), indent=1)
