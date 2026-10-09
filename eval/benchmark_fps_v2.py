"""FPS benchmark v2 -- fixes a methodology bug in v1.

v1 only synced CUDA once after all iterations, which measured pipelined
throughput instead of true per-call latency (the real streaming use case
needs a score for every window right away).

Fix: inputs are pre-generated before timing, batch=1 latency syncs after
every iteration, and batch=64 throughput still syncs only at the end since
that matches real offline processing."""
import os
import sys
import time
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "training"))
sys.path.insert(0, _ROOT)

from common.tcn_ae import TCNAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WARMUP = 30
N_ITERS = 300
BATCH_OFFLINE = 64


def bench_latency_synced(model, x_fixed, n_iters=N_ITERS, n_warmup=N_WARMUP):
    """Per-call latence, batch=1, s vynucenym sync po KAZDEM volani."""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_warmup):
            out = model(x_fixed)
            _ = out.sum().item()
        for _ in range(n_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x_fixed)
            _ = out.sum().item()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    times = np.array(times)
    return times.mean(), times.std()


def bench_throughput_pipelined(model, x_fixed, n_iters=N_ITERS, n_warmup=N_WARMUP):
    """Batchova propustnost, sync az na konci (legitimni pro offline zpracovani)."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x_fixed)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(x_fixed)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    elapsed = t1 - t0
    total_windows = n_iters * x_fixed.shape[0]
    return total_windows / elapsed


results = []


def report(name, n_params, seq_len, lat_mean_s, lat_std_s, throughput_boff):
    fps_true = 1.0 / lat_mean_s
    results.append(dict(
        name=name, n_params=n_params, seq_len=seq_len,
        fps_synced_batch1=fps_true,
        latency_synced_ms=lat_mean_s * 1000,
        latency_synced_std_ms=lat_std_s * 1000,
        throughput_batch_wps=throughput_boff,
    ))
    print(f"{name:55s} | params={n_params:>9,d} | "
          f"true batch=1 latency={lat_mean_s*1000:7.4f}+-{lat_std_s*1000:.4f} ms "
          f"({fps_true:>8.1f} FPS) | batch{BATCH_OFFLINE} pipelined throughput={throughput_boff:>9.1f} windows/s")


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def run_bench(label, model, n_feat_or_shape, seq_len, make_shape_fn):
    x1 = make_shape_fn(1)
    xB = make_shape_fn(BATCH_OFFLINE)
    lat_mean, lat_std = bench_latency_synced(model, x1)
    thrB = bench_throughput_pipelined(model, xB)
    report(label, count_params(model), seq_len, lat_mean, lat_std, thrB)


print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 110)

for label, n_feat, seq_len in [
    ("TCN AE - Alpha (39d, Dense)", 39, 60),
    ("TCN AE - Beta (52d, Dense)", 52, 60),
    ("TCN AE - Gamma (104d, Dense)", 104, 60),
    ("TCN AE - Delta (56d, Dense)", 56, 60),
    ("TCN AE - Epsilon (60d, Dense)", 60, 60),
    ("TCN AE - Zeta (65d, Dense)", 65, 60),
]:
    model = TCNAutoencoder(n_feat).to(device)
    run_bench(label, model, n_feat, seq_len, lambda bs, nf=n_feat, sl=seq_len: torch.randn(bs, nf, sl, device=device))

model = TCNAutoencoder(52).to(device)
run_bench("TCN AE - Beta (52d, Long-term 30/3)", model, 52, 30, lambda bs: torch.randn(bs, 52, 30, device=device))

import train_eval_mamba_beta_v2 as mamba_simple_mod

model = mamba_simple_mod.MambaLikeAutoencoder(52).to(device)
run_bench("Mamba-like AE simplified (Beta, Long-term 30/3)", model, 52, 30, lambda bs: torch.randn(bs, 52, 30, device=device))

import train_loso_beta_mamba_real_v2 as mamba_real_mod

model = mamba_real_mod.MambaAutoencoder(52, d_state=8).to(device)
run_bench("Mamba real (selective S6, Beta, Dense 60/1)", model, 52, 60, lambda bs: torch.randn(bs, 52, 60, device=device))

import train_eval_stgcn_v2 as stgcn_mod

V = 13
A_joints = stgcn_mod.build_adjacency(V, stgcn_mod.EDGES)
model = stgcn_mod.TwoStreamSTGCNAutoencoder(V, stgcn_mod.EDGES, A_joints).to(device)
run_bench("ST-GCN two-stream (Alpha-per-node, Dense 60/1)", model, 3, 60, lambda bs: torch.randn(bs, 3, 60, V, device=device))

import train_eval_stgcn_gammapernode_v2 as stgcn_gamma_mod

A_joints_g = stgcn_gamma_mod.build_adjacency(V, stgcn_gamma_mod.EDGES)
model = stgcn_gamma_mod.TwoStreamSTGCNAutoencoder(V, stgcn_gamma_mod.EDGES, A_joints_g).to(device)
NC = stgcn_gamma_mod.NC
run_bench("ST-GCN two-stream (Gamma-per-node, Dense 60/1)", model, NC, 60, lambda bs: torch.randn(bs, NC, 60, V, device=device))

import train_loso_beta_stgcn_improved_v2 as stgcn_imp_mod

A_joints_imp = stgcn_imp_mod.build_adjacency(V, stgcn_imp_mod.PHYSICAL_EDGES)
model = stgcn_imp_mod.ImprovedTwoStreamSTGCNAutoencoder(
    V, stgcn_imp_mod.PHYSICAL_EDGES, stgcn_imp_mod.ALL_EDGES, A_joints_imp
).to(device)
Cimp = stgcn_imp_mod.C
run_bench("ST-GCN improved (virtual edges, Dense 60/1)", model, Cimp, 60, lambda bs: torch.randn(bs, Cimp, 60, V, device=device))

import train_eval_predictive_beta_v2 as pred_mod

model = pred_mod.PredictiveTCN(52, pred_mod.HORIZON).to(device)
CTX = pred_mod.CONTEXT
run_bench("Predictive TCN (Beta, context45->horizon15)", model, 52, CTX, lambda bs: torch.randn(bs, 52, CTX, device=device))

import train_loso_ensemble_v2 as ens_mod

ens_dims = [("Beta", 52), ("Epsilon", 60), ("Zeta", 65)]
ens_models = [ens_mod.TCNAutoencoder(d).to(device).eval() for _, d in ens_dims]


def bench_ensemble_latency_synced(n_iters=N_ITERS, n_warmup=N_WARMUP):
    xs1 = [torch.randn(1, d, 60, device=device) for _, d in ens_dims]
    with torch.no_grad():
        for _ in range(n_warmup):
            for m, x in zip(ens_models, xs1):
                _ = m(x).sum().item()
        times = []
        for _ in range(n_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for m, x in zip(ens_models, xs1):
                _ = m(x).sum().item()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    times = np.array(times)
    return times.mean(), times.std()


def bench_ensemble_throughput(n_iters=N_ITERS, n_warmup=N_WARMUP, bs=BATCH_OFFLINE):
    xsB = [torch.randn(bs, d, 60, device=device) for _, d in ens_dims]
    with torch.no_grad():
        for _ in range(n_warmup):
            for m, x in zip(ens_models, xsB):
                _ = m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            for m, x in zip(ens_models, xsB):
                _ = m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    elapsed = t1 - t0
    return (n_iters * bs) / elapsed


lat_mean, lat_std = bench_ensemble_latency_synced()
thrB = bench_ensemble_throughput()
n_params_ens = sum(count_params(m) for m in ens_models)
report("Ensemble (Beta+Epsilon+Zeta, 3x TCN AE sequential)", n_params_ens, 60, lat_mean, lat_std, thrB)

print("=" * 110)
print("\nSouhrn, serazeno podle skutecne (synced) batch=1 FPS:\n")
for r in sorted(results, key=lambda r: -r["fps_synced_batch1"]):
    print(f"{r['name']:58s} {r['fps_synced_batch1']:>9.1f} FPS   "
          f"(latence {r['latency_synced_ms']:.4f}+-{r['latency_synced_std_ms']:.4f} ms, {r['n_params']:>9,d} params)")

import json
from common.paths import p
results_path = p("fps_benchmark_v2_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print("\nUlozeno:", results_path)
