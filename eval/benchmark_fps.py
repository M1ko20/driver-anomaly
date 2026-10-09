"""Benchmarks FPS/throughput of all autoencoder architectures used in the
project, on sumo2.

Each model gets random input of the right shape (architecture, not trained
weights, determines speed). Measures GPU inference time with proper warmup
and CUDA sync. Reports both streaming latency (batch=1) and offline
throughput (batch=64)."""
import os
import sys
import time
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "training"))
sys.path.insert(0, _ROOT)

from common.tcn_ae import TCNAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WARMUP = 20
N_ITERS = 200
BATCH_OFFLINE = 64


def bench_forward(model, make_input, batch_size, n_iters=N_ITERS, n_warmup=N_WARMUP):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            x = make_input(batch_size)
            _ = model(x) if not isinstance(x, tuple) else model(*x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            x = make_input(batch_size)
            _ = model(x) if not isinstance(x, tuple) else model(*x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    total_windows = n_iters * batch_size
    elapsed = t1 - t0
    return elapsed / n_iters, total_windows / elapsed


results = []


def report(name, n_params, seq_len, latency_b1_s, throughput_b1, latency_boff_s, throughput_boff):
    fps_streaming = 1.0 / latency_b1_s
    results.append(dict(
        name=name, n_params=n_params, seq_len=seq_len,
        fps_streaming=fps_streaming,
        latency_b1_ms=latency_b1_s * 1000,
        throughput_batch_wps=throughput_boff,
        latency_boff_ms=latency_boff_s * 1000,
    ))
    print(f"{name:40s} | params={n_params:>9,d} | seq_len={seq_len:>3d} | "
          f"streaming FPS={fps_streaming:>8.1f} (batch=1 latency {latency_b1_s*1000:.3f} ms) | "
          f"batch{BATCH_OFFLINE} throughput={throughput_boff:>9.1f} windows/s")


def count_params(m):
    return sum(p.numel() for p in m.parameters())


print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 100)

for label, n_feat, seq_len in [
    ("TCN AE - Alpha (39d, Dense)", 39, 60),
    ("TCN AE - Beta (52d, Dense)", 52, 60),
    ("TCN AE - Gamma (104d, Dense)", 104, 60),
    ("TCN AE - Delta (56d, Dense)", 56, 60),
    ("TCN AE - Epsilon (60d, Dense)", 60, 60),
    ("TCN AE - Zeta (65d, Dense)", 65, 60),
]:
    model = TCNAutoencoder(n_feat).to(device)

    def make_input(bs, n_feat=n_feat, seq_len=seq_len):
        return torch.randn(bs, n_feat, seq_len, device=device)

    lat1, thr1 = bench_forward(model, make_input, 1)
    latB, thrB = bench_forward(model, make_input, BATCH_OFFLINE)
    report(label, count_params(model), seq_len, lat1, thr1, latB, thrB)

model = TCNAutoencoder(52).to(device)


def make_input_lt(bs):
    return torch.randn(bs, 52, 30, device=device)


lat1, thr1 = bench_forward(model, make_input_lt, 1)
latB, thrB = bench_forward(model, make_input_lt, BATCH_OFFLINE)
report("TCN AE - Beta (52d, Long-term 30/3)", count_params(model), 30, lat1, thr1, latB, thrB)

import train_eval_mamba_beta_v2 as mamba_simple_mod

model = mamba_simple_mod.MambaLikeAutoencoder(52).to(device)


def make_input_mamba_simple(bs):
    return torch.randn(bs, 52, 30, device=device)


lat1, thr1 = bench_forward(model, make_input_mamba_simple, 1)
latB, thrB = bench_forward(model, make_input_mamba_simple, BATCH_OFFLINE)
report("Mamba-like AE simplified (Beta, Long-term 30/3)", count_params(model), 30, lat1, thr1, latB, thrB)

import train_loso_beta_mamba_real_v2 as mamba_real_mod

model = mamba_real_mod.MambaAutoencoder(52, d_state=8).to(device)


def make_input_mamba_real(bs):
    return torch.randn(bs, 52, 60, device=device)


lat1, thr1 = bench_forward(model, make_input_mamba_real, 1)
latB, thrB = bench_forward(model, make_input_mamba_real, BATCH_OFFLINE)
report("Mamba real (selective S6, Beta, Dense 60/1)", count_params(model), 60, lat1, thr1, latB, thrB)

import train_eval_stgcn_v2 as stgcn_mod

V = 13
A_joints = stgcn_mod.build_adjacency(V, stgcn_mod.EDGES)
model = stgcn_mod.TwoStreamSTGCNAutoencoder(V, stgcn_mod.EDGES, A_joints).to(device)
A_joints_dev = A_joints.to(device) if torch.is_tensor(A_joints) else torch.tensor(A_joints, dtype=torch.float32, device=device)


def make_input_stgcn(bs, C=3):
    return torch.randn(bs, C, 60, V, device=device)


lat1, thr1 = bench_forward(model, make_input_stgcn, 1)
latB, thrB = bench_forward(model, make_input_stgcn, BATCH_OFFLINE)
report("ST-GCN two-stream (Alpha-per-node, Dense 60/1)", count_params(model), 60, lat1, thr1, latB, thrB)

import train_eval_stgcn_gammapernode_v2 as stgcn_gamma_mod

A_joints_g = stgcn_gamma_mod.build_adjacency(V, stgcn_gamma_mod.EDGES)
model = stgcn_gamma_mod.TwoStreamSTGCNAutoencoder(V, stgcn_gamma_mod.EDGES, A_joints_g).to(device)


def make_input_stgcn_gamma(bs, C=stgcn_gamma_mod.NC):
    return torch.randn(bs, C, 60, V, device=device)


lat1, thr1 = bench_forward(model, make_input_stgcn_gamma, 1)
latB, thrB = bench_forward(model, make_input_stgcn_gamma, BATCH_OFFLINE)
report("ST-GCN two-stream (Gamma-per-node, Dense 60/1)", count_params(model), 60, lat1, thr1, latB, thrB)

import train_loso_beta_stgcn_improved_v2 as stgcn_imp_mod

A_joints_imp = stgcn_imp_mod.build_adjacency(V, stgcn_imp_mod.PHYSICAL_EDGES)
model = stgcn_imp_mod.ImprovedTwoStreamSTGCNAutoencoder(
    V, stgcn_imp_mod.PHYSICAL_EDGES, stgcn_imp_mod.ALL_EDGES, A_joints_imp
).to(device)


def make_input_stgcn_imp(bs, C=stgcn_imp_mod.C):
    return torch.randn(bs, C, 60, V, device=device)


lat1, thr1 = bench_forward(model, make_input_stgcn_imp, 1)
latB, thrB = bench_forward(model, make_input_stgcn_imp, BATCH_OFFLINE)
report("ST-GCN improved (virtual edges, Alpha-per-node, Dense 60/1)", count_params(model), 60, lat1, thr1, latB, thrB)

import train_eval_predictive_beta_v2 as pred_mod

model = pred_mod.PredictiveTCN(52, pred_mod.HORIZON).to(device)


def make_input_pred(bs):
    return torch.randn(bs, 52, pred_mod.CONTEXT, device=device)


lat1, thr1 = bench_forward(model, make_input_pred, 1)
latB, thrB = bench_forward(model, make_input_pred, BATCH_OFFLINE)
report("Predictive TCN (Beta, context45->horizon15)", count_params(model), 45, lat1, thr1, latB, thrB)

import train_loso_ensemble_v2 as ens_mod

ens_dims = [("Beta", 52), ("Epsilon", 60), ("Zeta", 65)]
ens_models = [ens_mod.TCNAutoencoder(d).to(device) for _, d in ens_dims]
for m in ens_models:
    m.eval()


def bench_ensemble(batch_size, n_iters=N_ITERS, n_warmup=N_WARMUP):
    with torch.no_grad():
        for _ in range(n_warmup):
            for m, (_, d) in zip(ens_models, ens_dims):
                x = torch.randn(batch_size, d, 60, device=device)
                _ = m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            for m, (_, d) in zip(ens_models, ens_dims):
                x = torch.randn(batch_size, d, 60, device=device)
                _ = m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    elapsed = t1 - t0
    total_windows = n_iters * batch_size
    return elapsed / n_iters, total_windows / elapsed


lat1, thr1 = bench_ensemble(1)
latB, thrB = bench_ensemble(BATCH_OFFLINE)
n_params_ens = sum(count_params(m) for m in ens_models)
report("Ensemble (Beta+Epsilon+Zeta, 3x TCN AE sequential)", n_params_ens, 60, lat1, thr1, latB, thrB)

print("=" * 100)
print("\nSouhrn, serazeno podle streaming FPS (rychlejsi = lepsi):\n")
for r in sorted(results, key=lambda r: -r["fps_streaming"]):
    print(f"{r['name']:52s} {r['fps_streaming']:>9.1f} FPS   ({r['n_params']:>9,d} params)")

import json
from common.paths import p
results_path = p("fps_benchmark_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print("\nUlozeno:", results_path)
