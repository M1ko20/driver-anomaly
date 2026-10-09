"""Simplest possible sanity check, no clever timing tricks: runs the TCN AE
(Beta, Dense) forward pass 50000x in a plain loop and times the whole thing
with shell `time`, so there's nothing to distrust in the Python timing code
itself."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch

from common.tcn_ae import TCNAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'N/A'}")

model = TCNAutoencoder(52).to(device).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

x = torch.randn(1, 52, 60, device=device)

N = 50000
print("Warmup 100 iteraci...")
with torch.no_grad():
    for _ in range(100):
        _ = model(x).sum().item()
if device.type == "cuda":
    torch.cuda.synchronize()

print(f"Spoustim {N} sekvencnich forward+sync volani, obycejnou smyckou...")
t_start_wall = time.time()
with torch.no_grad():
    for i in range(N):
        out = model(x)
        s = out.sum().item()
t_end_wall = time.time()

elapsed = t_end_wall - t_start_wall
print("\n=== VYSLEDEK ===")
print(f"Celkovy cas (Python time.time, cely for-loop): {elapsed:.3f} s")
print(f"Pocet volani: {N}")
print(f"Prumerna latence: {elapsed/N*1000:.4f} ms/volani")
print(f"FPS (1/prumerna latence): {N/elapsed:.1f}")
print(f"Posledni vystup .sum() (jen aby se overilo ze se fakt neco pocitalo): {s}")
