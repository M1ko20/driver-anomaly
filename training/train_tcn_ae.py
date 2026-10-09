"""Step 4+5+6 (Alpha only, pipeline validation).

Dense window (SEQ_LEN=60, FRAME_STRIDE=1, WINDOW_STEP=1), TCN autoencoder
(39 features x 60 frames), trained on safe-drive only with early stopping on
val MSE."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.paths import p
from common.tcn_ae import TCNAutoencoder, WindowDataset, train_with_early_stopping

FEAT_DIR = p("alpha_features")
SEQ_LEN = 60
FRAME_STRIDE = 1
WINDOW_STEP = 1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
CKPT_PATH = p("tcn_ae_alpha_best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_windows(split):
    files = sorted(glob.glob(os.path.join(FEAT_DIR, split, "*.npy")))
    windows = []
    for path in files:
        feat = np.load(path)
        T = feat.shape[0]
        span = (SEQ_LEN - 1) * FRAME_STRIDE + 1
        if T < span:
            continue
        for start in range(0, T - span + 1, WINDOW_STEP):
            idx = np.arange(start, start + span, FRAME_STRIDE)
            windows.append(feat[idx])
    if not windows:
        return np.zeros((0, SEQ_LEN, 39), dtype=np.float32)
    return np.stack(windows).astype(np.float32)


def main():
    print("Device:", device)
    train_windows = make_windows("train")
    val_windows = make_windows("val")
    print("Train windows:", train_windows.shape, "Val windows:", val_windows.shape)

    train_loader = DataLoader(WindowDataset(train_windows), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(val_windows), batch_size=BATCH_SIZE, shuffle=False)

    model = TCNAutoencoder(train_windows.shape[2]).to(device)
    best_val, _ = train_with_early_stopping(
        model, train_loader, val_loader, device, LR, MAX_EPOCHS, PATIENCE, ckpt_path=CKPT_PATH)

    print("Best val MSE:", best_val, "-> saved to", CKPT_PATH)


if __name__ == "__main__":
    main()
