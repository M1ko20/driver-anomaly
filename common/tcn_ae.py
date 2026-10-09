"""The TCN autoencoder architecture and its train/score helpers, shared by
every train_eval_tcn_ae_*.py, train_loso_{beta,beta_longterm,beta_diag,
delta,epsilon,zeta,ensemble}_v2.py, train_tcn_ae.py, and sanity_check.py.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return torch.from_numpy(self.windows[i].T)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        pad = dilation * 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.shape[2]] if out.shape[2] > x.shape[2] else out
        return self.act(self.bn(out))


class TCNAutoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            TCNBlock(n_features, 64, dilation=1),
            TCNBlock(64, 32, dilation=2),
            TCNBlock(32, 16, dilation=4),
        )
        self.decoder = nn.Sequential(
            TCNBlock(16, 32, dilation=4),
            TCNBlock(32, 64, dilation=2),
            nn.Conv1d(64, n_features, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_n = 0.0, 0
    criterion = nn.MSELoss()
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.shape[0]
            total_n += batch.shape[0]
    return total_loss / max(total_n, 1)


def per_window_mse(model, windows, device, batch_size=256):
    model.eval()
    mses = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i:i + batch_size].transpose(0, 2, 1)).to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            mses.append(mse.cpu().numpy())
    return np.concatenate(mses) if mses else np.array([])


def train_with_early_stopping(model, train_loader, val_loader, device, lr, max_epochs,
                               patience, ckpt_path=None, log_every=1, log_prefix="",
                               val_label="val"):
    """Trains `model` with Adam + MSE, early-stopping on val loss. Returns
    (best_val_loss, best_state_dict). If ckpt_path is given, also saves the
    best state_dict there as training progresses (matching the train_eval_*
    scripts); if not, the best state is only kept in memory (matching the
    LOSO scripts, which never touch disk mid-fold)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None)
        if log_every and (epoch == 1 or epoch % log_every == 0):
            print("{}Epoch {:3d} | train MSE {:.6f} | {} MSE {:.6f}".format(
                log_prefix, epoch, train_loss, val_label, val_loss))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("{}Early stopping at epoch {} (best {} MSE {:.6f})".format(
                    log_prefix, epoch, val_label, best_val))
                break

    return best_val, best_state
