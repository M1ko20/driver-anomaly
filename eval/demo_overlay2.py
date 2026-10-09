"""Demo v2: same as demo_overlay.py but also draws the detected skeleton on top
of the video, so you can see exactly what the model is looking at frame by
frame -- including bad detections or missed anomalies.

Usage: python3 demo_overlay2.py <clip_name> <fold_ckpt.pt> <out_mp4>"""
import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from sklearn.metrics import roc_curve

RAW_DIR = "/home/fei/mik0542/beta_raw_v2"
VIDEO_DIR = "/home/fei/mik0542/clips_v2"
POSE_DIR = "/home/fei/mik0542/pose_json_v2"
MANIFEST_PATH = "/home/fei/mik0542/clips_v2_manifest.json"
N_FEATURES = 52

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SKELETON_EDGES = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
MIN_VIS_DRAW = 0.15


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


def load_fold_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TCNAutoencoder(ckpt["n_features"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], ckpt["seq_len"], ckpt["frame_stride"], ckpt["test_driver"]


def per_frame_scores(model, feat_raw, mean, std, seq_len, frame_stride):
    T = feat_raw.shape[0]
    span = (seq_len - 1) * frame_stride + 1
    feat = (feat_raw - mean) / std
    scores = np.full(T, np.nan, dtype=np.float32)
    windows, end_idxs = [], []
    for start in range(0, T - span + 1):
        idx = np.arange(start, start + span, frame_stride)
        windows.append(feat[idx])
        end_idxs.append(start + span - 1)
    if not windows:
        return scores
    windows = np.stack(windows).astype(np.float32)
    with torch.no_grad():
        batch = torch.from_numpy(windows.transpose(0, 2, 1)).to(device)
        recon = model(batch)
        mse = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
    for e, m in zip(end_idxs, mse):
        scores[e] = m
    first_valid = end_idxs[0]
    scores[:first_valid] = scores[first_valid]
    return scores


def fold_threshold_and_range(model, mean, std, seq_len, frame_stride, test_driver):
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    clip_names = [e["clip"] for e in manifest if e["driver_id"] == test_driver]
    labels_by_clip = {e["clip"]: e["label"] for e in manifest}

    all_scores, all_labels = [], []
    for cn in clip_names:
        npy_path = os.path.join(RAW_DIR, cn + ".npy")
        if not os.path.exists(npy_path):
            continue
        feat_raw = np.load(npy_path)
        span = (seq_len - 1) * frame_stride + 1
        if feat_raw.shape[0] < span:
            continue
        s = per_frame_scores(model, feat_raw, mean, std, seq_len, frame_stride)
        valid = ~np.isnan(s)
        lbl = 1 if labels_by_clip[cn] == "anomaly" else 0
        all_scores.extend(s[valid].tolist())
        all_labels.extend([lbl] * valid.sum())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    fpr, tpr, thr = roc_curve(all_labels, all_scores)
    j = tpr - fpr
    best_thr = thr[np.argmax(j)]

    safe_scores = all_scores[all_labels == 0]
    anomaly_scores = all_scores[all_labels == 1]
    lo = float(np.percentile(safe_scores, 90)) if len(safe_scores) else float(all_scores.min())
    hi = float(np.percentile(anomaly_scores, 90)) if len(anomaly_scores) else float(all_scores.max())
    if hi <= lo:
        hi = lo + 1e-6
    return float(best_thr), lo, hi


def load_raw_keypoints_per_frame(clip_name):
    """Pixel-space keypoints per frame (best-bbox_conf person), for drawing."""
    with open(os.path.join(POSE_DIR, clip_name + ".json")) as f:
        data = json.load(f)
    per_frame = {}
    for fr in data["frames"]:
        persons = fr["persons"]
        if not persons:
            continue
        person = max(persons, key=lambda p: p["bbox_conf"] if p["bbox_conf"] is not None else -1.0)
        per_frame[fr["frame_id"]] = {k["name"]: k for k in person["keypoints"]}
    return per_frame


def draw_skeleton(canvas, kp_by_name, y_offset):
    if kp_by_name is None:
        return
    for a, b in SKELETON_EDGES:
        ka, kb = kp_by_name.get(a), kp_by_name.get(b)
        if ka and kb and ka["visibility"] >= MIN_VIS_DRAW and kb["visibility"] >= MIN_VIS_DRAW:
            pa = (int(ka["x"]), int(ka["y"]) + y_offset)
            pb = (int(kb["x"]), int(kb["y"]) + y_offset)
            cv2.line(canvas, pa, pb, (0, 220, 255), 2, cv2.LINE_AA)
    for name, k in kp_by_name.items():
        if k["visibility"] >= MIN_VIS_DRAW:
            p = (int(k["x"]), int(k["y"]) + y_offset)
            cv2.circle(canvas, p, 4, (255, 60, 60), -1, cv2.LINE_AA)


def main():
    clip_name, ckpt_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    model, mean, std, seq_len, frame_stride, test_driver = load_fold_model(ckpt_path)
    print("Loaded fold model for held-out driver: {}".format(test_driver))

    thr, lo, hi = fold_threshold_and_range(model, mean, std, seq_len, frame_stride, test_driver)
    print("Fold threshold (Youden J) = {:.4f} | display range (safe p90 -> anomaly p90) = [{:.4f}, {:.4f}]".format(thr, lo, hi))

    feat_raw = np.load(os.path.join(RAW_DIR, clip_name + ".npy"))
    scores = per_frame_scores(model, feat_raw, mean, std, seq_len, frame_stride)
    kp_per_frame = load_raw_keypoints_per_frame(clip_name)

    video_path = os.path.join(VIDEO_DIR, clip_name + ".mp4")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    banner_h = 70
    tmp_path = out_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h + banner_h))
    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i < len(scores) and not np.isnan(scores[frame_i]):
            raw = scores[frame_i]
            pct = float(np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 100.0)
            is_anomaly = raw >= thr
        else:
            pct = 0.0
            is_anomaly = False

        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[banner_h:, :, :] = frame
        draw_skeleton(canvas, kp_per_frame.get(frame_i), banner_h)

        color = (0, 0, 220) if is_anomaly else (0, 170, 0)
        label = "NOT OK  ({:.0f}% anomaly)".format(pct) if is_anomaly else "OK  ({:.0f}% anomaly)".format(pct)
        cv2.rectangle(canvas, (0, 0), (w, banner_h), color, thickness=-1)
        cv2.putText(canvas, label, (15, int(banner_h * 0.68)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        bar_w = int((w - 30) * pct / 100.0)
        cv2.rectangle(canvas, (15, banner_h - 12), (w - 15, banner_h - 6), (255, 255, 255), 1)
        cv2.rectangle(canvas, (15, banner_h - 12), (15 + bar_w, banner_h - 6), (255, 255, 255), -1)

        writer.write(canvas)
        frame_i += 1

    cap.release()
    writer.release()

    os.system("ffmpeg -y -i {} -vcodec libx264 -pix_fmt yuv420p -loglevel error {}".format(tmp_path, out_path))
    os.remove(tmp_path)
    print("Saved demo video: {} ({} frames, held-out driver {})".format(out_path, frame_i, test_driver))


if __name__ == "__main__":
    main()
