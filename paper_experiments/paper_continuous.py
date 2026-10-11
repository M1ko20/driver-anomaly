"""Scores of the models on the uninterrupted recordings (whole sessions) of the test driver,
LOSO protocol with the same folds and the same training data as paper_classical.py /
paper_ablation.py (the pre-cut normal clips of the 12 training drivers, beta_raw_v2). The
models are those of the article; they are only applied to all sliding windows (60 frames,
step 1) of the whole body-camera videos of the test driver (sessions_v2, see
pipeline/run_pose_sessions.py), including the transitions between the activities, the
intervals that are shorter than the window and the intervals without an annotation.

Labels of the frames: the annotation of DMD is in the timeline of the mosaic video; the
frame t of the body video corresponds to the annotated frame t + frame_shift.

Methods
  ocsvm        OC-SVM on the average of the feature vectors of the window (--feat)
  ocsvm_frame  OC-SVM on the single frames (used only to check the alignment of the labels)
  tcn          the TCN autoencoder saved by paper_ablation.py --save_models (same --feat, seed)

Output: paper_experiments/cont/<feat>_seed<k>_<driver>.npz with, for every session i of the
test driver, score arrays s<i>_<method> (index = frame of the body video; nan where no window
ends), lab<i> (-1 no annotation, 0 safe driving, 1..10 type of anomaly, 11 other annotated
activity), and the statistics of the scores of the validation driver (clip protocol)."""
import os
import sys
import json
import argparse
import importlib
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--feat", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--methods", default="ocsvm,ocsvm_frame")
ap.add_argument("--only_fold", type=int, default=-1)
ap.add_argument("--threads", type=int, default=2)
cargs = ap.parse_args()
sys.argv = ["paper_ablation.py", "--feat", cargs.feat if cargs.feat != "coords" else "geom", "--threads", str(cargs.threads)]
pa = importlib.import_module("paper_ablation")          # build_features, TCNAutoencoder
import torch  # noqa: E402

HOME = "/home/fei/mik0542"
OUT = os.path.join(HOME, "paper_experiments", "cont")
SESS = os.path.join(HOME, "sessions_v2")
RAW = os.path.join(HOME, "beta_raw_v2")
ACTS = ["reach_backseat", "reach_side", "hair_and_makeup", "phonecall_left", "phonecall_right", "drinking",
        "texting_right", "talking_to_passenger", "radio", "texting_left"]
T = 60
METHODS = cargs.methods.split(",")


def win_mean(x):
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x.astype(np.float64), axis=0)])
    return (cs[T:] - cs[:-T]) / T


def session_labels(video, n_frames):
    o = json.load(open(video.replace("_rgb_body.mp4", "_rgb_ann_distraction.json")))["openlabel"]
    shift = int(o["streams"]["body_camera"]["stream_properties"]["sync"]["frame_shift"])
    n_ann = o["frame_intervals"][0]["frame_end"] + 1
    lab = np.full(max(n_ann, n_frames + shift) + 1, -1, np.int8)
    for a in o["actions"].values():
        if not a["type"].startswith("driver_actions/"):
            continue
        act = a["type"].split("/", 1)[1]
        code = 0 if act == "safe_drive" else (ACTS.index(act) + 1 if act in ACTS else 11)
        for iv in a["frame_intervals"]:
            lab[iv["frame_start"]:iv["frame_end"] + 1] = code
    return lab[shift:shift + n_frames].copy(), shift


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(cargs.threads)
    meta = json.load(open(os.path.join(RAW, "meta.json")))
    feats = {m["clip"]: pa.build_features(np.load(os.path.join(RAW, m["clip"] + ".npy")), cargs.feat) for m in meta}
    mbc = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    cbd = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}
    manifest = json.load(open(os.path.join(HOME, "clips_v2_manifest.json")))
    vids = {d: sorted(set(m["source_video"] for m in manifest if m["driver_id"] == d)) for d in drivers}

    def clip_sets(clips, label=None):
        return [feats[c] for c in clips if (label is None or mbc[c]["label"] == label) and feats[c].shape[0] >= T]

    for fold_idx, test in enumerate(drivers):
        if cargs.only_fold >= 0 and fold_idx != cargs.only_fold:
            continue
        out_path = os.path.join(OUT, "{}_seed{}_{}.npz".format(cargs.feat, cargs.seed, test))
        if os.path.exists(out_path):
            continue
        rem = [d for d in drivers if d != test]
        val = rem[fold_idx % len(rem)]
        tr_clips = clip_sets([c for d in rem if d != val for c in cbd[d]], "safe")
        va_clips = clip_sets(cbd[val], "safe")
        te_clips = [c for c in cbd[test] if feats[c].shape[0] >= T]
        sessions = []
        for v in vids[test]:
            B = np.load(os.path.join(SESS, os.path.splitext(os.path.basename(v))[0] + ".npz"))["beta"]
            lab, shift = session_labels(v, len(B))
            sessions.append((os.path.basename(v), pa.build_features(B, cargs.feat), lab, shift))
        save = {"acts": np.array(ACTS), "val_driver": val, "n_sessions": len(sessions),
                "session_names": np.array([s[0] for s in sessions]), "shifts": np.array([s[3] for s in sessions])}
        for i, s in enumerate(sessions):
            save["lab{}".format(i)] = s[2]
        rng = np.random.RandomState(cargs.seed * 1000 + fold_idx)

        def run(name, model_score, fit_stats):
            """model_score(list of (n, D) per-frame matrices) -> list of score arrays aligned to the end frame"""
            sv = np.concatenate(model_score(va_clips, clip=True))
            st = model_score([feats[c] for c in te_clips], clip=True)
            y = np.concatenate([[1 if mbc[c]["label"] == "anomaly" else 0] * len(x) for c, x in zip(te_clips, st)])
            save["{}_clip_auc".format(name)] = roc_auc_score(y, np.concatenate(st))
            save["{}_val_mean".format(name)], save["{}_val_std".format(name)] = sv.mean(), sv.std()
            save["{}_val_p95".format(name)], save["{}_val_p99".format(name)] = np.percentile(sv, 95), np.percentile(sv, 99)
            for i, s in enumerate(sessions):
                save["s{}_{}".format(i, name)] = model_score([s[1]], clip=False)[0].astype(np.float32)
            print("fold {} {} {} clip-protocol AUC {:.4f}".format(fold_idx + 1, test, name, save[name + "_clip_auc"]), flush=True)

        if "ocsvm" in METHODS:
            tr = np.concatenate([win_mean(x) for x in tr_clips])
            mean, std = tr.mean(axis=0), tr.std(axis=0)
            std[std < 1e-6] = 1e-6
            tr = ((tr - mean) / std).astype(np.float32)
            sub = tr[rng.choice(len(tr), min(8000, len(tr)), replace=False)]
            oc = OneClassSVM(kernel="rbf", nu=0.05, gamma=1.0 / (sub.shape[1] * sub.var())).fit(sub)

            def sc(mats, clip):
                out = []
                for x in mats:
                    s = -oc.decision_function(((win_mean(x) - mean) / std).astype(np.float32))
                    out.append(s if clip else np.concatenate([np.full(T - 1, np.nan), s]))
                return out
            run("ocsvm", sc, None)
        if "ocsvm_frame" in METHODS:
            rng2 = np.random.RandomState(cargs.seed * 1000 + fold_idx)
            tr = np.concatenate(tr_clips)
            mean2, std2 = tr.mean(axis=0), tr.std(axis=0)
            std2[std2 < 1e-6] = 1e-6
            tr = ((tr - mean2) / std2).astype(np.float32)
            sub = tr[rng2.choice(len(tr), 8000, replace=False)]
            ocf = OneClassSVM(kernel="rbf", nu=0.05, gamma=1.0 / (sub.shape[1] * sub.var())).fit(sub)

            def scf(mats, clip):
                return [-ocf.decision_function(((x - mean2) / std2).astype(np.float32)) for x in mats]

            # frame-level scores: only for the sessions (the clip statistics are not needed)
            for i, s in enumerate(sessions):
                save["s{}_ocsvm_frame".format(i)] = scf([s[1]], False)[0].astype(np.float32)
        if "tcn" in METHODS:
            ck = torch.load(os.path.join(HOME, "paper_experiments", "models", "a_tcn_{}_seed{}_{}.pt".format(cargs.feat, cargs.seed, test)),
                            map_location="cpu")
            model = pa.TCNAutoencoder(ck["n_features"], ck["latent"], ck["kernel"])
            model.load_state_dict(ck["state"])
            model.eval()
            m3, s3 = np.asarray(ck["mean"]), np.asarray(ck["std"])
            ar = torch.arange(T)

            def sct(mats, clip):
                out = []
                with torch.no_grad():
                    for x in mats:
                        X = torch.from_numpy(((x - m3) / s3).astype(np.float32))
                        n = X.shape[0] - T + 1
                        sc_ = []
                        for j in range(0, n, 512):
                            idx = torch.arange(j, min(j + 512, n))[:, None] + ar[None, :]
                            b = X[idx].transpose(1, 2)
                            sc_.append(((model(b) - b) ** 2).mean(dim=(1, 2)).numpy())
                        s = np.concatenate(sc_)
                        out.append(s if clip else np.concatenate([np.full(T - 1, np.nan), s]))
                return out
            run("tcn", sct, None)
        np.savez_compressed(out_path, **save)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
