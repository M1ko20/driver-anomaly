"""Geometry features of the clips taken from the session-level pose (run_pose_sessions.py)
instead of from the pre-cut clips.

Reason: the annotation of DMD is in the timeline of the mosaic video (= the face camera for
all sessions used here); the body camera starts `frame_shift` frames later
(openlabel.streams.body_camera.stream_properties.sync.frame_shift, 1 to 41 frames), so the
annotated frame f is the frame f - frame_shift of the body video (see checkFrameInStream of
the DMD tool accessDMDAnn.py). extract_clips_v2.py cut the clips at the annotated frame
numbers without this correction, i.e. every clip of beta_raw_v2 starts and ends frame_shift
frames too late.

Two directories are written, with the same clip names and the same meta.json as beta_raw_v2:
  beta_raw_v3    frames [start - shift, end - shift] of the session (corrected alignment)
  beta_raw_v2s   frames [start, end] of the session (the old alignment, but the pose from the
                 original video instead of the re-encoded clip; control)
"""
import os
import json
import numpy as np

HOME = "/home/fei/mik0542"
SESS = os.path.join(HOME, "sessions_v2")


def main():
    manifest = {m["clip"]: m for m in json.load(open(os.path.join(HOME, "clips_v2_manifest.json")))}
    meta = json.load(open(os.path.join(HOME, "beta_raw_v2", "meta.json")))
    shifts, sess = {}, {}
    out = {"beta_raw_v3": [], "beta_raw_v2s": []}
    for d in out:
        os.makedirs(os.path.join(HOME, d), exist_ok=True)
    stats = []
    for m in meta:
        e = manifest[m["clip"]]
        v = e["source_video"]
        if v not in shifts:
            ann = json.load(open(v.replace("_rgb_body.mp4", "_rgb_ann_distraction.json")))["openlabel"]
            shifts[v] = int(ann["streams"]["body_camera"]["stream_properties"]["sync"]["frame_shift"])
            sess[v] = np.load(os.path.join(SESS, os.path.splitext(os.path.basename(v))[0] + ".npz"))["beta"]
        s, B = shifts[v], sess[v]
        old = np.load(os.path.join(HOME, "beta_raw_v2", m["clip"] + ".npy"))
        for d, sh in (("beta_raw_v3", s), ("beta_raw_v2s", 0)):
            a, b = max(e["start_frame"] - sh, 0), min(e["end_frame"] - sh, len(B) - 1)
            f = B[a:b + 1]
            np.save(os.path.join(HOME, d, m["clip"] + ".npy"), f)
            out[d].append(dict(m, shift=s, n_frames=int(len(f))))
        stats.append((s, len(old), out["beta_raw_v3"][-1]["n_frames"], out["beta_raw_v2s"][-1]["n_frames"]))
    for d in out:
        json.dump(out[d], open(os.path.join(HOME, d, "meta.json"), "w"), indent=1)
    st = np.array(stats)
    print("clips", len(st), "mean shift {:.1f} median {:.0f} max {}".format(st[:, 0].mean(), np.median(st[:, 0]), st[:, 0].max()))
    print("frames: old clips {}, corrected {}, uncorrected from sessions {}".format(st[:, 1].sum(), st[:, 2].sum(), st[:, 3].sum()))
    print("clips with at least 60 frames: old {}, corrected {}, uncorrected from sessions {}".format(
        (st[:, 1] >= 60).sum(), (st[:, 2] >= 60).sum(), (st[:, 3] >= 60).sum()))
    print("share of the frames of the old clips that lie outside the annotated interval: {:.4f}".format(
        np.minimum(st[:, 0], st[:, 1]).sum() / st[:, 1].sum()))


if __name__ == "__main__":
    main()
