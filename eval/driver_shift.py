"""How much does the normal driving of every driver differ from the normal driving of the
other 13 drivers, separately for the groups of features (coordinates, confidences, the 13
selected distances and angles)? For every driver and feature, the mean over the frames of the
normal clips is compared with the frames of the normal clips of the other drivers:
  z = |mean_driver - mean_others| / std_others  (std over the frames of the other drivers).
The table shows the average z over the features of the group. Used for the paragraph about
Driver-9 in the article.
  python3 driver_shift.py"""
import os
import json
import numpy as np

RAW = "/home/fei/mik0542/beta_raw_v2"
V = 13
NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr", "l_hip", "r_hip"]
meta = json.load(open(os.path.join(RAW, "meta.json")))
drivers = sorted(set(m["driver_id"] for m in meta))
X = {d: np.concatenate([np.load(os.path.join(RAW, m["clip"] + ".npy")) for m in meta
                        if m["driver_id"] == d and m["label"] == "safe"]).astype(np.float64) for d in drivers}
kp = lambda a: a[:, :39].reshape(-1, V, 3)
groups = {"coordinates": lambda a: kp(a)[:, :, :2].reshape(len(a), -1), "confidences": lambda a: kp(a)[:, :, 2],
          "selected": lambda a: a[:, 39:]}
print("{:10s} {:>8s} {:>12s} {:>12s} {:>10s} | mean conf  frames".format("driver", "DMD", *groups))
Z = {}
for i, d in enumerate(drivers):
    others = np.concatenate([X[o] for o in drivers if o != d])
    row = []
    for g, f in groups.items():
        a, b = f(X[d]), f(others)
        z = np.abs(a.mean(axis=0) - b.mean(axis=0)) / np.maximum(b.std(axis=0), 1e-6)
        Z[(d, g)] = z
        row.append(z.mean())
    print("Driver-{:<3d} {:>8s} {:12.2f} {:12.2f} {:10.2f} | {:.3f}  {}".format(i + 1, d, *row, kp(X[d])[:, :, 2].mean(), len(X[d])))
d9 = drivers[8]
others = np.concatenate([X[o] for o in drivers if o != d9])
print("\nDriver-9: keypoint, mean x, y, conf (driver / others), z of x, y, conf")
a, b = kp(X[d9]), kp(others)
for j, n in enumerate(NAMES):
    print("  {:6s} x {:6.2f}/{:6.2f}  y {:6.2f}/{:6.2f}  conf {:.2f}/{:.2f}   z {:.2f} {:.2f} {:.2f}".format(
        n, a[:, j, 0].mean(), b[:, j, 0].mean(), a[:, j, 1].mean(), b[:, j, 1].mean(), a[:, j, 2].mean(), b[:, j, 2].mean(),
        Z[(d9, "coordinates")][2 * j], Z[(d9, "coordinates")][2 * j + 1], Z[(d9, "confidences")][j]))
print("  selected features z:", np.round(Z[(d9, "selected")], 2))
