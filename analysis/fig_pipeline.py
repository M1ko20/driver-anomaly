"""Fig. 1 of the article (overview of the method): (a) one autoencoder, (b) window-level
one-class method, (c) combination of autoencoders at the level of anomaly scores.
  python fig_pipeline.py [output directory, default ../figs]"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
EDGE, BLUE, ORANGE = "#1f3552", "#e6edf6", "#fdeede"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(figsize=(9.1, 5.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 55)
ax.axis("off")


def box(x, y, w, h, text, color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=color, ec=EDGE, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.25)
    return x, y, w, h


def arrow(a, b):
    """From the right edge of box a to the left edge of box b."""
    ax.annotate("", xy=(b[0], b[1] + b[3] / 2), xytext=(a[0] + a[2], a[1] + a[3] / 2),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0, shrinkA=0, shrinkB=0))


def title(y, text):
    ax.text(0.5, y, text, ha="left", va="bottom", fontweight="bold", fontsize=9)


def chain(y, h, items, x0=0.5, gap=2.4, total=99.0):
    w = (total - gap * (len(items) - 1)) / len(items)
    prev = None
    for i, it in enumerate(items):
        text, color = it if isinstance(it, tuple) else (it, BLUE)
        b = box(x0 + i * (w + gap), y, w, h, text, color)
        if prev:
            arrow(prev, b)
        prev = b


# (a)
title(52.2, "(a) One autoencoder")
chain(44, 7, ["RGB video\n(body camera)", "YOLO26-pose\n(17 keypoints)", "Normalization\n(13 keypoints)",
              "Features\n(e.g. 52 values\nper frame)", "Window\n(60 frames)", "Autoencoder\n(e.g. TCN)",
              ("Reconstruction\nerror\n(anomaly score)", ORANGE)])
# (b)
title(38.7, "(b) Window-level one-class method")
chain(30.5, 7, ["RGB video\n(body camera)", "YOLO26-pose\n(17 keypoints)", "Normalization\n(13 keypoints)",
                "Features (e.g.\n13 distances\nand angles)", "Average over\nthe window\n(60 frames)",
                "One-class\nmethod\n(e.g. OC-SVM)", ("Distance from\nthe boundary\n(anomaly score)", ORANGE)])
# (c)
title(25.2, "(c) Combination of autoencoders at the level of anomaly scores")
rows = [("YOLO26-pose", "Features (52)"), ("YOLO26-pose\n+ hand keypoints", "Features (60)"),
        ("YOLO26-pose + hand\n+ head pose, gaze", "Features (65)")]
out = box(81, 11.8, 18.5, 5.4, "Mean\n(anomaly score)", ORANGE)
for i, (det, feat) in enumerate(rows):
    y = 18.4 - i * 6.6
    b1 = box(0.5, y, 19.5, 5.4, det)
    b2 = box(23.5, y, 19.5, 5.4, feat)
    b3 = box(46.5, y, 14, 5.4, "Autoencoder\n(TCN)")
    b4 = box(64, y, 14, 5.4, "Normalized\nerror")
    arrow(b1, b2), arrow(b2, b3), arrow(b3, b4)
    ax.annotate("", xy=(out[0], out[1] + out[3] / 2), xytext=(b4[0] + b4[2], y + 2.7),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0, shrinkA=0, shrinkB=0))
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"))
fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=110)
