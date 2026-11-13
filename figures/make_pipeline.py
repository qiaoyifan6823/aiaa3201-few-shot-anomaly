"""Draw the project pipeline figure (Part 1 / Part 2 / Part 3) as a PNG.

    MPLCONFIGDIR=/tmp/mpl python3 figures/make_pipeline.py

Output: figures/pipeline.png (300 dpi) -- put it on the progress slide and in
the Method section of the report.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.png")

C1 = "#3F6BA8"   # Part 1
C2 = "#2E7D32"   # Part 2
C3 = "#C07616"   # Part 3
INK = "#1F2933"
GREY = "#5B6770"
RULE = "#B9C2CC"

W, H = 124, 74
fig, ax = plt.subplots(figsize=(13.6, 8.1))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

ROW = {"p1": 55.0, "p2": 37.0, "p3": 18.5}
X0, BW, GAP = 33.0, 15.8, 2.6
XS = [X0 + i * (BW + GAP) for i in range(5)]


def box(x, y, w, h, text, color, fill="white", dashed=False, fs=9.6, sub_fs=8.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
        linewidth=1.7, edgecolor=color, facecolor=fill,
        linestyle=(0, (4, 3)) if dashed else "solid", zorder=3))
    ax.text(x + w / 2, y + h / 2 + 1.7, text, ha="center", va="center",
            fontsize=fs, color=INK, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h / 2 - 2.3, sub, ha="center", va="center",
            fontsize=sub_fs, color=GREY, zorder=4) if False else None


def cell(x, yc, text, sub, color, fill="white", dashed=False, fs=9.6, sub_fs=8.0, h=9.6):
    box(x, yc - h / 2, BW, h, text, color, fill, dashed, fs, sub_fs)
    ax.text(x + BW / 2, yc - h / 2 + 3.0, sub, ha="center", va="center",
            fontsize=sub_fs, color=GREY, zorder=4)


def arrow(x0, y0, x1, y1, color=GREY, dashed=False, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=color, zorder=2, linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=0.5, shrinkB=0.5))


def row_title(yc, name, desc, color):
    ax.text(2.0, yc + 1.9, name, fontsize=12.5, color=color, fontweight="bold",
            va="center", ha="left")
    ax.text(2.0, yc - 2.4, desc, fontsize=8.6, color=color, style="italic",
            va="center", ha="left")


# ---------------------------------------------------------------- input blocks
def input_box(y, title, sub):
    ax.add_patch(FancyBboxPatch((14.0, y), 13.8, 8.4,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                linewidth=1.3, edgecolor=RULE, facecolor="#F7F9FC", zorder=3))
    ax.text(20.9, y + 5.2, title, ha="center", va="center", fontsize=9.4,
            color=INK, fontweight="bold", zorder=4)
    ax.text(20.9, y + 2.3, sub, ha="center", va="center", fontsize=7.4,
            color=GREY, zorder=4)


input_box(46.0, "Reference set", "1 / 2 / 4 / 8 normal images")
input_box(34.4, "Query image", "448 x 448")

# vertical bus feeding the three rows
BUS = 31.0
ax.plot([BUS, BUS], [ROW["p3"], ROW["p1"]], color=RULE, linewidth=1.4, zorder=1)
arrow(28.2, 50.2, BUS, 50.2)
arrow(28.2, 38.6, BUS, 38.6)
for key in ("p1", "p2"):
    arrow(BUS, ROW[key], XS[0] - 1.2, ROW[key])
arrow(BUS, ROW["p3"], XS[0] - 1.2, ROW["p3"], C3, dashed=True)
for key in ("p1", "p2", "p3"):
    ax.plot([BUS], [ROW[key]], marker="o", ms=3.2, color=RULE, zorder=2)

# ---------------------------------------------------------------- Part 1
row_title(ROW["p1"], "Part 1", "template difference", C1)
CELLS1 = [
    ("ECC affine", "to canonical reference"),
    ("Diff maps", "|dRGB| |dGray| 1-SSIM"),
    ("Smooth + morph.", "Gaussian, open/close"),
    ("Anomaly map", "image score = top-1%"),
    ("OK / NG", "95th pct. of calibration"),
]
for x, (t, s) in zip(XS, CELLS1):
    cell(x, ROW["p1"], t, s, C1, "#F2F5FA")

# ---------------------------------------------------------------- Part 2
row_title(ROW["p2"], "Part 2", "patch retrieval", C2)
CELLS2 = [
    ("DINOv2 ViT-S/14", "frozen, 448 input"),
    ("Patch tokens", "32x32x384, L2-norm"),
    ("Memory bank", "all reference patches"),
    ("1 - max cosine", "nearest normal patch"),
    ("Map + score", "upsample, smooth"),
]
for x, (t, s) in zip(XS, CELLS2):
    cell(x, ROW["p2"], t, s, C2, "#F1F8F1")

# ---------------------------------------------------------------- Part 3
row_title(ROW["p3"], "Part 3", "our improvement (planned)", C3)
CELLS3 = [
    ("Blocks 6 / 9 / 12", "+ 448 / 672 scales"),
    ("Per-layer maps", "distance per layer"),
    ("Weighted fusion", "calibrated gate"),
    ("Fused map", "same scoring rule"),
    ("Better AUROC", "thin scratches"),
]
for x, (t, s) in zip(XS, CELLS3):
    cell(x, ROW["p3"], t, s, C3, "#FDF3E6", dashed=True)

# horizontal arrows inside rows
for key, color, dashed in (("p1", GREY, False), ("p2", GREY, False), ("p3", C3, True)):
    for i in range(4):
        arrow(XS[i] + BW + 0.15, ROW[key], XS[i + 1] - 0.35, ROW[key], color, dashed)

# Part 2 -> Part 3 dependency (reuses the same cached features)
arrow(XS[2] + BW / 2, ROW["p2"] - 4.9, XS[2] + BW / 2, ROW["p3"] + 5.0, C3, dashed=True)
ax.text(XS[2] + BW / 2 + 1.2, (ROW["p2"] + ROW["p3"]) / 2, "same cached features",
        fontsize=7.6, color=C3, va="center", ha="left", style="italic")

# ---------------------------------------------------------------- evaluation strip
ax.add_patch(FancyBboxPatch((2.0, 0.8), 119.5, 6.2,
                            boxstyle="round,pad=0.3,rounding_size=0.9",
                            linewidth=1.1, edgecolor=RULE, facecolor="#F7F9FC", zorder=3))
ax.text(5.0, 3.9,
        "Evaluation (identical for every method):   VisA 1-class split, 6 categories   |   "
        "1 / 2 / 4 / 8 shots  x  seeds 0/1/2   |   image AUROC (main) + pixel AUROC   |   "
        "OK/NG threshold from 20 held-out normal images",
        fontsize=8.8, color=INK, va="center", ha="left", zorder=4)

# ---------------------------------------------------------------- titles
ax.text(2.0, 70.6, "Few-shot industrial visual anomaly detection: pipeline",
        fontsize=14.5, color=INK, fontweight="bold", ha="left", va="center")
ax.text(2.0, 66.8, "Same references, same splits and same metrics in all three parts",
        fontsize=9.8, color=GREY, ha="left", va="center", style="italic")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT, dpi=300, facecolor="white")
print("wrote", OUT)
