"""Part 3 qualitative figure: what the fusion changes on one image.

Columns: input | block 12 @448 | blocks 6+9+12 @448 | blocks 6+9+12 @448+672
Rows: two categories (chewing gum, fryum), same four references (shot 4, seed 0).

    PYTHONPATH=.pylibs python3 report/make_part3_figure.py
"""

import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "report"))

from common import build_split, load_references, topk_score  # noqa: E402
from part3_score import fuse, maps_for, load_scale  # noqa: E402

SIZE, SHOT, SEED = 448, 4, 0
CATS = ["chewinggum", "fryum"]


def main():
    os.chdir(ROOT)
    fig, axes = plt.subplots(len(CATS), 4, figsize=(13.2, 6.4))
    titles = ["Input", "block 12 @448", "blocks 6+9+12 @448", "blocks 6+9+12 @448+672"]

    for r, cat in enumerate(CATS):
        split = build_split("data/VisA_pytorch/1cls", cat)
        refs = load_references("refs", cat, SHOT, SEED)
        d448, d672 = load_scale("features", cat, 448), load_scale("features", cat, 672)
        test_paths = [str(p) for p in d448["test_paths"]]

        # pick the anomalous test image where the fused model is most confident
        keep = list(range(len(test_paths)))
        single = maps_for(d448, refs, keep, 12, SIZE, 4.0)
        multi = np.stack([maps_for(d448, refs, keep, b, SIZE, 4.0) for b in (6, 9, 12)], 0).mean(0)
        multi672 = fuse([multi, np.stack([maps_for(d672, refs, keep, b, SIZE, 4.0) for b in (6, 12)], 0).mean(0)], "mean")
        labels = [split.labels[split.test.index(p)] if p in split.test else 0 for p in test_paths]
        scores = [topk_score(m) for m in multi672]
        cand = [i for i, y in enumerate(labels) if y == 1] or keep
        idx = max(cand, key=lambda i: scores[i])

        rgb = cv2.cvtColor(cv2.resize(cv2.imread(test_paths[idx]), (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
        for c, m in enumerate([None, single[idx], multi[idx], multi672[idx]]):
            ax = axes[r][c]
            if m is None:
                ax.imshow(rgb)
            else:
                ax.imshow(m, cmap="inferno")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#B9C2CC")
            if r == 0:
                ax.set_title(titles[c], fontsize=10.5, color="#1F2933", pad=6)
        axes[r][0].set_ylabel(cat, fontsize=11, color="#1F2933")

    fig.text(0.5, 0.015, "Same test image and the same four references. Colour scale is per-image min-max.",
             ha="center", fontsize=8.6, color="#5B6770")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    out = os.path.join(HERE, "figs", "part3_qual.png")
    fig.savefig(out, dpi=220, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
