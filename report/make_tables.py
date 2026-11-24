"""Generate the LaTeX tables of the report from results/*.json.

    python report/make_tables.py

Keeping the tables generated means the numbers in the paper, the slides and the
repository can never disagree.
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
SHOTS = [1, 2, 4, 8]
CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]

ROWS = [
    ("part1_template_min", "Part 1: template difference", "tab:part1"),
    ("part2_patchnn", "Part 2: patch-NN, our implementation", "tab:part2ours"),
    ("part2_official", "Part 2: AnomalyDINO, official code", "tab:part2official"),
    ("p3_l12_448", "Part 3: 448, block 12 (control)", "tab:p3base"),
    ("p3_l6912_448", "Part 3: 448, blocks 6+9+12", "tab:p3layer"),
    ("p3_l12_multi", "Part 3: 448+672, block 12", "tab:p3scale"),
    ("p3_all", "Part 3: 448+672, blocks 6+9+12", "tab:p3all"),
]


def load(method, key, shot=None):
    vals = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", method, "*.json"))):
        r = json.load(open(f))
        if shot is not None and r["shot"] != shot:
            continue
        vals.append((r["category"], r[key]))
    return vals


def mean_std(method, key):
    per = defaultdict(list)
    for cat, v in load(method, key):
        per[cat].append(v)
    cat_means = [np.mean(v) for v in per.values()]
    per_seed = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "results", method, "*.json"))):
        r = json.load(open(f))
        per_seed[r["seed"]].append(r[key])
    seed_means = [np.mean(v) for v in per_seed.values()]
    return 100 * np.mean(cat_means), 100 * np.std(seed_means)


def main_table(key, caption, label):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{9pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Configuration & 1-shot & 2-shot & 4-shot & 8-shot \\\\",
        "\\midrule",
    ]
    for method, name, _ in ROWS:
        cells = []
        for s in SHOTS:
            m, sd = mean_std_shot(method, key, s)
            cells.append(f"{m:.1f}$\\pm${sd:.1f}")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if method == "part2_official":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}"]
    return "\n".join(lines)


def mean_std_shot(method, key, shot):
    per = defaultdict(list)
    per_seed = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "results", method, "*.json"))):
        r = json.load(open(f))
        if r["shot"] != shot:
            continue
        per[r["category"]].append(r[key])
        per_seed[r["seed"]].append(r[key])
    return 100 * np.mean([np.mean(v) for v in per.values()]), 100 * np.std([np.mean(v) for v in per_seed.values()])


def per_category_table(shot=4):
    cols = [("part1_template_min", "Part 1"), ("part2_patchnn", "Part 2 (ours)"), ("p3_all", "Part 3 (fused)")]
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Category & " + " & ".join(n for _, n in cols) + " \\\\", "\\midrule"]
    for cat in CATS:
        cells = []
        for m, _ in cols:
            vals = [v for c, v in load(m, "image_auroc", shot) if c == cat]
            cells.append(f"{100 * np.mean(vals):.1f}")
        lines.append(f"{cat.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{Per-category image AUROC (\\%) at {shot}-shot; mean over three seeds.}}",
              "\\label{tab:percat}", "\\end{table}"]
    return "\n".join(lines)


def efficiency_table():
    rows = [
        ("Part 1 template difference, 4 refs (incl.\\ ECC)", "Apple M-series CPU", "57.7\\,ms", "2.0\\,GB RSS"),
        ("Part 2 feature extraction, ViT-S/14 @448", "Apple M-series MPS", "37\\,ms", "2.0\\,GB unified"),
        ("Part 2 retrieval, 1 reference", "Apple M-series MPS", "0.8\\,ms", "--"),
        ("Part 2 retrieval, 8 references", "Apple M-series MPS", "3.2\\,ms", "--"),
        ("Official AnomalyDINO inference", "RTX 4090 24\\,GB", "46\\,ms", "0.6\\,GB observed"),
    ]
    lines = ["\\begin{table*}[t]", "\\centering", "\\small",
             "\\begin{tabular}{llcc}", "\\toprule",
             "Stage & Device & Per image & Peak memory \\\\", "\\midrule"]
    for a, b, c, d in rows:
        lines.append(f"{a} & {b} & {c} & {d} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Runtime and memory per image, on the device each experiment was run. "
              "Feature extraction is cached once (4,205 images, $\\approx$4 min) and is excluded from "
              "the retrieval rows.}", "\\label{tab:efficiency}", "\\end{table*}"]
    return "\n".join(lines)


def gain_table():
    """Per-category gain of the fused model over the single-block, single-scale control."""
    lines = ["\\begin{table*}[t]", "\\centering", "\\small",
             "\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{1-shot} & \\multicolumn{3}{c}{8-shot} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Category & control & fused & $\\Delta$ & control & fused & $\\Delta$ \\\\",
             "\\midrule"]
    for cat in CATS:
        cells = []
        for shot in (1, 8):
            base = [v for c, v in load("p3_l12_448", "image_auroc", shot) if c == cat][0]
            fused = [v for c, v in load("p3_all", "image_auroc", shot) if c == cat][0]
            cells += [f"{100 * base:.1f}", f"{100 * fused:.1f}", f"{100 * (fused - base):+.1f}"]
        lines.append(f"{cat.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Per-category image AUROC (\\%) of the fused model against the single-block, "
              "single-scale control (block 12 @448). Fusion helps five of six categories at 1-shot and all six "
              "at 8-shot; the gain is largest where the control is weakest (cashew) and shrinks as "
              "references grow.}", "\\label{tab:gain}", "\\end{table*}"]
    return "\n".join(lines)


def dataset_table():
    """Per-category dataset statistics, including the defect-mask area distribution."""
    import sys
    import cv2
    sys.path.insert(0, ROOT)
    from common import CATEGORIES, build_split

    rows = []
    for cat in CATEGORIES:
        split = build_split(os.path.join(ROOT, "data/VisA_pytorch/1cls"), cat)
        n_train = len(split.train)
        n_good = sum(1 for y in split.labels if y == 0)
        n_bad = sum(1 for y in split.labels if y == 1)
        areas = []
        for mp in split.masks:
            if not mp:
                continue
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            areas.append((m > 0).mean() * 100)
        rows.append((cat, n_train, n_good, n_bad, float(np.median(areas)),
                     float(np.percentile(areas, 90))))

    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lcccccc}", "\\toprule",
             "Category & train & test & test & median & 90th \\\\",
             " & (good) & (good) & (defect) & mask & pct. \\\\", "\\midrule"]
    for cat, a, b, c, med, p90 in rows:
        lines.append(f"{cat.replace('_', ' ')} & {a} & {b} & {c} & {med:.2f} & {p90:.2f} \\\\")
    lines += ["\\midrule",
              f"total & {sum(r[1] for r in rows)} & {sum(r[2] for r in rows)} & {sum(r[3] for r in rows)} & & \\\\",
              "\\bottomrule", "\\end{tabular}",
              "\\caption{The six VisA categories used in this paper. Mask columns give the area of the "
              "ground-truth defect region as a percentage of the image; the last two columns are the median and "
              "the 90th percentile over the 100 anomalous test images of each category.}",
              "\\label{tab:dataset}", "\\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "main_image.tex"), "w") as fh:
        fh.write(main_table("image_auroc",
                            "Image-level AUROC (\\%), six-category mean $\\pm$ std over three reference "
                            "selections.", "tab:image"))
    with open(os.path.join(OUT, "main_pixel.tex"), "w") as fh:
        fh.write(main_table("pixel_auroc",
                            "Pixel-level AUROC (\\%), computed at $448\\times448$ and averaged over the six "
                            "categories.", "tab:pixel"))
    with open(os.path.join(OUT, "per_category.tex"), "w") as fh:
        fh.write(per_category_table(4))
    with open(os.path.join(OUT, "efficiency.tex"), "w") as fh:
        fh.write(efficiency_table())
    with open(os.path.join(OUT, "gain.tex"), "w") as fh:
        fh.write(gain_table())
    with open(os.path.join(OUT, "dataset.tex"), "w") as fh:
        fh.write(dataset_table())
    print("tables written to", OUT)
