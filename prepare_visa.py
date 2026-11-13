"""Reorganise the raw VisA download into the MVTec-style 1-class layout.

Only the six categories required by the course brief are prepared:

    candle, capsules, cashew, chewinggum, fryum, pipe_fryum

Usage
-----
    python prepare_visa.py --raw data/VisA --out data/VisA_pytorch/1cls

The split file (``data/split_csv/1cls.csv``) is vendored from the official
SPot-the-Difference repository (Apache-2.0):
https://github.com/amazon-science/spot-diff
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATEGORIES  # noqa: E402


def binarise_mask(src: str, dst: str) -> None:
    img = Image.open(src)
    arr = np.array(img)
    arr[arr != 0] = 255
    Image.fromarray(arr).save(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/VisA", help="folder that contains candle/, capsules/, ...")
    ap.add_argument("--out", default="data/VisA_pytorch/1cls", help="destination root")
    ap.add_argument("--split-file", default="data/split_csv/1cls.csv")
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = ap.parse_args()

    if not os.path.isdir(args.raw):
        raise SystemExit(f"raw VisA folder not found: {args.raw}")

    for cat in args.categories:
        for sub in ("train/good", "test/good", "test/bad", "ground_truth/bad"):
            os.makedirs(os.path.join(args.out, cat, sub), exist_ok=True)

    counts = {c: {"train": 0, "good": 0, "bad": 0, "mask": 0, "missing_mask": 0}
              for c in args.categories}

    with open(args.split_file, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # header: object,split,label,image,mask
        for row in reader:
            if len(row) < 5:
                continue
            obj, split, label, image_rel, mask_rel = row[:5]
            if obj not in counts:
                continue
            is_bad = label.strip().lower() != "normal"
            name = os.path.basename(image_rel)
            src = os.path.join(args.raw, image_rel)
            if not os.path.exists(src):
                continue

            if split == "train":
                dst = os.path.join(args.out, obj, "train", "good", name)
                shutil.copyfile(src, dst)
                counts[obj]["train"] += 1
            else:
                sub = "bad" if is_bad else "good"
                dst = os.path.join(args.out, obj, "test", sub, name)
                shutil.copyfile(src, dst)
                counts[obj][sub] += 1
                if is_bad:
                    if mask_rel.strip():
                        m_src = os.path.join(args.raw, mask_rel)
                        m_dst = os.path.join(args.out, obj, "ground_truth", "bad",
                                             os.path.splitext(os.path.basename(mask_rel))[0] + ".png")
                        if os.path.exists(m_src):
                            binarise_mask(m_src, m_dst)
                            counts[obj]["mask"] += 1
                        else:
                            counts[obj]["missing_mask"] += 1
                    else:
                        counts[obj]["missing_mask"] += 1

    print(f"{'category':<12}{'train':>7}{'test_good':>11}{'test_bad':>11}{'masks':>7}{'missing':>9}")
    for cat in args.categories:
        c = counts[cat]
        print(f"{cat:<12}{c['train']:>7}{c['good']:>11}{c['bad']:>11}{c['mask']:>7}{c['missing_mask']:>9}")

    total = sum(c["train"] + c["good"] + c["bad"] for c in counts.values())
    print(f"\ntotal images prepared: {total}")
    if any(c["missing_mask"] for c in counts.values()):
        print("warning: some anomalous test images have no mask; they still count for image AUROC")


if __name__ == "__main__":
    main()
