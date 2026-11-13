"""Collect every result JSON into (a) a long CSV and (b) a markdown table.

    python summarize.py --out results --methods part1_template part2_patchnn

The markdown block can be pasted straight into the report draft / slides.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from common import summarize


def load_long(out_dir: str, method: str):
    folder = os.path.join(out_dir, method)
    rows = []
    if not os.path.isdir(folder):
        return rows
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as fh:
            r = json.load(fh)
        rows.append({
            "method": method,
            "category": r["category"],
            "shot": r["shot"],
            "seed": r["seed"],
            "image_auroc": r.get("image_auroc", ""),
            "pixel_auroc": r.get("pixel_auroc", ""),
            "fpr_at_threshold": r.get("fpr_at_threshold", ""),
            "fnr_at_threshold": r.get("fnr_at_threshold", ""),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--methods", nargs="+", default=["part1_template", "part2_patchnn"])
    ap.add_argument("--csv", default="results/summary.csv")
    args = ap.parse_args()

    all_rows = []
    for m in args.methods:
        all_rows.extend(load_long(args.out, m))

    if all_rows:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {args.csv} ({len(all_rows)} rows)\n")

    summaries = {m: summarize(args.out, m) for m in args.methods}
    shots = sorted({s for m in summaries.values() for s in m})
    if not shots:
        print("no results found")
        return

    for metric, key in (("image AUROC", "image_auroc"), ("pixel AUROC", "pixel_auroc")):
        print(f"## {metric} (%), 6-category mean +- std over seeds\n")
        header = "| shot | " + " | ".join(args.methods) + " |"
        print(header)
        print("|" + "---|" * (len(args.methods) + 1))
        for shot in shots:
            cells = []
            for m in args.methods:
                s = summaries[m].get(shot)
                if not s or np.isnan(s[f"{key}_mean"]):
                    cells.append("–")
                else:
                    cells.append(f"{100 * s[f'{key}_mean']:.1f} ± {100 * s[f'{key}_std']:.1f}")
            print(f"| {shot}-shot | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
