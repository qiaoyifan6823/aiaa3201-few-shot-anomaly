"""Generate the representative outputs that go into anomaly_demo.zip.

    PYTHONPATH=.pylibs python3 demo/make_outputs.py

Writes one three-panel figure per test image (input | heatmap | overlay with the
verdict) into demo/outputs/, plus an index.md summarising scores and decisions.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from infer import Detector, overlay  # noqa: E402

# (category, subset, how many) — defects, normals and both error directions
PLAN = [("chewinggum", "bad", 4), ("chewinggum", "good", 2),
        ("candle", "bad", 3), ("candle", "good", 2),
        ("fryum", "bad", 3), ("fryum", "good", 1)]


def main() -> None:
    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for cat, subset, n in PLAN:
        det = Detector(cat)
        folder = os.path.join(ROOT, "data/VisA_pytorch/1cls", cat, "test", subset)
        images = sorted(os.listdir(folder))[:n]
        for name in images:
            path = os.path.join(folder, name)
            score, heat, info = det.score(path)
            stem = f"{cat}_{subset}_{os.path.splitext(name)[0]}"
            out = os.path.join(out_dir, f"{stem}.png")
            overlay(path, heat, out, score, info)
            truth = "NG" if subset == "bad" else "OK"
            correct = info["decision"] == truth
            rows.append((stem, cat, truth, info["decision"], score, info["threshold"], correct))
            print(f"{stem:34s} truth={truth} pred={info['decision']} score={score:.4f} "
                  f"({'ok' if correct else 'ERROR'})")

    n_ok = sum(1 for r in rows if r[6])
    with open(os.path.join(out_dir, "index.md"), "w", encoding="utf-8") as fh:
        fh.write("# Representative demo outputs\n\n")
        fh.write(f"{len(rows)} images, {n_ok} decisions agree with the ground truth "
                 f"({100 * n_ok / len(rows):.0f}%).\n\n")
        fh.write("| file | category | truth | decision | score | threshold |\n|---|---|---|---|---|---|\n")
        for stem, cat, truth, dec, score, thr, correct in rows:
            mark = "" if correct else " **(miss)**"
            fh.write(f"| {stem}.png{mark} | {cat} | {truth} | {dec} | {score:.4f} | {thr:.4f} |\n")
    print(f"\n{len(rows)} figures written to {out_dir} ({n_ok} correct)")


if __name__ == "__main__":
    main()
