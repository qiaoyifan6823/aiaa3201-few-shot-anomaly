"""Build the compact demo assets: reference banks + calibrated OK/NG thresholds.

The full feature cache is 26 GB; the demo only needs the features of the four
reference images of each showcased category plus a threshold, so this script
writes ``demo/assets/<category>.npz`` (a few MB per category) and the demo runs
on a laptop without the full cache.

    PYTHONPATH=.pylibs python3 demo/build_assets.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from common import build_split, load_references, topk_score  # noqa: E402

CATEGORIES = ["chewinggum", "candle", "fryum"]
SHOT, SEED = 4, 0
SIZE = 448
SCALES = {448: (6, 9, 12), 672: (6, 12)}


def load_cache(cat, scale):
    data = np.load(os.path.join("features", f"{cat}_s{scale}.npz"), allow_pickle=False)
    paths = [str(p) for p in data["train_paths"]]
    return data, {p: i for i, p in enumerate(paths)}


def dist_map(q, bank, grid):
    """1 - max cosine, upsampled to 448 and smoothed (same rule as the paper)."""
    import torch

    bank = bank.reshape(-1, bank.shape[-1])          # (R, P, D) -> (R*P, D)
    sim = torch.from_numpy(q.astype(np.float32)) @ torch.from_numpy(bank.astype(np.float32)).T
    d = (1.0 - sim.max(dim=1).values).numpy()
    m = cv2.resize(d.reshape(grid, grid), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(m, (0, 0), sigmaX=4.0)


def fused_map_from_assets(assets, feats_by_scale):
    """Fuse the per (scale, layer) distance maps, exactly as part3_score does."""
    per_scale = []
    for scale, layers in SCALES.items():
        maps = [dist_map(feats_by_scale[(scale, b)], assets[f"bank_{scale}_{b}"], int(assets[f"grid_{scale}"]))
                for b in layers]
        per_scale.append(np.stack(maps, 0).mean(0))
    return np.stack(per_scale, 0).mean(0)


def main() -> None:
    os.chdir(ROOT)
    out_dir = os.path.join(HERE, "assets")
    os.makedirs(out_dir, exist_ok=True)

    for cat in CATEGORIES:
        refs = load_references("refs", cat, SHOT, SEED)
        split = build_split("data/VisA_pytorch/1cls", cat)
        calib = split.train[:20]

        payload = {"references": np.array(refs), "shot": np.array(SHOT), "seed": np.array(SEED)}
        for scale, layers in SCALES.items():
            data, idx = load_cache(cat, scale)
            payload[f"grid_{scale}"] = np.array(int(data["grid"]))
            for b in layers:
                payload[f"bank_{scale}_{b}"] = data[f"train_b{b}"][[idx[r] for r in refs]]
            # keep the calibration features too: they are only 20 images
            for b in layers:
                payload[f"calib_{scale}_{b}"] = data[f"train_b{b}"][[idx[p] for p in calib]]

        # calibrated threshold = 95th percentile of the 20 held-out normal images
        scores = []
        for i in range(len(calib)):
            feats = {(scale, b): payload[f"calib_{scale}_{b}"][i]
                     for scale, layers in SCALES.items() for b in layers}
            scores.append(topk_score(fused_map_from_assets(payload, feats)))
        payload["calibration_scores"] = np.array(scores, dtype=np.float32)
        payload["threshold"] = np.array(float(np.percentile(scores, 95)))
        # the calibration features were only needed to compute the threshold
        for k in [k for k in payload if k.startswith("calib_")]:
            del payload[k]

        out_path = os.path.join(out_dir, f"{cat}.npz")
        np.savez(out_path, **payload)
        print(f"{cat}: {len(refs)} refs, threshold={payload['threshold']:.4f}, "
              f"{os.path.getsize(out_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
