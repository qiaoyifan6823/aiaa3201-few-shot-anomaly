"""Part 2 step 1: cache DINOv2 patch features for every image of every category.

Features are extracted **once** and reused by every shot/seed configuration,
which is what makes the full 6 x 4 x 3 experiment grid cheap.

    python part2_extract.py --root data/VisA_pytorch/1cls --feat-dir features
    python part2_extract.py --cats candle --limit 32          # smoke test

Output: ``<feat_dir>/<category>.npz`` with keys
    train_paths (M,), train_feats (M, P, D) float16,
    test_paths  (K,), test_feats  (K, P, D) float16,
    grid, size, model

``P = (size // 14) ** 2`` patch tokens, ``D = 384`` for ViT-S/14.
With ``--rot-aug`` the training features are additionally stored for
0/90/180/270 degree rotations (``train_feats_rot``), which approximates the
reference augmentation used by the official AnomalyDINO implementation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATEGORIES, SIZE, balanced_indices, build_split  # noqa: E402


def pick_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_batch(paths, size, rotations=0):
    """Load images as a float32 array (B, 3, size, size) in [0, 1]."""
    out = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if rotations:
            img = np.rot90(img, rotations // 90)
        out.append(img.transpose(2, 0, 1))
    return np.stack(out)


def extract(model, paths, size, batch, device, rotations=0):
    import torch

    feats = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        x = torch.from_numpy(load_batch(chunk, size, rotations)).to(device)
        with torch.no_grad():
            out = model(pixel_values=x, interpolate_pos_encoding=True)
        patch = out.last_hidden_state[:, 1:, :]                    # drop CLS token
        patch = torch.nn.functional.normalize(patch, dim=-1)
        feats.append(patch.float().cpu().numpy().astype(np.float16))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 1, 1), np.float16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/VisA_pytorch/1cls")
    ap.add_argument("--feat-dir", default="features")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--cats", nargs="+", default=CATEGORIES)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="cap images per split (smoke test)")
    ap.add_argument("--rot-aug", action="store_true", help="also cache 4 rotations of train images")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import Dinov2Model

    device = pick_device(args.device)
    print(f"device: {device}")
    model = Dinov2Model.from_pretrained(args.model).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    os.makedirs(args.feat_dir, exist_ok=True)
    grid = args.size // 14

    for cat in args.cats:
        out_path = os.path.join(args.feat_dir, f"{cat}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{cat}] exists, skipped (use --overwrite)")
            continue

        split = build_split(args.root, cat)
        train, test = list(split.train), list(split.test)
        test_labels = list(split.labels)
        if args.limit:
            train = train[:args.limit]
            keep = balanced_indices(test_labels, args.limit)
            test = [test[i] for i in keep]
            test_labels = [test_labels[i] for i in keep]
        if not train or not test:
            print(f"[{cat}] missing images, skipped")
            continue

        t0 = time.time()
        train_feats = extract(model, train, args.size, args.batch, device)
        test_feats = extract(model, test, args.size, args.batch, device)
        payload = {
            "train_paths": np.array(train),
            "train_feats": train_feats,
            "test_paths": np.array(test),
            "test_feats": test_feats,
            "test_labels": np.array(test_labels),
            "grid": np.array(grid),
            "size": np.array(args.size),
            "model": np.array(args.model),
        }
        if args.rot_aug:
            rots = []
            for rot in (90, 180, 270):
                rots.append(extract(model, train, args.size, args.batch, device, rotations=rot))
            payload["train_feats_rot"] = np.stack(rots, axis=1)   # (M, 3, P, D)
        np.savez(out_path, **payload)
        print(f"[{cat}] train={len(train)} test={len(test)} "
              f"feats={train_feats.shape} in {time.time() - t0:.1f}s -> {out_path}")

    print("done. next: python part2_score.py")


if __name__ == "__main__":
    main()
