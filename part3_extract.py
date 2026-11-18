"""Part 3 step 1: cache multi-layer, multi-scale DINOv2 patch features.

One forward pass per (image, scale) with ``output_hidden_states=True`` gives all
12 blocks, so caching three layers costs the same as caching one.

    python part3_extract.py --layers 6 9 12 --scales 448 672
    python part3_extract.py --cats candle --limit 40 --layers 6 12 --scales 448   # smoke test

Output: ``<feat_dir>/<category>_s<scale>.npz`` with keys
    train_paths (M,), train_b6 (M, P, D) float16, ... test_paths, test_b6, ...
    P = (scale // 14) ** 2, D = 384
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATEGORIES, build_split  # noqa: E402


def pick_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_batch(paths, size):
    out = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(img.transpose(2, 0, 1))
    return np.stack(out)


def extract_layers(model, paths, size, layers, batch, device):
    """Return {block_index: (N, P, D) float16} for the requested transformer blocks."""
    import torch

    feats = {b: [] for b in layers}
    for i in range(0, len(paths), batch):
        x = torch.from_numpy(load_batch(paths[i:i + batch], size)).to(device)
        with torch.no_grad():
            out = model(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True)
        for b in layers:
            h = out.hidden_states[b][:, 1:, :]                     # drop CLS
            h = torch.nn.functional.normalize(h, dim=-1)
            feats[b].append(h.float().cpu().numpy().astype(np.float16))
    return {b: np.concatenate(v, axis=0) for b, v in feats.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/VisA_pytorch/1cls")
    ap.add_argument("--feat-dir", default="features")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--cats", nargs="+", default=CATEGORIES)
    ap.add_argument("--layers", nargs="+", type=int, default=[6, 9, 12])
    ap.add_argument("--scales", nargs="+", type=int, default=[448])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="cap images per split (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import Dinov2Model

    device = pick_device(args.device)
    print(f"device: {device}  layers: {args.layers}  scales: {args.scales}")
    model = Dinov2Model.from_pretrained(args.model).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    os.makedirs(args.feat_dir, exist_ok=True)
    for scale in args.scales:
        for cat in args.cats:
            out_path = os.path.join(args.feat_dir, f"{cat}_s{scale}.npz")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"[{cat} @{scale}] exists, skipped")
                continue
            split = build_split(args.root, cat)
            train, test = list(split.train), list(split.test)
            if args.limit:
                train, test = train[:args.limit], test[:args.limit]
            if not train or not test:
                print(f"[{cat}] missing images, skipped")
                continue

            t0 = time.time()
            tr = extract_layers(model, train, scale, args.layers, args.batch, device)
            te = extract_layers(model, test, scale, args.layers, args.batch, device)
            payload = {
                "train_paths": np.array(train), "test_paths": np.array(test),
                "test_labels": np.array(list(split.labels)),
                "grid": np.array(scale // 14), "scale": np.array(scale),
                "model": np.array(args.model), "layers": np.array(args.layers),
            }
            for b in args.layers:
                payload[f"train_b{b}"] = tr[b]
                payload[f"test_b{b}"] = te[b]
            np.savez(out_path, **payload)
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"[{cat} @{scale}] train={len(train)} test={len(test)} "
                  f"P={(scale // 14) ** 2} {size_mb:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
