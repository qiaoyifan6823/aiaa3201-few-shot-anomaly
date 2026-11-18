"""Part 3 step 2: multi-layer / multi-scale patch-NN fusion + evaluation.

For each (category, shot, seed):
    per layer b in --layers:  d_b = 1 - max cosine to the reference bank of block b
    per scale s in --scales:  upsample d^(s,b) to --size, Gaussian smooth
    fused map = mean (or z-score mean) of the selected maps
    image score = mean of the top 1 % of the fused map

    python part3_score.py --layers 6 9 12 --scales 448 --method-name p3_l6-9-12
    python part3_score.py --layers 12 --scales 448          --method-name p3_l12
    python part3_score.py --layers 6 9 12 --scales 448 672  --method-name p3_multi
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CATEGORIES, SEEDS, SHOTS, SIZE,
    balanced_indices, build_split, image_auroc, load_references, pixel_auroc,
    sample_references, save_references, save_result, topk_score,
)


def load_scale(feat_dir, cat, scale):
    path = os.path.join(feat_dir, f"{cat}_s{scale}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


def maps_for(data, refs, test_idx, layer, size, sigma):
    """One distance map per test image for one (scale, layer)."""
    import torch

    tr = [str(p) for p in data["train_paths"]]
    idx = {p: i for i, p in enumerate(tr)}
    if any(r not in idx for r in refs):
        return None
    if f"train_b{layer}" not in data:
        return None
    bank = np.concatenate([data[f"train_b{layer}"][idx[r]] for r in refs], axis=0).astype(np.float32)
    bank = torch.from_numpy(bank)
    grid = int(data["grid"])
    outs = []
    for i in test_idx:
        q = torch.from_numpy(data[f"test_b{layer}"][i].astype(np.float32))
        dist = (1.0 - (q @ bank.T).max(dim=1).values).numpy()
        m = cv2.resize(dist.reshape(grid, grid), (size, size), interpolation=cv2.INTER_LINEAR)
        outs.append(cv2.GaussianBlur(m, (0, 0), sigmaX=sigma))
    return outs


def fuse(maps_list, how):
    stack = np.stack(maps_list, axis=0)
    if how == "mean":
        return stack.mean(axis=0)
    # z-score each map with its own statistics, then average
    mu = stack.mean(axis=(1, 2), keepdims=True)
    sd = stack.std(axis=(1, 2), keepdims=True) + 1e-8
    return ((stack - mu) / sd).mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/VisA_pytorch/1cls")
    ap.add_argument("--feat-dir", default="features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--cats", nargs="+", default=CATEGORIES)
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--layers", nargs="+", type=int, default=[12])
    ap.add_argument("--scales", nargs="+", type=int, default=[448])
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--fusion", default="mean", choices=["mean", "zscore"])
    ap.add_argument("--topk", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--method-name", default="")
    args = ap.parse_args()

    method = args.method_name or (
        "p3_" + "+".join(f"l{''.join(map(str, args.layers))}") + "_" + "+".join(map(str, args.scales))
        + ("_z" if args.fusion == "zscore" else "")
    )
    os.makedirs(args.out, exist_ok=True)

    for cat in args.cats:
        datas = {s: load_scale(args.feat_dir, cat, s) for s in args.scales}
        missing = [s for s, d in datas.items() if d is None]
        if missing:
            print(f"[{cat}] missing features for scale(s) {missing}, skipped")
            continue
        split = build_split(args.root, cat)
        test_paths = [str(p) for p in datas[args.scales[0]]["test_paths"]]
        labels_all = [split.labels[split.test.index(p)] if p in split.test else 0 for p in test_paths]
        keep = balanced_indices(labels_all, args.limit)
        test_paths = [test_paths[i] for i in keep]
        labels = [labels_all[i] for i in keep]

        masks = []
        for p in test_paths:
            mp = split.masks[split.test.index(p)] if p in split.test else None
            if mp and os.path.exists(mp):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                masks.append(cv2.resize(m, (args.size, args.size), interpolation=cv2.INTER_NEAREST) > 0)
            else:
                masks.append(np.zeros((args.size, args.size), bool))

        for shot in args.shots:
            for seed in args.seeds:
                ref_file = os.path.join(args.refs, f"{cat}_shot{shot}_seed{seed}.json")
                if os.path.exists(ref_file):
                    refs = load_references(args.refs, cat, shot, seed)
                else:
                    refs = sample_references(split.train, shot, seed)
                    save_references(args.refs, cat, shot, seed, refs)

                per_scale = []
                ok = True
                for s in args.scales:
                    layer_maps = []
                    for b in args.layers:
                        m = maps_for(datas[s], refs, keep, b, args.size, args.sigma)
                        if m is None:
                            continue          # this layer is not cached at this scale
                        layer_maps.append(m)
                    if not layer_maps:
                        ok = False
                        break
                    if not ok:
                        break
                    per_scale.append(np.stack(layer_maps, axis=0).mean(axis=0))
                if not ok:
                    print(f"[{cat}] shot={shot} seed={seed}: refs missing in cache, skipped")
                    continue

                fused = fuse(per_scale, args.fusion)
                scores = [topk_score(m, args.topk) for m in fused]
                res = {
                    "image_auroc": image_auroc(scores, labels),
                    "pixel_auroc": pixel_auroc(list(fused), masks),
                    "n_test": len(labels), "size": args.size,
                    "layers": args.layers, "scales": args.scales,
                    "fusion": args.fusion, "sigma": args.sigma,
                }
                save_result(args.out, method, cat, shot, seed, res)
                print(f"[{cat}] shot={shot} seed={seed} img={100*res['image_auroc']:.1f} "
                      f"px={100*res['pixel_auroc']:.1f}")


if __name__ == "__main__":
    main()
