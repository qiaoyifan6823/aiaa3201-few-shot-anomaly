"""Part 2 step 2: few-shot patch nearest-neighbour scoring (AnomalyDINO-style).

For each (category, shot, seed):
    reference patch bank  = all patch features of the sampled normal images
    patch score           = 1 - max cosine similarity to the reference bank
    anomaly map           = patch grid -> bilinear upsample -> Gaussian smoothing
    image score           = mean of the top 1% pixels

``--layer`` additionally allows fusing several encoder blocks (needs the extra
features written by ``part2_extract.py --layers``); this is the hook used by the
Part 3 improvement (multi-layer / multi-scale feature fusion).

Usage:
    python part2_score.py --feat-dir features --cats candle --shots 1 2 4 8
    python part2_score.py --feat-dir features            # full grid
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
    save_result, sample_references, save_references, topk_score,
)


def pick_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def upsample_map(dist: np.ndarray, size: int, sigma: float = 4.0) -> np.ndarray:
    m = cv2.resize(dist.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma)
    return m


def save_overlay(path: str, image_path: str, score_map: np.ndarray, size: int) -> None:
    base = cv2.imread(image_path, cv2.IMREAD_COLOR)
    base = cv2.resize(base, (size, size), interpolation=cv2.INTER_AREA)
    m = score_map - score_map.min()
    m = m / (m.max() + 1e-8)
    heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/VisA_pytorch/1cls")
    ap.add_argument("--feat-dir", default="features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--cats", nargs="+", default=CATEGORIES)
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--topk", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--pixel-stride", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-rot-aug", action="store_true", help="use cached rotated reference features")
    ap.add_argument("--method-name", default="part2_patchnn")
    ap.add_argument("--examples", type=int, default=2)
    ap.add_argument("--examples-dir", default="figures/part2")
    args = ap.parse_args()

    import torch

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    for cat in args.cats:
        npz_path = os.path.join(args.feat_dir, f"{cat}.npz")
        if not os.path.exists(npz_path):
            print(f"[{cat}] missing {npz_path}, run part2_extract.py first")
            continue
        data = np.load(npz_path, allow_pickle=False)
        train_paths = [str(p) for p in data["train_paths"]]
        test_paths = [str(p) for p in data["test_paths"]]
        train_feats = data["train_feats"]
        test_feats = data["test_feats"]
        rot_feats = data["train_feats_rot"] if (args.use_rot_aug and "train_feats_rot" in data) else None
        grid = int(data["grid"])
        size = int(data["size"])

        split = build_split(args.root, cat)
        label_by_path = dict(zip(split.test, split.labels))
        mask_by_path = dict(zip(split.test, split.masks))
        labels_all = [label_by_path.get(p, 0) for p in test_paths]
        keep = balanced_indices(labels_all, args.limit)
        test_paths = [test_paths[i] for i in keep]
        test_feats = test_feats[keep]
        labels = [labels_all[i] for i in keep]
        masks = []
        for p in test_paths:
            mp = mask_by_path.get(p)
            if mp and os.path.exists(mp):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
                masks.append((m > 0).astype(np.uint8))
            else:
                masks.append(np.zeros((size, size), np.uint8))

        index = {p: i for i, p in enumerate(train_paths)}

        for shot in args.shots:
            for seed in args.seeds:
                # reuse the reference selection saved by part1 (creates it if missing)
                ref_file = os.path.join(args.refs, f"{cat}_shot{shot}_seed{seed}.json")
                if os.path.exists(ref_file):
                    refs = load_references(args.refs, cat, shot, seed)
                else:
                    refs = sample_references(split.train, shot, seed)
                    save_references(args.refs, cat, shot, seed, refs)

                missing = [r for r in refs if r not in index]
                if missing:
                    print(f"[{cat}] {len(missing)} references not in feature cache (limit mode?) -> skip")
                    continue

                bank = np.concatenate([train_feats[index[r]] for r in refs], axis=0)
                if rot_feats is not None:
                    bank = np.concatenate([bank] + [rot_feats[index[r]] for r in refs], axis=0)
                bank = torch.from_numpy(np.asarray(bank, dtype=np.float32)).to(device)

                maps, scores = [], []
                for i, p in enumerate(test_paths):
                    q = torch.from_numpy(np.asarray(test_feats[i], dtype=np.float32)).to(device)
                    sim = q @ bank.T                              # (P, R*P) cosine similarity
                    dist = (1.0 - sim.max(dim=1).values).cpu().numpy()
                    m = upsample_map(dist.reshape(grid, grid), size, args.sigma)
                    maps.append(m)
                    scores.append(topk_score(m, args.topk))

                res = {
                    "image_auroc": image_auroc(scores, labels),
                    "pixel_auroc": pixel_auroc(maps, masks, args.pixel_stride),
                    "n_test": len(labels),
                    "size": size,
                    "grid": grid,
                    "topk_frac": args.topk,
                    "sigma": args.sigma,
                    "rot_aug": bool(rot_feats is not None),
                    "references": [os.path.basename(r) for r in refs],
                }
                save_result(args.out, args.method_name, cat, shot, seed, res)
                print(f"[{cat}] shot={shot} seed={seed} "
                      f"img_auroc={res['image_auroc']:.4f} px_auroc={res['pixel_auroc']:.4f}")

                if args.examples and shot == args.shots[0] and seed == SEEDS[0]:
                    order = np.argsort(scores)
                    picks = [order[-1]]
                    normals = [j for j, y in enumerate(labels) if y == 0]
                    if normals:
                        picks.append(max(normals, key=lambda j: scores[j]))
                    for rank, j in enumerate(picks):
                        tag = "anomalous" if labels[j] == 1 else "normal_fp"
                        save_overlay(os.path.join(args.examples_dir, f"{cat}_{tag}_{rank}.png"),
                                     test_paths[j], maps[j], size)


if __name__ == "__main__":
    main()
