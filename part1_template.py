"""Part 1 baseline: aligned template difference (classical normality model).

Pipeline (per category, per shot, per seed):

    reference images --(ECC affine to canonical template)--> aligned refs
    test image       --(ECC affine to canonical template)--> aligned test
    diff  = mean( |dRGB|, |dGray|, 1 - SSIM )   taken as the per-pixel minimum
            over the reference set  ("closest template" in image space)
    diff  = Gaussian smoothing + morphological open/close
    image score = mean of the top 1% pixels of the diff map
    OK/NG threshold = 95th percentile of the first 20 normal training images
                      (calibration only -- never used as references)

Usage:
    python part1_template.py --root data/VisA_pytorch/1cls --cats candle --shots 1 --seeds 0
    python part1_template.py --root data/VisA_pytorch/1cls          # full 6x4x3 grid
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CATEGORIES, N_CALIB, SEEDS, SHOTS, SIZE,
    balanced_indices, build_split, image_auroc, pixel_auroc, save_references,
    save_result, sample_references, topk_score,
)

ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)


def load_rgb(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def estimate_affine(ref_gray: np.ndarray, mov_gray: np.ndarray) -> np.ndarray | None:
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(ref_gray, mov_gray, warp, cv2.MOTION_AFFINE,
                                       ECC_CRITERIA, None, 5)
        return warp
    except cv2.error:
        return None


def apply_affine(img: np.ndarray, warp: np.ndarray | None, size: int) -> np.ndarray:
    if warp is None:
        return img
    return cv2.warpAffine(img, warp, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def diff_map(a_rgb: np.ndarray, a_gray: np.ndarray, b_rgb: np.ndarray, b_gray: np.ndarray) -> np.ndarray:
    d_rgb = np.abs(a_rgb - b_rgb).mean(axis=2)
    d_gray = np.abs(a_gray - b_gray)
    d_ssim = 1.0 - structural_similarity(a_gray, b_gray, data_range=1.0)
    d = (d_rgb + d_gray + d_ssim) / 3.0
    d = cv2.GaussianBlur(d, (0, 0), sigmaX=2.0)
    k = np.ones((3, 3), np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, k)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, k)
    return d.astype(np.float32)


def load_mask(path: str | None, size: int) -> np.ndarray:
    if path is None or not os.path.exists(path):
        return np.zeros((size, size), np.uint8)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((size, size), np.uint8)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def save_overlay(image: np.ndarray, score_map: np.ndarray, path: str) -> None:
    m = score_map - score_map.min()
    m = m / (m.max() + 1e-8)
    heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = (image * 255).astype(np.uint8)[:, :, ::-1]           # RGB -> BGR
    out = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, out)


def run_category(args, category: str) -> None:
    split = build_split(args.root, category)
    if not split.train:
        print(f"[{category}] no training images, skipped")
        return
    if len(split.train) <= N_CALIB + max(args.shots):
        raise SystemExit(f"[{category}] not enough normal training images for shot={max(args.shots)}")

    test_idx = balanced_indices(split.labels, args.limit)
    n_test = len(test_idx)

    canonical_path = split.train[N_CALIB]                      # fixed alignment anchor
    canonical = load_rgb(canonical_path, args.size)
    canonical_gray = to_gray(canonical)

    # --- align every test image once (alignment does not depend on the refs) ---
    t0 = time.time()
    tests = []
    for idx in test_idx:
        rgb = load_rgb(split.test[idx], args.size)
        gray = to_gray(rgb)
        warp = estimate_affine(canonical_gray, gray) if args.align else None
        tests.append({
            "rgb": apply_affine(rgb, warp, args.size),
            "gray": apply_affine(gray, warp, args.size),
            "label": split.labels[idx],
            "mask": load_mask(split.masks[idx], args.size),
            "path": split.test[idx],
        })
    labels = [t["label"] for t in tests]
    masks = [t["mask"] for t in tests]
    print(f"[{category}] {n_test} test images aligned in {time.time() - t0:.1f}s")

    # --- calibration images: threshold only, never references ---
    calib = []
    for path in split.train[:N_CALIB]:
        rgb = load_rgb(path, args.size)
        gray = to_gray(rgb)
        warp = estimate_affine(canonical_gray, gray) if args.align else None
        calib.append((apply_affine(rgb, warp, args.size), apply_affine(gray, warp, args.size)))

    for shot in args.shots:
        for seed in args.seeds:
            t0 = time.time()
            refs = sample_references(split.train, shot, seed)
            save_references(args.refs, category, shot, seed, refs)

            ref_data = []
            for rp in refs:
                rgb = load_rgb(rp, args.size)
                gray = to_gray(rgb)
                warp = estimate_affine(canonical_gray, gray) if args.align else None
                ref_data.append((apply_affine(rgb, warp, args.size), apply_affine(gray, warp, args.size)))

            maps, scores = [], []
            for t in tests:
                best = None
                for ref_rgb, ref_gray in ref_data:
                    d = diff_map(t["rgb"], t["gray"], ref_rgb, ref_gray)
                    best = d if best is None else np.minimum(best, d)
                maps.append(best)
                scores.append(topk_score(best, args.topk))

            calib_scores = []
            for c_rgb, c_gray in calib:
                best = None
                for ref_rgb, ref_gray in ref_data:
                    d = diff_map(c_rgb, c_gray, ref_rgb, ref_gray)
                    best = d if best is None else np.minimum(best, d)
                calib_scores.append(topk_score(best, args.topk))
            threshold = float(np.percentile(calib_scores, 95))

            pred = [1 if s >= threshold else 0 for s in scores]
            tp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 1)
            fn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 1)
            fp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 0)
            tn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 0)

            result = {
                "image_auroc": image_auroc(scores, labels),
                "pixel_auroc": pixel_auroc(maps, masks, args.pixel_stride),
                "threshold_p95_calib": threshold,
                "fpr_at_threshold": fp / max(1, fp + tn),
                "fnr_at_threshold": fn / max(1, fn + tp),
                "n_test": len(labels),
                "size": args.size,
                "align": bool(args.align),
                "topk_frac": args.topk,
                "references": [os.path.basename(p) for p in refs],
            }
            save_result(args.out, "part1_template", category, shot, seed, result)
            print(f"[{category}] shot={shot} seed={seed} "
                  f"img_auroc={result['image_auroc']:.4f} px_auroc={result['pixel_auroc']:.4f} "
                  f"thr={threshold:.4f} ({time.time() - t0:.1f}s)")

            if args.examples and shot == args.shots[0] and seed == SEEDS[0]:
                order = np.argsort(scores)
                picks = [order[-1]]                                  # most anomalous-looking
                normal_idx = [i for i, y in enumerate(labels) if y == 0]
                if normal_idx:
                    picks.append(max(normal_idx, key=lambda i: scores[i]))
                for rank, i in enumerate(picks):
                    tag = "anomalous" if labels[i] == 1 else "normal_fp"
                    save_overlay(tests[i]["rgb"], maps[i],
                                 os.path.join(args.examples_dir, f"{category}_{tag}_{rank}.png"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/VisA_pytorch/1cls")
    ap.add_argument("--out", default="results")
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--cats", nargs="+", default=CATEGORIES)
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--topk", type=float, default=0.01)
    ap.add_argument("--pixel-stride", type=int, default=1, help=">1 subsamples pixels (memory)")
    ap.add_argument("--limit", type=int, default=0, help="cap test images per category (smoke test)")
    ap.add_argument("--no-align", dest="align", action="store_false")
    ap.add_argument("--examples", type=int, default=2, help="number of overlay PNGs per category")
    ap.add_argument("--examples-dir", default="figures/part1")
    args = ap.parse_args()

    args.refs = os.path.abspath(args.refs)
    os.makedirs(args.out, exist_ok=True)
    for cat in args.cats:
        run_category(args, cat)


if __name__ == "__main__":
    main()
