"""Shared utilities for the VisA few-shot anomaly detection pipeline.

Everything here is method-agnostic: dataset indexing, reference sampling,
score aggregation and metrics. Both the template-difference baseline
(``part1_template.py``) and the patch-NN method (``part2_score.py``) import
from this module so that the two share exactly the same split, the same
reference images and the same metric code.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Course protocol constants (AIAA 3201, Project 2, 31-Oct brief)
# ---------------------------------------------------------------------------
CATEGORIES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]
N_CALIB = 20          # first N sorted normal training files: calibration only
SHOTS = [1, 2, 4, 8]
SEEDS = [0, 1, 2]
SIZE = 448            # working resolution (DINOv2 ViT-S/14 -> 32x32 patches)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder: str) -> List[str]:
    """Sorted list of image files in ``folder`` (empty list if missing)."""
    if not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS)]
    return [os.path.join(folder, n) for n in sorted(names)]


def find_mask(gt_root: str, image_path: str) -> Optional[str]:
    """Locate the ground-truth mask for an anomalous test image."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for ext in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".bmp"):
        cand = os.path.join(gt_root, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def balanced_indices(labels, limit: int):
    """Deterministic balanced subset for smoke tests (half normal, half anomalous)."""
    if not limit or limit >= len(labels):
        return list(range(len(labels)))
    normal = [i for i, y in enumerate(labels) if y == 0]
    bad = [i for i, y in enumerate(labels) if y == 1]
    k = max(1, limit // 2)
    keep = normal[:k] + bad[:k]
    if len(keep) < limit:
        rest = [i for i in range(len(labels)) if i not in set(keep)]
        keep += rest[:limit - len(keep)]
    return sorted(keep[:limit])


@dataclass
class Split:
    category: str
    train: List[str]          # normal training images, sorted
    test: List[str]           # test images (good + bad), sorted by path
    labels: List[int]         # 0 = normal, 1 = anomalous
    masks: List[Optional[str]]


def build_split(root: str, category: str) -> Split:
    """Build the 1-class split for one category.

    Expected layout (produced by ``prepare_visa.py``)::

        <root>/<category>/train/good/*.JPG
        <root>/<category>/test/good/*.JPG
        <root>/<category>/test/bad/*.JPG
        <root>/<category>/ground_truth/bad/*.png
    """
    cat_root = os.path.join(root, category)
    train = list_images(os.path.join(cat_root, "train", "good"))
    good = list_images(os.path.join(cat_root, "test", "good"))
    bad = list_images(os.path.join(cat_root, "test", "bad"))
    gt_root = os.path.join(cat_root, "ground_truth", "bad")

    test, labels, masks = [], [], []
    for p in sorted(good):
        test.append(p)
        labels.append(0)
        masks.append(None)
    for p in sorted(bad):
        test.append(p)
        labels.append(1)
        masks.append(find_mask(gt_root, p))
    return Split(category=category, train=train, test=test, labels=labels, masks=masks)


# ---------------------------------------------------------------------------
# Reference selection -- identical for every method, saved and reused
# ---------------------------------------------------------------------------
def reference_pool(train: List[str]) -> List[str]:
    """Normal images that may be sampled as references.

    The first ``N_CALIB`` sorted files are reserved for demo threshold
    calibration and must never enter the memory bank.
    """
    return train[N_CALIB:]


def sample_references(train: List[str], shot: int, seed: int) -> List[str]:
    pool = reference_pool(train)
    if shot > len(pool):
        raise ValueError(f"shot={shot} but only {len(pool)} reference images available")
    rng = random.Random(seed)
    return sorted(rng.sample(pool, shot))


def refs_path(refs_dir: str, category: str, shot: int, seed: int) -> str:
    return os.path.join(refs_dir, f"{category}_shot{shot}_seed{seed}.json")


def save_references(refs_dir: str, category: str, shot: int, seed: int, refs: List[str]) -> str:
    os.makedirs(refs_dir, exist_ok=True)
    path = refs_path(refs_dir, category, shot, seed)
    payload = {
        "category": category,
        "shot": shot,
        "seed": seed,
        "n_calib_excluded": N_CALIB,
        "references": [os.path.basename(p) for p in refs],
        "full_paths": refs,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def load_references(refs_dir: str, category: str, shot: int, seed: int) -> List[str]:
    with open(refs_path(refs_dir, category, shot, seed), encoding="utf-8") as fh:
        return json.load(fh)["full_paths"]


# ---------------------------------------------------------------------------
# Scores and metrics
# ---------------------------------------------------------------------------
def topk_score(pixel_map: np.ndarray, frac: float = 0.01) -> float:
    """Image-level score = mean of the top-``frac`` pixels of the anomaly map."""
    values = np.asarray(pixel_map, dtype=np.float32).ravel()
    k = max(1, int(round(values.size * frac)))
    return float(np.sort(values)[-k:].mean())


def image_auroc(scores: List[float], labels: List[int]) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels, dtype=np.int64)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=np.float64)))


def pixel_auroc(maps: List[np.ndarray], masks: List[np.ndarray], stride: int = 1) -> float:
    """Pixel-level AUROC over all test images (normal ones contribute all-zero masks)."""
    from sklearn.metrics import roc_auc_score

    ys, ss = [], []
    for m, gt in zip(maps, masks):
        ys.append((gt.ravel()[::stride] > 0).astype(np.uint8))
        ss.append(m.astype(np.float32).ravel()[::stride])
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------
def save_result(out_dir: str, method: str, category: str, shot: int, seed: int, payload: dict) -> str:
    folder = os.path.join(out_dir, method)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{category}_shot{shot}_seed{seed}.json")
    payload = dict(payload)
    payload.update({"method": method, "category": category, "shot": shot, "seed": seed})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def summarize(out_dir: str, method: str) -> Dict[str, dict]:
    """Aggregate results into mean +- std over seeds (6 categories averaged first)."""
    from collections import defaultdict

    folder = os.path.join(out_dir, method)
    table = defaultdict(dict)          # (shot, seed) -> {category: run}
    if not os.path.isdir(folder):
        return {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as fh:
            r = json.load(fh)
        table[(r["shot"], r["seed"])][r["category"]] = r

    summary = {}
    for shot in sorted({k[0] for k in table}):
        per_seed_img, per_seed_px, cats = [], [], set()
        for (s, seed), runs in sorted(table.items()):
            if s != shot:
                continue
            img = [r.get("image_auroc", np.nan) for r in runs.values()]
            px = [r.get("pixel_auroc", np.nan) for r in runs.values()]
            cats.update(runs.keys())
            per_seed_img.append(float(np.nanmean(img)))
            per_seed_px.append(float(np.nanmean(px)))
        summary[shot] = {
            "image_auroc_mean": float(np.mean(per_seed_img)),
            "image_auroc_std": float(np.std(per_seed_img)),
            "pixel_auroc_mean": float(np.mean(per_seed_px)),
            "pixel_auroc_std": float(np.std(per_seed_px)),
            "n_categories": len(cats),
            "n_seeds": len(per_seed_img),
        }
    return summary
