# Few-Shot Industrial Visual Anomaly Detection on VisA

Course project for **AIAA 3201 — Introduction to Computer Vision** (HKUST-GZ).

Given only a handful of defect-free reference images per product category, decide whether a
test image is anomalous (**OK / NG**) and localise the defect at pixel level.

This repository contains a deliberately small, dependency-light implementation of the whole
course protocol — a classical baseline, a DINOv2 patch-nearest-neighbour method, and the
measurement code shared by both — so that every experiment in the report is reproducible
from a single machine, with or without a GPU.

| Part | Method | What it is |
|---|---|---|
| 1 | `part1_template.py` | aligned template difference (`\|ΔRGB\|`, `\|ΔGray\|`, `1−SSIM`) + threshold calibration; three reference-aggregation strategies (`min` / `best` / `median`) |
| 2 | `part2_extract.py` + `part2_score.py` | DINOv2 ViT-S/14 @448 patch features + nearest-neighbour memory bank (AnomalyDINO-style) |
| 3 | (hook) | multi-layer / multi-scale feature fusion — see `--layer` note at the bottom |

## Course protocol implemented here

| Item | Setting |
|---|---|
| Dataset | VisA, 1-class split, **6 categories**: candle, capsules, cashew, chewinggum, fryum, pipe_fryum |
| Shots | 1 / 2 / 4 / 8 normal references |
| Seeds | 0 / 1 / 2 (reference selection) |
| Calibration | first 20 sorted normal training files per category, never used as references |
| OK/NG threshold | 95th percentile of the calibration image scores |
| Backbone | `facebook/dinov2-small` (ViT-S/14), 448×448 input → 32×32 patch tokens |
| Image metric | image AUROC (main) |
| Pixel metric | pixel AUROC over all test images, including normal images with all-zero masks |
| Aggregation | mean over 6 categories first, then mean ± std over the 3 seeds |

## Layout

```
p2/
├── common.py            # splits, reference sampling, metrics, result IO  (shared by all methods)
├── prepare_visa.py      # raw VisA -> MVTec-style 1cls layout (6 categories only)
├── check_env.py         # device report + DINOv2 throughput + runtime estimate
├── part1_template.py    # Part 1 baseline
├── part2_extract.py     # Part 2, step 1: cache DINOv2 patch features
├── part2_score.py       # Part 2, step 2: few-shot NN scoring + evaluation
├── summarize.py         # results/*.json -> long CSV + markdown tables
├── data/split_csv/1cls.csv   # vendored from amazon-science/spot-diff (Apache-2.0)
├── refs/                # sampled reference filenames per (category, shot, seed)
├── features/            # cached patch features (one .npz per category)
├── results/             # one JSON per (method, category, shot, seed)
└── figures/             # qualitative overlays
```

## Setup

```bash
# torch is assumed to be installed already; --system-site-packages avoids reinstalling it
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, opencv-python, scikit-image, scikit-learn, transformers, tqdm
python check_env.py                      # prints device, ms/image and estimated runtimes
```

If HuggingFace is slow, export `HF_ENDPOINT=https://hf-mirror.com` before running.

## Data

```bash
# 1.9 GB, official mirror
curl -O https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
mkdir -p data/VisA && tar -xf VisA_20220922.tar -C data/VisA   # archive already contains candle/, capsules/, ...

# reorganise only the six required categories
python prepare_visa.py --raw data/VisA --out data/VisA_pytorch/1cls
```

Expected output (counts come from the official release):

```
category      train  test_good   test_bad  masks  missing
candle          900        100       100    100        0
capsules        542         60       100    100        0
cashew          450         50       100    100        0
chewinggum      453         50       100    100        0
fryum           450         50       100    100        0
pipe_fryum      450         50       100    100        0

total images prepared: 4205
```

## Measured cost on an M-series MacBook (no GPU)

| Step | Cost |
|---|---|
| DINOv2-S/14 @448 forward | **37 ms/image** on MPS (77 ms on CPU) |
| `part2_extract.py`, all 4,205 images | ~4 min (3.1 GB of cached features) |
| `part1_template.py`, one category (12 configs) | 3-5 min |
| `part2_score.py`, one configuration | seconds |

So the whole 6 x 4 x 3 grid is a coffee break, not a GPU rental.

## Quick start (smoke test, ~2 minutes)

```bash
python part1_template.py --cats candle --shots 1 --seeds 0 --limit 40
python part2_extract.py  --cats candle --limit 40
python part2_score.py    --cats candle --shots 1 --seeds 0 --limit 40
python summarize.py      --methods part1_template part2_patchnn
```

If all four commands print a number, the pipeline works and you can scale it up.

## Full run

```bash
# Part 1: 6 categories x {1,2,4,8} shots x 3 seeds
python part1_template.py

# Part 2: cache features once, then score the whole grid (seconds per configuration)
python part2_extract.py
python part2_score.py

# optional: reference rotation augmentation (closer to the official implementation)
python part2_extract.py --rot-aug --overwrite
python part2_score.py --use-rot-aug

python summarize.py --methods part1_template part2_patchnn
```

Outputs:

* `results/<method>/<category>_shot<k>_seed<s>.json` — per-run metrics, threshold, FPR/FNR at the threshold
* `results/summary.csv` — long table, one row per run
* `figures/part1|part2/*.png` — overlay of the anomaly map on the input image
* `refs/*.json` — the exact reference filenames, reused by Part 1, Part 2 and Part 3

## Official vs re-implementation (Part 2)

Both were run with the same 6 categories, 1/2/4/8 references and seeds 0/1/2
(`results/part2_official` is produced from the official repo's `metrics_seed=*.json`).

| shot | ours (image) | official (image) | ours (pixel) | official (pixel) |
|---|---|---|---|---|
| 1 | 91.7 | 92.8 | 96.5 | 97.7 |
| 2 | 93.6 | 94.3 | 97.3 | 97.9 |
| 4 | 94.5 | 95.7 | 97.8 | 98.1 |
| 8 | 95.3 | 96.6 | 98.0 | 98.2 |

The official pipeline is 1.0-1.3 points better on image AUROC because it adds PCA background
masking and reference rotation augmentation. Both are documented in `REPRO_NOTES_autodl.md`
together with the two patches the reproduction needed (local DINOv2 hub cache, stride-4 pixel
metric). Our implementation is therefore a faithful simplified version, not a re-run.

## Part 1 variants (ablation)

`part1_template.py --strategy {min,best,median}` controls how a test image is compared with the
reference set: per-pixel minimum over references, the single closest reference, or a pixel-wise
median template. `--per-ref-align` additionally re-aligns the test image to every reference.

Measured (6 categories, 3 seeds): `min` is the best of the three on image AUROC (78.3 at 8-shot
vs 75.3 / 73.9), `median` is better on pixel AUROC at high shots (75.6 vs 74.0), and per-reference
alignment adds ≈0.5 point for ≈4x the cost. The classical baseline's real bottleneck is the
representation, not the aggregation or the alignment -- see `FAILURE_ANALYSIS.md`.

## Deliberate deviations from the official AnomalyDINO

Being explicit about these matters for the report:

1. **No PCA-based background masking.** The official "agnostic" preprocessing applies
   PCA masking where it helps. We apply none by default (Part 3 direction B is exactly this).
2. **Reference rotation augmentation is optional** (`--rot-aug`), off by default to keep the
   first local run cheap.
3. **Image score = mean of the top 1% of the smoothed map**, identical for Part 1 and Part 2,
   so that the comparison between methods is apples-to-apples.
4. **Pixel AUROC is computed at 448×448** (prediction resolution), masks resized with
   nearest neighbour. All methods use the same resolution, so the ranking is unaffected.

## Part 3 hook

`part2_score.py` fuses a single layer today. To study multi-layer fusion, extend
`part2_extract.py` to cache `last_hidden_state` of several blocks (e.g. blocks 6, 9, 12),
then average or concatenate the normalized per-layer distance maps inside `part2_score.py`.
The ablation grid is `{single layer, multi layer} × {448, 448+336} × {no mask, PCA mask}`,
all of which reuse the same cached features.
