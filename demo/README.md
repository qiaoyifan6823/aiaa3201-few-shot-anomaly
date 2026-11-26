# Demo — few-shot anomaly detection

Upload (or point to) a product image; the demo returns

1. an **image-level anomaly score** (mean of the top 1 % of the fused patch-distance map),
2. an **OK / NG decision** against the threshold calibrated on 20 held-out normal images,
3. a **pixel-level heatmap** overlaid on the input.

It runs the model of the report: frozen DINOv2 ViT-S/14, blocks 6/9/12 at 448 and
blocks 6/12 at 672, distance maps averaged, same scoring rule as Parts 1–3.

## Install

```bash
pip install torch torchvision transformers opencv-python numpy gradio
```

(Only `transformers`, `opencv-python`, `numpy` and `gradio` are needed on top of
PyTorch. On an Apple-silicon laptop this installs in a couple of minutes.)

## Run

**Interactive (what was recorded):**

```bash
sh demo/run.sh              # open http://127.0.0.1:7860
```

The interface has two ways in: upload any product image (the box on the left) or
press one of the four **one-click sample** buttons, which score the bundled
images instantly - no upload, no clearing, so a screen recording has no dead
time. The three reference banks are preloaded at start-up, so every click returns
in well under a second.

**Single image from the command line:**

```bash
python demo/infer.py --image demo/samples/chewinggum_ng.jpg --category chewinggum
```

Output:

```
category : chewinggum
score    : 0.4765
threshold: 0.1584
decision : NG  (margin +0.3181)
figure   : demo/out/chewinggum_000_result.png
```

**Regenerate the representative outputs** (`outputs/`, index in `outputs/index.md`):

```bash
python demo/make_outputs.py
```

## What is in this folder

```
demo/
├── app.py             Gradio interface (upload → score / verdict / heatmap)
├── infer.py           Detector class + CLI; the actual model code
├── build_assets.py    rebuilds assets/ from the full feature cache (not needed to run)
├── make_outputs.py    writes outputs/ and outputs/index.md
├── assets/            reference feature banks + calibrated thresholds (prebuilt, ~68 MB)
├── samples/           5 example images used by the UI
└── outputs/           15 three-panel results, 14 of which agree with the ground truth
```

`assets/` holds the four reference images' patch features for each showcased
category and the per-category threshold, so the demo runs **without** the 26 GB
feature cache used for the experiments.

## Showcased categories

| category | why it is shown |
|---|---|
| `chewinggum` | small defect (median 0.53 % of the image) — the case where the classical baseline fails |
| `candle` | multiple instances in one frame |
| `fryum` | thin crack, elongated defect |

## Honest note on accuracy

The threshold is the 95th percentile of 20 normal images, as required by the
brief, so the operating point is deliberately conservative rather than tuned. On
the 15 showcased images 14 decisions agree with the ground truth; the miss is
`candle_bad_000` (score 0.1289 vs threshold 0.1319), i.e. a false negative of the
kind the report discusses in the limitations: with this threshold the false
negative rate on the full test split is above 50 %.
