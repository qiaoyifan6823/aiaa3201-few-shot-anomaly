# Why the Part 1 baseline fails (measured, not guessed)

The template-difference baseline reaches 71.7 → 78.3 image AUROC over 1 → 8 shots, but only
72.9 → 74.0 pixel AUROC, and on `chewinggum` its pixel AUROC is **below chance** (≈0.40) while
the patch-NN method reaches ≈0.97 on the same masks. This note records the three hypotheses we
tested and what the measurements say.

## H1 — "the aggregation over references is the problem"

We implemented three ways of comparing a test image with the reference set:

| strategy | 1-shot | 2-shot | 4-shot | 8-shot |
|---|---|---|---|---|
| `min` — per-pixel minimum over references | 71.7 | **74.8** | **76.5** | **78.3** |
| `best` — single closest reference (whole-image residual) | 71.7 | 72.7 | 74.5 | 75.3 |
| `median` — pixel-wise median reference image | 71.7 | 73.3 | 74.5 | 73.9 |

(image AUROC %, 6-category mean over seeds 0/1/2)

**Rejected.** Per-pixel `min` is already the best of the three; replacing it makes image AUROC
worse. It does help pixel AUROC at higher shots (`median`: 75.6 vs `min`: 74.0 at 8-shot), so the
two metrics prefer different aggregation — worth one sentence in the report, not a fix.

## H2 — "alignment to a single canonical template is the problem"

We added `--per-ref-align`, which re-estimates an affine transform for every (test image,
reference) pair instead of using one canonical anchor. Run on the two hardest categories
(1/2/4-shot, 3 seeds):

| category | metric | canonical align (`min`) | per-reference align |
|---|---|---|---|
| candle | image AUROC (4-shot) | 71.5 | 71.8 |
| candle | image AUROC (8-shot) | 75.3 | 75.9 |
| chewinggum | image AUROC (4-shot) | 82.7 | 83.0 |
| chewinggum | pixel AUROC (4-shot) | 38.9 | 39.4 |

**Rejected as the main cause.** It is ~4x slower and buys ≈0.5 AUROC point, and it does not move
the pathological pixel AUROC at all.

## H3 — "the defect signal is simply not in a pixel-wise residual" (accepted)

Direct measurement on six anomalous `chewinggum` test images, comparing the residual inside the
ground-truth mask with the residual outside it:

| image | mask area | mean diff inside | mean diff outside | ratio |
|---|---|---|---|---|
| 50 | 1.1 % | 0.216 | 0.218 | 0.99 |
| 51 | 1.0 % | 0.225 | 0.218 | 1.03 |
| 52 | 0.7 % | 0.210 | 0.227 | 0.93 |
| 53 | 0.5 % | 0.210 | 0.233 | 0.90 |
| 54 | 1.3 % | 0.267 | 0.231 | 1.16 |
| 55 | 1.3 % | 0.242 | 0.227 | 1.06 |

Two facts explain everything:

1. **The defects are tiny** — 0.5–1.3 % of the image. A category-averaged pixel AUROC is
   dominated by whether the map is quiet outside the mask.
2. **The residual is a near-constant ~0.22 everywhere.** Inside-mask and outside-mask residuals
   are indistinguishable (ratio 0.90–1.16). That offset is a global appearance shift
   (illumination / instance layout), which no global affine transform can remove. So the map
   carries "how much this product looks like the reference product in general", not "where the
   defect is" — hence chance-level, even anti-correlated, localisation.

This is the textbook limitation the course brief points at ("the baseline confuses pose,
background and illumination changes with defects"), and it is the cleanest motivation we have
for the feature-based method and for the Part 3 improvement: a pixel-wise residual cannot respond
to a small local appearance change, whereas a 14x14 patch token can — but only if the patch grid
is fine enough, which is exactly the multi-scale/multi-layer direction.

## Practical consequences

* Report `min` as the Part 1 baseline; keep the other two variants as an ablation table.
* Use `chewinggum` (pixel AUROC ≈ 0.40 for the baseline vs ≈ 0.97 for patch-NN) as the failure
  case figure.
* Do not spend more time tuning the classical baseline: the ceiling is set by the representation,
  not by the aggregation rule or the alignment.
