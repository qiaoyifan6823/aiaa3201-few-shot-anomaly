# 首批实验结果（6 类 VisA，本地 MPS，无 GPU）

协议：1/2/4/8 shot × seeds 0/1/2，DINOv2 ViT-S/14 @448，阈值取校准集 95 分位，pixel AUROC 在 448×448 上计算（含正常图的零 mask）。

## 汇总（6 类等权平均，mean ± std over 3 seeds）

### Image AUROC (%)

| shot | part1_template | part2_patchnn |
|---|---|---|
| 1-shot | 71.7 ± 1.8 | 91.7 ± 0.2 |
| 2-shot | 74.8 ± 2.0 | 93.6 ± 0.4 |
| 4-shot | 76.5 ± 0.7 | 94.5 ± 0.2 |
| 8-shot | 78.3 ± 0.4 | 95.3 ± 0.3 |

### Pixel AUROC (%)

| shot | part1_template | part2_patchnn |
|---|---|---|
| 1-shot | 72.9 ± 0.3 | 96.5 ± 0.5 |
| 2-shot | 73.4 ± 0.5 | 97.3 ± 0.2 |
| 4-shot | 73.7 ± 0.3 | 97.8 ± 0.1 |
| 8-shot | 74.0 ± 0.3 | 98.0 ± 0.1 |

## 逐类 Image AUROC (%) — 4-shot

| category | part1_template | part2_patchnn |
|---|---|---|
| candle | 71.5 | 92.5 |
| capsules | 67.8 | 92.5 |
| cashew | 90.4 | 93.7 |
| chewinggum | 82.7 | 96.9 |
| fryum | 76.6 | 93.5 |
| pipe_fryum | 70.4 | 97.8 |

## Part 1 在校准阈值下的表现（6 类平均）

| shot | FPR | FNR |
|---|---|---|
| 1-shot | 10.0% | 70.9% |
| 2-shot | 13.8% | 59.1% |
| 4-shot | 12.2% | 55.8% |
| 8-shot | 12.1% | 53.4% |

## 观察（可直接写进 progress）

1. 模板差异基线随 shot 缓慢上升（71.7 → 78.3），但 8-shot 仍远低于 patch-NN 的 1-shot（91.7）：参考图越多，逐像素取最小值会让正常图的差异也被压低，图像级对比度反而变钝。
2. 模板差异的失败集中在 candle / capsules / pipe_fryum 这类位置、姿态变化的类别（4-shot 67–71%），而 cashew（90.4%）这种对齐良好的类别明显更好 —— 正好是 brief 说的「pose / background 被当成缺陷」。
3. patch-NN 在 1-shot 就到 91.7，8-shot 95.3；像素级 96.5 → 98.0，定位几乎不是瓶颈，瓶颈在图像级判别与细小缺陷（后续 Part 3 的动机）。
4. Part 1 的阈值按 95 分位校准，实测 FPR ≈ 10%、FNR ≈ 71%：说明「按校准集定阈值」在分布偏移下会明显失效，这也是 Direction F（选择性预测/校准）的切入点。
