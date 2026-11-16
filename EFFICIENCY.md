# Efficiency (measured, per the brief's requirement)

The brief asks for "average inference time per image and peak GPU memory **on the same device**;
identify the input resolution and whether feature extraction is included" (Section 6.2), and
"report runtime, peak GPU memory, and all test-time refinements" (Section 2). It also says a CUDA
GPU is **recommended**, not required (Section 6.1). This note records the device, the timing and
the memory for every stage we report.

| Stage | Device | Per-image time | Peak memory |
|---|---|---|---|
| Part 1 template difference, 4 references (includes ECC alignment) | Apple M-series **CPU** | **57.7 ms** | 2.0 GB process peak RSS |
| Part 2 feature extraction, ViT-S/14 @448 | Apple M-series **MPS** | **37 ms** | 2.0 GB unified memory |
| Part 2 scoring, 1 reference (excludes extraction) | MPS | **0.8 ms** | — |
| Part 2 scoring, 8 references (excludes extraction) | MPS | **3.2 ms** | — |
| Official AnomalyDINO inference | **RTX 4090 24 GB** (AutoDL) | **46 ms / sample** | ≈0.6 GB observed with `nvidia-smi` |

## How these were measured

* Part 1: 60 test images of `candle`, 4 references, wall-clock over the full alignment + three-channel
  difference + morphology pipeline, single process.
* Part 2 extraction: batch size 8, 448×448, `torch.mps.synchronize()` after each batch; the whole
  4,205-image dataset takes ≈4 min and is cached to disk (3.1 GB float16), so it is paid once.
* Part 2 scoring: 40 test images, `kNN` against the memory bank only, synchronised, excluding the
  one-off extraction.
* Official run: `mean inference time: 0.046 s/sample` printed by `run_anomalydino.py`; peak memory is
  the value observed with `nvidia-smi --query-gpu=memory.used` during the run — we did **not** record
  `torch.cuda.max_memory_allocated()`, so the figure is an observation, not an allocator peak.

## Why the main experiments did not use CUDA

The backbone has 22 M parameters and is only ever run forward. On a laptop GPU (MPS) the entire
4,205-image feature cache takes ≈4 minutes and one scoring configuration takes seconds, so the full
380-run grid was completed locally. The brief marks CUDA as recommended rather than required; we used
a rented 4090 only for the official-code reproduction, where the risk was the environment
(`torchvision` + `faiss`), not the compute.

## What to write in the report

* Report the two lines separately: main experiments on Apple MPS/CPU, official reproduction on a 4090 —
  never mix devices inside one comparison.
* State 448×448 as the input resolution for every reported number, and say explicitly whether the
  timing includes feature extraction.
* "Peak GPU memory" is not meaningful for unified memory; report process peak RSS for the Apple runs
  and the observed device memory for the 4090 run, with the method of measurement.
