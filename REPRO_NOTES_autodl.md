# Official AnomalyDINO run on AutoDL (RTX 4090)

Environment: Python 3.10.8, torch 2.1.2+cu121, torchvision 0.16.2+cu121,
faiss-gpu-cu12 1.12.0, opencv-python 4.11.0.86, timm 1.0.31, AnomalyDINO @ dd35b0f.

Data: VisA 1-class split (6 categories), uploaded from the local machine, 4205 images.

## Command

```bash
python run_anomalydino.py --dataset VisA --shots 1 2 4 8 --num_seeds 3 \
  --preprocess agnostic --data_root /root/autodl-tmp/p2/VisA_pytorch/1cls/ --eval_segm
```

Objects restricted to candle, capsules, cashew, chewinggum, fryum, pipe_fryum in src/utils.py.

## Two patches applied to the official repo (disclose in the report)

1. src/backbones.py - torch.hub parses repo info from github.com, which times out on this
   network; the loader now uses the locally cloned hub repo (source="local") when it exists.
   No numerical change.

2. src/post_eval.py - pixel metrics are computed on a regular pixel subsample (PIX_STRIDE,
   default 4) instead of every pixel. Without it, full-resolution Pixel-AUROC over 200 images
   at ~1.5 M pixels each takes ~10 minutes per category. The same setting is used for every
   official run, so the comparison between shots/seeds is unaffected.

## Notes

- DINOv2 ViT-S/14 weights (84 MB) are cached in ~/.cache/torch/hub/checkpoints.
- Mean inference time: 0.046 s/sample on the 4090.
