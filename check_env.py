"""Environment check: prints the compute device and times the DINOv2 backbone.

Run this once after installing dependencies -- it tells you whether the local
machine is fast enough to run the whole project without renting a GPU.

    python check_env.py                # auto device
    python check_env.py --device cpu   # force CPU
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"torch           : {torch.__version__}")
    print(f"device          : {device}")
    print(f"cuda available  : {torch.cuda.is_available()}")
    print(f"mps available   : {torch.backends.mps.is_available()}")

    if device == "cuda":
        print(f"gpu             : {torch.cuda.get_device_name(0)}")

    from transformers import Dinov2Model

    t0 = time.time()
    model = Dinov2Model.from_pretrained(args.model).eval().to(device)
    print(f"load model      : {time.time() - t0:.1f}s")

    x = torch.randn(args.batch, 3, args.size, args.size, device=device)
    with torch.no_grad():
        out = model(pixel_values=x, interpolate_pos_encoding=True)   # warm-up
        tokens = out.last_hidden_state.shape
        t0 = time.time()
        for _ in range(args.iters):
            model(pixel_values=x, interpolate_pos_encoding=True)
        per_img = (time.time() - t0) / (args.iters * args.batch)

    print(f"tokens/image    : {tuple(tokens)}")
    print(f"throughput      : {1 / per_img:.1f} images/s  ({per_img * 1000:.0f} ms/image)")
    for n, label in ((4205, "6 categories, one pass"), (17000, "train x4 rotations + test")):
        print(f"est. {label:<28}: {n * per_img / 60:.1f} min")


if __name__ == "__main__":
    main()
