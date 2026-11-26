"""Single-image inference: anomaly score, OK/NG decision, pixel-level heatmap.

    PYTHONPATH=.pylibs python3 demo/infer.py --image path/to/product.jpg --category chewinggum
    PYTHONPATH=.pylibs python3 demo/infer.py --image a.jpg --category candle --out demo/out

The model is the Part-3 fused model (blocks 6/9/12 at 448 + blocks 6/12 at 672);
the decision threshold is the one used in the paper (95th percentile of the 20
held-out normal images of that category), stored in demo/assets/.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from transformers import Dinov2Model

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

SIZE = 448
SCALES = {448: (6, 9, 12), 672: (6, 12)}


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]


class Detector:
    """Fused multi-layer / multi-scale patch nearest-neighbour detector."""

    def __init__(self, category: str, device: str = "auto", weights: str = "facebook/dinov2-small"):
        self.category = category
        self.asset = np.load(os.path.join(HERE, "assets", f"{category}.npz"), allow_pickle=False)
        self.threshold = float(self.asset["threshold"])
        self.device = pick_device(device)
        self.model = Dinov2Model.from_pretrained(weights).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def features(self, image_path: str) -> dict:
        """Patch features for every (scale, block) the model needs."""
        out = {}
        x = torch.from_numpy(load_image(image_path, max(SCALES))).to(self.device)
        for scale, layers in SCALES.items():
            # one forward per scale; the 448 input is a resize of the same image
            img = load_image(image_path, scale)
            xi = torch.from_numpy(img).to(self.device)
            hidden = self.model(pixel_values=xi, interpolate_pos_encoding=True,
                                output_hidden_states=True).hidden_states
            for b in layers:
                f = torch.nn.functional.normalize(hidden[b][:, 1:, :], dim=-1)
                out[(scale, b)] = f[0].float().cpu().numpy()
        del x
        return out

    def score(self, image_path: str) -> tuple[float, np.ndarray, dict]:
        feats = self.features(image_path)
        per_scale = []
        for scale, layers in SCALES.items():
            maps = []
            for b in layers:
                bank = self.asset[f"bank_{scale}_{b}"].reshape(-1, 384).astype(np.float32)
                sim = torch.from_numpy(feats[(scale, b)].astype(np.float32)) @ torch.from_numpy(bank).T
                d = (1.0 - sim.max(dim=1).values).numpy()
                grid = int(self.asset[f"grid_{scale}"])
                m = cv2.resize(d.reshape(grid, grid), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
                maps.append(cv2.GaussianBlur(m, (0, 0), sigmaX=4.0))
            per_scale.append(np.stack(maps, 0).mean(0))
        fused = np.stack(per_scale, 0).mean(0)
        k = max(1, int(round(fused.size * 0.01)))
        image_score = float(np.sort(fused.ravel())[-k:].mean())
        decision = "NG" if image_score >= self.threshold else "OK"
        info = {"threshold": self.threshold, "decision": decision,
                "margin": image_score - self.threshold}
        return image_score, fused, info


def overlay(image_path: str, heatmap: np.ndarray, out_path: str, score: float, info: dict) -> None:
    """Save a three-panel figure: input | heatmap | overlay with the verdict."""
    base = cv2.resize(cv2.imread(image_path), (SIZE, SIZE))
    m = heatmap - heatmap.min()
    m = m / (m.max() + 1e-8)
    heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(base, 0.55, heat, 0.45, 0)

    ok = info["decision"] == "OK"
    colour = (60, 160, 60) if ok else (40, 40, 200)
    panels = np.hstack([base, heat, blend])
    canvas = np.full((SIZE + 62, panels.shape[1], 3), 255, np.uint8)
    canvas[62:, :, :] = panels
    cv2.putText(canvas, f"score {score:.4f}   threshold {info['threshold']:.4f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, info["decision"], (panels.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, canvas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--category", required=True, choices=["chewinggum", "candle", "fryum"])
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    det = Detector(args.category, args.device)
    score, heat, info = det.score(args.image)
    print(f"category : {args.category}")
    print(f"score    : {score:.4f}")
    print(f"threshold: {info['threshold']:.4f}")
    print(f"decision : {info['decision']}  (margin {info['margin']:+.4f})")
    if not args.no_figure:
        name = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(args.out, f"{args.category}_{name}_result.png")
        overlay(args.image, heat, out_path, score, info)
        print(f"figure   : {out_path}")


if __name__ == "__main__":
    main()
