"""Interactive demo: upload a product image, get a score, a verdict and a heatmap.

    PYTHONPATH=.pylibs python3 demo/app.py
    # then open http://127.0.0.1:7860

The detector is the fused Part-3 model; the OK/NG threshold comes from the
20 held-out normal images of the selected category (see demo/build_assets.py).
"""

from __future__ import annotations

import os
import sys

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from infer import Detector, overlay  # noqa: E402

CATEGORIES = ["chewinggum", "candle", "fryum"]
_detectors: dict[str, Detector] = {}


def get_detector(category: str) -> Detector:
    if category not in _detectors:
        _detectors[category] = Detector(category)
    return _detectors[category]


def run(image_path: str | None, category: str):
    if not image_path:
        return None, "请先上传一张图片 / please upload an image", None
    det = get_detector(category)
    score, heat, info = det.score(image_path)
    out_path = os.path.join(HERE, "out", f"ui_{category}_{os.path.basename(image_path)}")
    overlay(image_path, heat, out_path, score, info)
    verdict = (f"{info['decision']}   (score {score:.4f} vs threshold "
               f"{info['threshold']:.4f}, margin {info['margin']:+.4f})")
    return round(score, 4), verdict, out_path


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="filepath", label="产品图片 / Product image"),
        gr.Dropdown(CATEGORIES, value="chewinggum", label="类别 / Category"),
    ],
    outputs=[
        gr.Number(label="异常分数 / Anomaly score (top-1% mean distance)"),
        gr.Textbox(label="判定 / Decision"),
        gr.Image(label="热力图 / Pixel-level heatmap"),
    ],
    title="Few-Shot Industrial Visual Anomaly Detection",
    description=("Group 31 · AIAA 3201. The model sees four defect-free references of the "
                 "selected category (DINOv2 ViT-S/14, blocks 6/9/12 at 448 + blocks 6/12 at 672, "
                 "fused). Score ≥ calibrated threshold → NG."),
    examples=[[p, c] for c, p in [
        ("chewinggum", "samples/chewinggum_ng.jpg"),
        ("chewinggum", "samples/chewinggum_ok.jpg"),
        ("candle", "samples/candle_ng.jpg"),
        ("fryum", "samples/fryum_ng.jpg"),
    ] if os.path.exists(os.path.join(HERE, p))],
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("GRADIO_PORT", 7860)),
                show_error=True)
