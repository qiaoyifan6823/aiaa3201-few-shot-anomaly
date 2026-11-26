"""Interactive demo: pick a sample (or upload your own image) -> score, verdict, heatmap.

    sh demo/run.sh          # then open http://127.0.0.1:7860

The detector is the fused Part-3 model (blocks 6/9/12 at 448 + blocks 6/12 at
672, distances averaged). The OK/NG threshold is the 95th percentile of the 20
held-out normal images of the selected category, as required by the brief.
"""

from __future__ import annotations

import os
import sys

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from infer import Detector, overlay  # noqa: E402

CATEGORIES = ["chewinggum", "candle", "fryum"]
SAMPLES = [
    ("① 正常口香糖 | normal", "chewinggum", "samples/chewinggum_ok.jpg"),
    ("② 缺陷口香糖 | defect", "chewinggum", "samples/chewinggum_ng.jpg"),
    ("③ 蜡烛缺角 | defect", "candle", "samples/candle_ng.jpg"),
    ("④ 脆片裂纹 | defect", "fryum", "samples/fryum_ng.jpg"),
]
_detectors: dict[str, Detector] = {}


def get_detector(category: str) -> Detector:
    if category not in _detectors:
        _detectors[category] = Detector(category)
    return _detectors[category]


def run(image_path, category: str):
    """Score one image; returns (score, verdict, overlay figure)."""
    if not image_path:
        return None, "请先上传一张图片 / please upload an image", None
    if not category:
        return None, "请选择类别 / please pick a category", None
    det = get_detector(category)
    score, heat, info = det.score(image_path)
    out_path = os.path.join(HERE, "out", f"ui_{category}_{os.path.basename(image_path)}")
    overlay(image_path, heat, out_path, score, info)
    verdict = (f"{info['decision']}    score {score:.4f}  vs  threshold "
               f"{info['threshold']:.4f}   (margin {info['margin']:+.4f})")
    return round(score, 4), verdict, out_path


def run_sample(category: str, relative_path: str):
    return run(os.path.join(HERE, relative_path), category)


with gr.Blocks(title="Few-Shot Industrial Visual Anomaly Detection") as demo:
    gr.Markdown(
        "# Few-Shot Industrial Visual Anomaly Detection\n"
        "**Group 31 · AIAA 3201** — DINOv2 ViT-S/14, blocks 6/9/12 at 448 + blocks 6/12 at 672, "
        "distance maps averaged. Score ≥ calibrated threshold → **NG**."
    )

    with gr.Row():
        with gr.Column(scale=3):
            image_in = gr.Image(type="filepath", label="产品图片 / Product image")
            category_in = gr.Dropdown(CATEGORIES, value="chewinggum", label="类别 / Category")
            with gr.Row():
                clear_btn = gr.Button("Clear", scale=1)
                submit_btn = gr.Button("Submit", variant="primary", scale=2)
            gr.Markdown("**一键示例 / One-click samples**（无需上传，点一下即出结果）")
            sample_buttons = []
            for label, cat, rel in SAMPLES:
                sample_buttons.append((gr.Button(label), cat, rel))

        with gr.Column(scale=4):
            score_out = gr.Number(label="异常分数 / Anomaly score (top-1% mean distance)")
            verdict_out = gr.Textbox(label="判定 / Decision")
            heat_out = gr.Image(label="热力图 / Pixel-level heatmap")

    submit_btn.click(run, [image_in, category_in], [score_out, verdict_out, heat_out])
    clear_btn.click(lambda: (None, None, None, None),
                    None, [image_in, score_out, verdict_out, heat_out])
    for btn, cat, rel in sample_buttons:
        btn.click(lambda c=cat, r=rel: run_sample(c, r), None, [score_out, verdict_out, heat_out])

    demo.load(None, None, None)

if __name__ == "__main__":
    # load the three reference banks once so the first click is instant
    print("预热模型 / warming up detectors ...")
    for c in CATEGORIES:
        get_detector(c)
    print("就绪 / ready")
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("GRADIO_PORT", 7860)),
                show_error=True)
