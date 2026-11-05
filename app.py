"""Web demo of my final project (for the defense).

YOLOv8n finds "person" boxes in a photo, and one of my three models checks every
box: green = kept as pedestrian, red = rejected (other object or background).

    python app.py            # then open http://127.0.0.1:7860
    python app.py --share    # public link, for example in Google Colab
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.data import CLASSES, pick_device  # noqa: E402
from src.verifiers import crops_from_boxes, load_verifier  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODELS = ["MobileNetV2", "Own CNN (improved)", "HOG + SVM", "No verifier (YOLOv8n only)"]
DEVICE = pick_device()
YOLO_MODEL = YOLO(str(ROOT / "models/yolov8n.pt"))
_verifiers: dict = {}


def get_verifier(name: str):
    if name not in _verifiers:
        _verifiers[name] = load_verifier(name, DEVICE)
    return _verifiers[name]


def draw(image: np.ndarray, boxes, keep, labels) -> Image.Image:
    img = Image.fromarray(image)
    d = ImageDraw.Draw(img)
    width = max(2, round(min(img.size) / 220))
    font = ImageFont.load_default(size=max(12, round(min(img.size) / 32)))
    for b, k, text in zip(boxes, keep, labels):
        color = (40, 200, 70) if k else (230, 50, 50)
        d.rectangle(tuple(b), outline=color, width=width)
        text_w = d.textlength(text, font=font)
        tx, ty = min(b[0], img.width - text_w - 2), max(0, b[1] - font.size - 4)  # keep the label inside
        d.rectangle(d.textbbox((tx, ty), text, font=font), fill=color)
        d.text((tx, ty), text, fill="white", font=font)
    return img


def analyze(image, model_name, threshold, yolo_conf):
    if image is None:
        return None, [], "Upload a photo or take one with the camera."
    t0 = time.perf_counter()
    res = YOLO_MODEL.predict(np.ascontiguousarray(image[:, :, ::-1]), conf=yolo_conf, classes=[0], verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    t1 = time.perf_counter()
    no_verifier = model_name.startswith("No verifier")
    if no_verifier:
        pred, p_ped = np.zeros(len(boxes), int), np.full(len(boxes), np.nan)
        keep = np.ones(len(boxes), bool)
    else:
        pred, p_ped = get_verifier(model_name).predict(crops_from_boxes(image, boxes))
        keep = pred == 0 if np.isnan(p_ped).any() else p_ped >= threshold
    t2 = time.perf_counter()

    labels, rows = [], []
    for i, (c, k, cls, p) in enumerate(zip(confs, keep, pred, p_ped), 1):
        cls_text = "–" if no_verifier else CLASSES[cls]
        p_text = "–" if np.isnan(p) else f"{p:.2f}"
        labels.append(f"{i} person {c:.2f}" if no_verifier else f"{i} {cls_text}" + ("" if np.isnan(p) else f" {p:.2f}"))
        rows.append([i, round(float(c), 2), cls_text, p_text, "kept" if k else "rejected"])
    summary = (
        f"**YOLOv8n found {len(boxes)} person box(es).** Verifier *{model_name}* kept **{int(keep.sum())}** "
        f"and rejected **{int((~keep).sum())}**.  \n"
        f"Time: YOLOv8n {1000 * (t1 - t0):.0f} ms, verifier {1000 * (t2 - t1):.0f} ms ({DEVICE.type})."
    )
    return draw(image, boxes, keep, labels), rows, summary


def build_ui() -> gr.Blocks:
    examples = sorted((ROOT / "app_examples").glob("*.jpg"))
    with gr.Blocks(title="Pedestrian verifier") as demo:
        gr.Markdown(
            "# Pedestrian verifier for YOLOv8n\n"
            "Final project, Machine Learning (AMO 3208) · Bolatov Bekdaulet, SE-2405\n\n"
            "YOLOv8n finds people, and my model checks every box. "
            "**Green** = kept as a pedestrian, **red** = rejected (other object or background)."
        )
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="numpy", sources=["upload", "webcam"], label="Photo")
                model = gr.Dropdown(MODELS, value=MODELS[0], label="Verifier model")
                threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05,
                                      label="Threshold on P(pedestrian) (CNN models only)")
                yolo_conf = gr.Slider(0.10, 0.90, value=0.30, step=0.05, label="YOLOv8n confidence")
                button = gr.Button("Check the photo", variant="primary")
            with gr.Column():
                result = gr.Image(type="pil", label="Result")
                summary = gr.Markdown()
        table = gr.Dataframe(headers=["box", "YOLO conf", "verifier class", "P(pedestrian)", "decision"],
                             label="All boxes", interactive=False)
        if examples:
            gr.Examples(examples=[[str(p)] for p in examples], inputs=[photo], label="Test photos (COCO val2017)")
        button.click(analyze, [photo, model, threshold, yolo_conf], [result, table, summary])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
