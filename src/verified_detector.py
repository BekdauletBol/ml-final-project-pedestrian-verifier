"""Practical application: YOLOv8n + my verifier, as a drop-in detector for public.safe.

public.safe (my diploma) calls ``detector.detect(image_bgr, conf)`` and gets a list of person
boxes (see ``backend/publicsafe/core/detector.py`` there). ``VerifiedDetector`` has the same
method: it asks YOLOv8n for boxes and then keeps only the boxes that the trained MobileNetV2
classifies as *pedestrian* with P >= threshold. So the zone rules and events of public.safe do
not need any change.

Command line demo (draws kept boxes green and rejected red):
    python -m src.verified_detector path/to/photo.jpg --out result.jpg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import IMAGENET_MEAN, IMAGENET_STD, pick_device, prepare_batch
from .models import mobilenet_v2_transfer

ROOT = Path(__file__).resolve().parents[1]
PAD_COLOR = (114, 114, 114)


@dataclass(frozen=True)
class Detection:
    """Same fields as ``publicsafe.core.models.Detection`` (box in pixels, confidence)."""

    bbox_xyxy: tuple[float, float, float, float]
    conf: float
    p_pedestrian: float = 1.0


def letterbox_crop(img: Image.Image, box, size: int = 128, context: float = 0.10) -> Image.Image:
    """The same crop as in the dataset: 10 % context, gray square padding, 128x128."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2, y1, y2 = x1 - context * w, x2 + context * w, y1 - context * h, y2 + context * h
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(img.width, int(round(x2))), min(img.height, int(round(y2)))
    crop = img.crop((x1, y1, x2, y2))
    side = max(crop.width, crop.height, 1)
    canvas = Image.new("RGB", (side, side), PAD_COLOR)
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


class VerifiedDetector:
    name = "yolo+verifier"

    def __init__(self, yolo_weights=ROOT / "models/yolov8n.pt",
                 verifier_weights=ROOT / "models/mobilenetv2_pedcrops.pt",
                 conf: float = 0.30, threshold: float = 0.5, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.device = torch.device(device) if device else pick_device()
        self.yolo = YOLO(str(yolo_weights))
        self.verifier = mobilenet_v2_transfer(pretrained=False)
        self.verifier.load_state_dict(torch.load(verifier_weights, map_location="cpu"))
        self.verifier.to(self.device).eval()
        self.conf, self.threshold = conf, threshold

    @torch.no_grad()
    def detect_all(self, image_bgr: np.ndarray, conf: float | None = None) -> tuple[list[Detection], list[Detection]]:
        """Returns (kept, rejected) person detections."""
        res = self.yolo.predict(image_bgr, conf=conf or self.conf, classes=[0], verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        if len(boxes) == 0:
            return [], []
        img = Image.fromarray(image_bgr[:, :, ::-1])
        crops = np.stack([np.asarray(letterbox_crop(img, b)) for b in boxes])
        x = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()
        x = prepare_batch(x, self.device, 128, IMAGENET_MEAN, IMAGENET_STD)
        p = torch.softmax(self.verifier(x), 1)[:, 0].cpu().numpy()
        kept, rejected = [], []
        for b, c, pp in zip(boxes, confs, p):
            d = Detection(tuple(float(v) for v in b), float(c), float(pp))
            (kept if pp >= self.threshold else rejected).append(d)
        return kept, rejected

    def detect(self, image_bgr: np.ndarray, conf: float | None = None) -> list[Detection]:
        """public.safe ``Detector`` protocol: only verified pedestrians."""
        return self.detect_all(image_bgr, conf)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("verified.jpg"))
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    det = VerifiedDetector(threshold=a.threshold)
    img = Image.open(a.image).convert("RGB")
    kept, rejected = det.detect_all(np.asarray(img)[:, :, ::-1].copy())
    draw = ImageDraw.Draw(img)
    for color, group in (("lime", kept), ("red", rejected)):
        for d in group:
            draw.rectangle(d.bbox_xyxy, outline=color, width=3)
            draw.text((d.bbox_xyxy[0] + 3, d.bbox_xyxy[1] + 3), f"{d.p_pedestrian:.2f}", fill=color)
    img.save(a.out)
    print(f"kept {len(kept)} pedestrians, rejected {len(rejected)} boxes -> {a.out}")


if __name__ == "__main__":
    main()
