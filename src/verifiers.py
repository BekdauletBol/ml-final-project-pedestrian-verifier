"""The three verifier models behind one small interface (used by the web app).

Every verifier gets the crops of the YOLOv8n boxes (N x 3 x 128 x 128, uint8,
made like the training crops) and returns the predicted class and P(pedestrian).
The SVM has no probability, so for it P(pedestrian) is NaN and only the
predicted class is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

from .classical import hog_features
from .data import CLASSES, IMAGENET_MEAN, IMAGENET_STD, pick_device, prepare_batch
from .models import PedCNN, mobilenet_v2_transfer
from .verified_detector import letterbox_crop

ROOT = Path(__file__).resolve().parents[1]


class CNNVerifier:
    def __init__(self, net: torch.nn.Module, image_size: int, mean, std, device: torch.device):
        self.net, self.size, self.mean, self.std, self.device = net.to(device).eval(), image_size, mean, std, device

    @torch.no_grad()
    def predict(self, crops: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if len(crops) == 0:
            return np.zeros(0, int), np.zeros(0)
        x = prepare_batch(crops, self.device, self.size, self.mean, self.std)
        p = torch.softmax(self.net(x), 1).cpu().numpy()
        return p.argmax(1), p[:, 0]


class SVMVerifier:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, crops: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if len(crops) == 0:
            return np.zeros(0, int), np.zeros(0)
        pred = self.pipeline.predict(hog_features(crops, n_jobs=1))
        return pred.astype(int), np.full(len(pred), np.nan)


def load_verifier(name: str, device: torch.device | None = None):
    """name: 'MobileNetV2', 'Own CNN (improved)' or 'HOG + SVM'."""
    device = device or pick_device()
    if name == "MobileNetV2":
        net = mobilenet_v2_transfer(pretrained=False)
        net.load_state_dict(torch.load(ROOT / "models/mobilenetv2_pedcrops.pt", map_location="cpu"))
        return CNNVerifier(net, 128, IMAGENET_MEAN, IMAGENET_STD, device)
    if name == "Own CNN (improved)":
        stats = json.loads((ROOT / "data/pedcrops_v1/stats.json").read_text())["normalization"]
        net = PedCNN(activation="relu", hidden_size=64, hidden_layers=1, batchnorm=True, dropout=0.3)
        net.load_state_dict(torch.load(ROOT / "models/pedcnn_improved.pt", map_location="cpu"))
        return CNNVerifier(net, 64, tuple(stats["mean"]), tuple(stats["std"]), device)
    if name == "HOG + SVM":
        return SVMVerifier(joblib.load(ROOT / "models/hog_svm.joblib"))
    raise ValueError(name)


def crops_from_boxes(img_rgb: np.ndarray, boxes: np.ndarray) -> torch.Tensor:
    """Cut every box out of the photo exactly like the dataset crops (10 % context, gray letterbox)."""
    if len(boxes) == 0:
        return torch.zeros((0, 3, 128, 128), dtype=torch.uint8)
    img = Image.fromarray(img_rgb)
    crops = np.stack([np.asarray(letterbox_crop(img, b)) for b in boxes])
    return torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()


__all__ = ["CLASSES", "CNNVerifier", "SVMVerifier", "load_verifier", "crops_from_boxes"]
