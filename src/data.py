"""Loading the PedCrops dataset and doing augmentation on the GPU.

The dataset is small (about 15k crops of 128x128), so I load all of it into
memory as uint8 tensors once. Then every batch is moved to the device,
augmented there (fast on MPS / CUDA) and normalized.
"""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

CLASSES = ["pedestrian", "other_object", "background"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pick_device() -> torch.device:
    """Same order as in my diploma project: Apple GPU, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(data_dir: str | Path, split: str) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Read ``<split>.zip`` and the metadata rows of this split.

    Returns images as uint8 tensor (N, 3, H, W), labels as int64 tensor (N,)
    and the metadata DataFrame in the same order.
    """
    data_dir = Path(data_dir)
    meta = pd.read_csv(data_dir / "metadata.csv")
    meta = meta[meta["split"] == split].reset_index(drop=True)
    images = []
    with zipfile.ZipFile(data_dir / f"{split}.zip") as zf:
        for name in meta["file"]:
            with zf.open(name) as fh:
                img = Image.open(io.BytesIO(fh.read())).convert("RGB")
            images.append(np.asarray(img, dtype=np.uint8))
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(meta["label_id"].to_numpy(), dtype=torch.long)
    return x, y, meta


def iterate_batches(n: int, batch_size: int, shuffle: bool, generator: torch.Generator | None = None):
    """Yield index tensors for mini-batches."""
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def gpu_augment(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Random flip, rotation, scaling, shift and brightness/contrast for a float batch in [0, 1].

    Every image in the batch gets its own random parameters. ``strength`` scales
    all ranges (0 = no augmentation).
    """
    if strength <= 0:
        return x
    n = x.shape[0]
    dev = x.device
    # horizontal flip (people look the same mirrored)
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    # random affine: rotation +-15 deg, scale 0.85..1.15, shift +-10 %
    angle = (torch.rand(n, device=dev) * 2 - 1) * math.radians(15) * strength
    scale = 1 + (torch.rand(n, device=dev) * 2 - 1) * 0.15 * strength
    tx = (torch.rand(n, device=dev) * 2 - 1) * 0.10 * strength
    ty = (torch.rand(n, device=dev) * 2 - 1) * 0.10 * strength
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack(
        [torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # brightness and contrast jitter +-20 %
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.2 * strength
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.2 * strength
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * c + mean) * b
    return x.clamp(0, 1)


def prepare_batch(
    x_uint8: torch.Tensor,
    device: torch.device,
    size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    augment: float = 0.0,
) -> torch.Tensor:
    """uint8 batch -> resized, (optionally) augmented, normalized float batch on ``device``."""
    x = x_uint8.to(device, non_blocking=True).float() / 255.0
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    x = gpu_augment(x, augment)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s
