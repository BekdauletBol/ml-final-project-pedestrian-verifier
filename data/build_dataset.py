"""Build the PedCrops dataset from COCO val2017 (Assignment 2, Task 3).

Input : COCO val2017 as the two parquet shards of the Hugging Face mirror
        ``detection-datasets/coco`` (5,000 images with boxes and categories).
Output: ``<out>/train.zip``, ``val.zip``, ``test.zip`` with 128x128 JPEG crops
        and ``<out>/metadata.csv`` with one row per crop.

Classes
  0 pedestrian    - a COCO "person" box
  1 other_object  - a box of a street-type object (car, bicycle, bench, dog, ...)
                    that does not overlap a person
  2 background    - a random person-shaped window with no person in it

These are exactly the three things a YOLOv8n "person" box can contain in my
diploma system (public.safe): a real person, a confusing object, or nothing.

Usage:
    python build_dataset.py --parquet-dir raw_coco --out data/pedcrops_v1
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

CLASSES = ["pedestrian", "other_object", "background"]
CROP_SIZE = 128
PAD_COLOR = (114, 114, 114)  # the same gray YOLO uses for letterboxing
CONTEXT = 0.10  # add 10 % context around every box
SEED = 42

# COCO categories that appear in streets, kerbs and loading bays (taken first
# for the other_object class; the rest of the 79 non-person categories fill up)
STREET_OBJECTS = {
    "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "backpack", "umbrella", "handbag", "suitcase", "skateboard",
    "chair", "potted plant", "bottle", "kite", "sports ball", "clock",
}

# filters (rationale in the report)
MIN_PERSON_H = 48  # YOLOv8n rarely gives reliable boxes below this height
MIN_PERSON_W = 16
MIN_OBJECT_SIDE = 32
MAX_IOU_WITH_PERSON = 0.10  # negatives must not contain a person
MAX_PERSON_COVER = 0.20  # share of the crop covered by person boxes
BG_PER_IMAGE = 3


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box ``a`` (x1, y1, x2, y2) with many boxes ``b``."""
    if len(b) == 0:
        return np.zeros(0)
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def cover(a: np.ndarray, b: np.ndarray) -> float:
    """Share of box ``a`` that is covered by the boxes ``b`` (upper bound: sum of intersections)."""
    if len(b) == 0:
        return 0.0
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return float(min(1.0, inter.sum() / ((a[2] - a[0]) * (a[3] - a[1]) + 1e-9)))


def letterbox_crop(img: Image.Image, box, size: int = CROP_SIZE, context: float = CONTEXT) -> Image.Image:
    """Crop ``box`` with some context, pad to a square with gray and resize.

    Padding (not stretching) keeps the tall shape of a person, which is an
    important cue. YOLO pads the same way.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - context * w, x2 + context * w
    y1, y2 = y1 - context * h, y2 + context * h
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(img.width, int(round(x2))), min(img.height, int(round(y2)))
    crop = img.crop((x1, y1, x2, y2))
    side = max(crop.width, crop.height)
    canvas = Image.new("RGB", (side, side), PAD_COLOR)
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


def ahash(im: Image.Image) -> str:
    """Average hash of a 16x16 gray thumbnail; equal hashes = near-duplicate crops."""
    a = np.asarray(im.convert("L").resize((16, 16)), dtype=np.float32)
    return hashlib.md5((a > a.mean()).tobytes()).hexdigest()


def read_coco_parquet(parquet_dir: Path):
    """Yield (image_id, PIL image, boxes xyxy (N,4), category names) for every image."""
    files = sorted(parquet_dir.glob("val-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no val-*.parquet in {parquet_dir}")
    names = COCO80
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                objs = row["objects"]
                boxes = np.asarray(objs["bbox"], dtype=np.float32).reshape(-1, 4)
                cats = [names[c] for c in objs["category"]]
                raw = row["image"]["bytes"]
                yield row["image_id"], raw, boxes, cats


# category ids 0..79 of the Hugging Face mirror (taken from its dataset_infos.json)
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def build(parquet_dir: Path, out: Path, verbose: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out.mkdir(parents=True, exist_ok=True)
    rows, crops = [], {}
    seen_image_hashes: set[str] = set()
    stats = {"images": 0, "corrupted": 0, "duplicates": 0, "no_annotations": 0, "gray_converted": 0,
             "person_boxes": 0, "person_too_small": 0, "object_boxes": 0, "object_overlap_person": 0,
             "object_too_small": 0, "bg_rejected": 0}

    for image_id, raw, boxes, cats in read_coco_parquet(parquet_dir):
        stats["images"] += 1
        digest = hashlib.md5(raw).hexdigest()
        if digest in seen_image_hashes:
            stats["duplicates"] += 1
            continue
        seen_image_hashes.add(digest)
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except Exception:  # broken JPEG
            stats["corrupted"] += 1
            continue
        if img.mode != "RGB":
            stats["gray_converted"] += 1
            img = img.convert("RGB")
        if len(boxes) == 0:
            stats["no_annotations"] += 1
        W, H = img.size
        # clip boxes to the image and drop degenerate ones
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        persons = np.array([b for b, c in zip(boxes, cats) if c == "person"]).reshape(-1, 4)

        def add(label: str, box, source: str) -> None:
            crop_id = f"{image_id:012d}_{len(rows):06d}"
            x1, y1, x2, y2 = (float(v) for v in box)
            crop = letterbox_crop(img, (x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, "JPEG", quality=85)
            crops[crop_id] = buf.getvalue()  # keep JPEG bytes, not pixels, to save memory
            rows.append({
                "ahash": ahash(crop),
                "crop_id": crop_id, "image_id": int(image_id), "label": label,
                "label_id": CLASSES.index(label), "source": source,
                "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                "box_w": round(x2 - x1, 1), "box_h": round(y2 - y1, 1), "img_w": W, "img_h": H,
                "touches_border": bool(x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1),
            })

        # 1) pedestrians
        for b in persons:
            stats["person_boxes"] += 1
            w, h = b[2] - b[0], b[3] - b[1]
            if h < MIN_PERSON_H or w < MIN_PERSON_W:
                stats["person_too_small"] += 1
                continue
            add("pedestrian", b, "person")
        # 2) other objects that look like possible false positives
        for b, c in zip(boxes, cats):
            if c == "person":
                continue
            stats["object_boxes"] += 1
            if min(b[2] - b[0], b[3] - b[1]) < MIN_OBJECT_SIDE:
                stats["object_too_small"] += 1
                continue
            if (iou(b, persons) > MAX_IOU_WITH_PERSON).any() or cover(b, persons) > MAX_PERSON_COVER:
                stats["object_overlap_person"] += 1
                continue
            add("other_object", b, c)
        # 3) background windows with a person-like shape
        made, tries = 0, 0
        while made < BG_PER_IMAGE and tries < 40:
            tries += 1
            h = rng.uniform(MIN_PERSON_H, 0.8 * H)
            w = h * rng.uniform(0.3, 0.7)
            if w >= W:
                continue
            x1, y1 = rng.uniform(0, W - w), rng.uniform(0, H - h)
            b = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)
            if (iou(b, persons) > 0.05).any() or cover(b, persons) > 0.10 or (iou(b, boxes) > 0.30).any():
                stats["bg_rejected"] += 1
                continue
            add("background", b, "random_window")
            made += 1

    meta = pd.DataFrame(rows)
    if verbose:
        print("raw counts:", meta["label"].value_counts().to_dict())
        print("cleaning stats:", stats)

    # near-duplicate crops (identical 16x16 gray thumbnails) are removed
    before = len(meta)
    meta = meta.drop_duplicates("ahash").reset_index(drop=True)
    stats["duplicate_crops"] = before - len(meta)

    # balance the classes: every class gets as many crops as the smallest one.
    # For other_object the street-type categories are taken first, then the rest.
    n = int(meta["label"].value_counts().min())
    meta["priority"] = rng.random(len(meta)) + meta["source"].isin(STREET_OBJECTS).astype(float)
    meta = (meta.sort_values("priority", ascending=False)
                .groupby("label").head(n)
                .drop(columns=["priority"])
                .reset_index(drop=True))

    # split BY IMAGE (70/15/15) so crops of one photo never land in two splits
    image_ids = np.array(sorted(meta["image_id"].unique()))
    rng.shuffle(image_ids)
    n_tr, n_va = int(0.70 * len(image_ids)), int(0.15 * len(image_ids))
    split_of = {i: "train" for i in image_ids[:n_tr]}
    split_of.update({i: "val" for i in image_ids[n_tr:n_tr + n_va]})
    split_of.update({i: "test" for i in image_ids[n_tr + n_va:]})
    meta["split"] = meta["image_id"].map(split_of)
    meta["file"] = meta["label"] + "/" + meta["crop_id"] + ".jpg"
    meta = meta.sort_values(["split", "label", "crop_id"]).reset_index(drop=True)

    for split in ["train", "val", "test"]:
        part = meta[meta["split"] == split]
        with zipfile.ZipFile(out / f"{split}.zip", "w", zipfile.ZIP_STORED) as zf:
            for crop_id, name in zip(part["crop_id"], part["file"]):
                zf.writestr(name, crops[crop_id])

    # mean / std of the training crops (used to normalize the custom CNN input)
    tr = meta[meta["split"] == "train"]
    sample = tr.sample(n=min(2000, len(tr)), random_state=SEED)["crop_id"]
    px = np.stack([np.asarray(Image.open(io.BytesIO(crops[c])), dtype=np.float32) / 255.0 for c in sample])
    norm = {"mean": [round(float(v), 4) for v in px.mean(axis=(0, 1, 2), dtype=np.float64)],
            "std": [round(float(v), 4) for v in px.std(axis=(0, 1, 2), dtype=np.float64)]}

    meta = meta.drop(columns=["ahash"])
    meta.to_csv(out / "metadata.csv", index=False)
    (out / "stats.json").write_text(json.dumps({"cleaning": stats, "normalization": norm,
                                                "classes": CLASSES, "crop_size": CROP_SIZE}, indent=2))
    if verbose:
        print("final:", meta.groupby(["split", "label"]).size().unstack().to_string())
        print("normalization:", norm)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/pedcrops_v1"))
    args = ap.parse_args()
    build(args.parquet_dir, args.out)
