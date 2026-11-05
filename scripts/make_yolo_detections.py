"""Run YOLOv8n (the detector of my diploma system) on the TEST photos and save its person boxes.

Every detection is compared with the COCO ground truth:
  IoU >= 0.5 with a real person  -> true positive (TP)
  otherwise                      -> false positive (FP) = a possible false alarm
Each box is cropped exactly like the training crops (10 % context, gray letterbox, 128x128),
so the trained verifier can be tested on REAL detector mistakes.

Output: data/yolo_test_detections/{crops.zip, detections.csv, summary.json, demo/*.jpg}
Usage:  python scripts/make_yolo_detections.py --parquet-dir <folder with the 2 COCO val parquet files>
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import build_dataset as bd  # noqa: E402

CONF = 0.30  # the default confidence of public.safe
IOU_TP = 0.5


def main(parquet_dir: Path, out: Path, n_demo: int = 8) -> None:
    from ultralytics import YOLO

    meta = pd.read_csv(ROOT / "data/pedcrops_v1/metadata.csv")
    test_ids = set(meta.loc[meta.split == "test", "image_id"])
    model = YOLO(str(ROOT / "models/yolov8n.pt"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo").mkdir(exist_ok=True)
    rows, gt_total, n_img = [], 0, 0
    zf = zipfile.ZipFile(out / "crops.zip", "w", zipfile.ZIP_STORED)
    for f in sorted(parquet_dir.glob("val-*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                if row["image_id"] not in test_ids:
                    continue
                n_img += 1
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                gt = np.array([b for c, b in zip(row["objects"]["category"], row["objects"]["bbox"]) if c == 0],
                              dtype=np.float32).reshape(-1, 4)
                gt_total += len(gt)
                res = model.predict(np.asarray(img)[:, :, ::-1], conf=CONF, classes=[0], imgsz=640,
                                    verbose=False)[0]
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                used = set()
                for k in np.argsort(-confs):  # greedy matching, best confidence first
                    b = boxes[k]
                    ious = bd.iou(b, gt)
                    j = int(ious.argmax()) if len(ious) else -1
                    tp = j >= 0 and ious[j] >= IOU_TP and j not in used
                    if tp:
                        used.add(j)
                    det_id = f"{row['image_id']:012d}_{k:03d}"
                    buf = io.BytesIO()
                    bd.letterbox_crop(img, b).save(buf, "JPEG", quality=90)
                    zf.writestr(f"{det_id}.jpg", buf.getvalue())
                    rows.append({"det_id": det_id, "image_id": row["image_id"], "conf": float(confs[k]),
                                 "x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                                 "box_h": float(b[3] - b[1]), "max_iou_gt": float(ious.max()) if len(ious) else 0.0,
                                 "is_tp": bool(tp)})
                if n_img <= 400 and len(boxes) and len(list((out / "demo").glob("*.jpg"))) < n_demo:
                    img.save(out / "demo" / f"{row['image_id']:012d}.jpg", quality=90)
    zf.close()
    det = pd.DataFrame(rows)
    det.to_csv(out / "detections.csv", index=False)
    summary = {"test_images": n_img, "gt_persons": gt_total, "detections": len(det),
               "tp": int(det.is_tp.sum()), "fp": int((~det.is_tp).sum()), "conf": CONF, "iou_tp": IOU_TP,
               "precision": float(det.is_tp.mean()), "recall": float(det.is_tp.sum() / gt_total)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data/yolo_test_detections")
    a = ap.parse_args()
    main(a.parquet_dir, a.out)
