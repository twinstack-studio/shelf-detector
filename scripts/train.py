"""Fine-tune YOLO11s (COCO-pretrained) on SKU-110K as a single "product" class, then score the held-out test split.

The exact settings of the published run are in results/args.yaml.

    python scripts/train.py            # trains into runs/shelf, writes results/test_metrics.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sku110k_yolo" / "sku.yaml"


def main() -> None:
    model = YOLO("yolo11s.pt")
    model.train(
        data=str(DATA), imgsz=832, epochs=40, batch=4, workers=1, device=0, amp=True,
        project=str(ROOT / "runs"), name="shelf", exist_ok=True, patience=12, cos_lr=True,
        close_mosaic=8, max_det=1000, plots=True, seed=7,
    )
    best = ROOT / "runs" / "shelf" / "weights" / "best.pt"
    shutil.copyfile(best, ROOT / "weights" / "shelf_yolo11s.pt")

    # Model selection used the val split; the test split is touched only here.
    result = YOLO(str(best)).val(data=str(DATA), split="test", imgsz=832, batch=4, device=0, max_det=1000)
    metrics = {
        "split": "test",
        "images": 2936,
        "map50": round(float(result.box.map50), 4),
        "map50_95": round(float(result.box.map), 4),
        "precision": round(float(result.box.mp), 4),
        "recall": round(float(result.box.mr), 4),
        "imgsz": 832,
        "model": "YOLO11s",
    }
    (ROOT / "results" / "test_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
