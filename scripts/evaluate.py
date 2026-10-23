"""Re-score the released checkpoint on the held-out SKU-110K test split, independently of training.

    python scripts/evaluate.py                       # weights/shelf_yolo11s.pt
    python scripts/evaluate.py path/to/weights.pt    # any checkpoint

Writes results/reproduced_test_metrics.json, so it can be compared with results/test_metrics.json.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sku110k_yolo" / "sku.yaml"


def main(weights: Path) -> None:
    started = time.time()
    result = YOLO(str(weights)).val(
        data=str(DATA), split="test", imgsz=832, batch=4, device=0 if torch.cuda.is_available() else "cpu",
        max_det=1000, workers=1, plots=False,
    )
    metrics = {
        "split": "test",
        "weights": weights.name,
        "map50": round(float(result.box.map50), 4),
        "map50_95": round(float(result.box.map), 4),
        "precision": round(float(result.box.mp), 4),
        "recall": round(float(result.box.mr), 4),
        "seconds": round(time.time() - started),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu",
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
    }
    (ROOT / "results" / "reproduced_test_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "weights" / "shelf_yolo11s.pt")
