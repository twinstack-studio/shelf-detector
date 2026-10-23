# Shelf product detector — YOLO11s fine-tuned on SKU-110K

A dense retail-shelf product detector: YOLO11s (COCO-pretrained) fine-tuned on
[SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) as a single class, `product`.
It is the detector behind the shelf-gap and planogram features of
[ShelfSense](https://shelfsense.twinstackstudio.com), a retail analytics demo by TwinStack Studio.

## Results (held-out test split, 2,936 images)

| Metric | Score |
|---|---|
| mAP@0.5 | **0.941** |
| mAP@0.5:0.95 | **0.595** |
| Precision | 0.914 |
| Recall | 0.879 |

- `results/test_metrics.json` — the test-split scores written by `scripts/train.py` after training.
- `results/results.csv` — per-epoch training and **validation** curves (40 epochs; best val mAP@0.5:0.95 0.589 at epoch 32).
- `results/args.yaml` — every training setting of this run.
- `results/plots/` — PR / F1 curves, confusion matrix, validation predictions next to ground truth.
- `weights/shelf_yolo11s.pt` — the checkpoint (19 MB).

The validation split was used for model selection (best epoch, early stopping). The test split was
never seen during training or selection; it is scored once, at the end.

## Before and after fine-tuning (same test split, 2,936 images, 431,546 boxes)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| YOLO11s COCO-pretrained, no fine-tuning (every COCO class counted as `product`) | 0.384 | 0.164 | 0.284 | 0.600 |
| **YOLO11s fine-tuned on SKU-110K (this repo)** | **0.941** | **0.595** | **0.914** | **0.879** |

- `results/baseline_coco_test_metrics.json` — the "before" row.
- `results/reproduced_test_metrics.json` — the "after" row re-scored with `scripts/evaluate.py`, independently of training (same numbers).

## Training run

| | |
|---|---|
| Base model | `yolo11s.pt` (Ultralytics, COCO-pretrained) |
| Data | SKU-110K train 8,219 / val 588 / test 2,936 images, images shrunk to ≤1024 px |
| Input size | 832 px |
| Epochs | 40 (cosine LR, mosaic off for the last 8, `max_det=1000` for dense shelves) |
| Batch | 4 |
| Hardware | 1× RTX 2080 Ti (11 GB), one CPU core for data loading (shared server) |
| Time | 3.8 GPU-hours |

## Reproduce

```bash
pip install -r requirements.txt
# SKU-110K converted to YOLO format (images/ + labels/ per split):
python scripts/prepare_data.py path/to/SKU110K_yolo
python scripts/evaluate.py          # score the released checkpoint on the test split
python scripts/train.py             # or train from scratch (~4 GPU-hours on a 2080 Ti)
```

The dataset is not included; download it from the SKU-110K authors and follow their terms of use.

---

TwinStack Studio · Wasil Nadeem
