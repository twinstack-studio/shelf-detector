"""Convert SKU-110K into a YOLO dataset and shrink images to at most 1024 px on the long side.

Input:  <src>/images/{train,val,test}/*.jpg and <src>/labels/{train,val,test}/*.txt
        (YOLO-normalised labels: class cx cy w h, one class "product").
Output: data/sku110k_yolo/{images,labels}/{train,val,test} and data/sku110k_yolo/sku.yaml

Labels are normalised, so resizing does not change them; they are copied as they are.

    python scripts/prepare_data.py path/to/SKU110K_yolo
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "data" / "sku110k_yolo"
MAX_SIDE = 1024


def main(src: Path) -> None:
    skipped = 0
    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
        files = sorted((src / "images" / split).glob("*.jpg"))
        for i, image_path in enumerate(files):
            label = src / "labels" / split / f"{image_path.stem}.txt"
            out = DST / "images" / split / image_path.name
            if not label.exists():
                skipped += 1
                continue
            if not out.exists():
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    skipped += 1
                    continue
                h, w = image.shape[:2]
                scale = MAX_SIDE / max(h, w)
                if scale < 1:
                    image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            shutil.copyfile(label, DST / "labels" / split / label.name)
            if i % 1000 == 0:
                print(split, i, "/", len(files), flush=True)
    (DST / "sku.yaml").write_text(
        f"path: {DST}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: product\n"
    )
    print("done, skipped", skipped)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/prepare_data.py path/to/SKU110K_yolo")
    main(Path(sys.argv[1]))
