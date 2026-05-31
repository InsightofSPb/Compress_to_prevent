#!/usr/bin/env python3
"""Export clean segmentation split samples as uniform PNG pairs.

The facade dataset contains mixed RGB extensions (.png/.jpg). MMSeg CustomDataset
configs in this repository use one fixed ``img_suffix``. This utility converts
images from a split manifest to RGB PNG and copies label maps as PNG without any
crop, resize, or augmentation. It is intended for clean stitched inference and
visualisation on train, validation, or test subsets.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clean RGB-mask split samples as PNG files.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [{str(key): (value or "") for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("Empty split manifest: {}".format(path))
    return rows


def main() -> None:
    args = parse_args()
    rows = read_rows(args.manifest_csv)
    split_root = args.out_root / args.split
    if split_root.exists() and any(split_root.rglob("*")):
        if not args.overwrite:
            raise FileExistsError("Output split is not empty: {}. Pass --overwrite.".format(split_root))
        shutil.rmtree(split_root)
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    seen_names = set()
    label_values = set()
    for row in rows:
        image_path = Path(row.get("image_path", ""))
        mask_path = Path(row.get("mask_path", ""))
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError("Missing RGB/mask pair: {}, {}".format(image_path, mask_path))
        stem = image_path.stem
        filename = stem + ".png"
        if filename in seen_names:
            raise ValueError("Duplicate destination filename after PNG conversion: {}".format(filename))
        seen_names.add(filename)

        with Image.open(image_path) as image:
            image.convert("RGB").save(images_dir / filename)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError("Could not read mask: {}".format(mask_path))
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        label_values.update(int(value) for value in set(mask.reshape(-1).tolist()))
        cv2.imwrite(str(masks_dir / filename), mask)

    report = {
        "manifest_csv": str(args.manifest_csv),
        "split": args.split,
        "n_samples": len(rows),
        "output_root": str(split_root),
        "operation": "format conversion only; no crop resize or augmentation",
        "mask_label_values": sorted(label_values),
    }
    (split_root / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Exported clean {} samples: {}".format(args.split, len(rows)))
    print("Images: {}".format(images_dir))
    print("Masks: {}".format(masks_dir))
    print("Mask labels: {}".format(sorted(label_values)))


if __name__ == "__main__":
    main()
