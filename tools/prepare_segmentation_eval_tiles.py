#!/usr/bin/env python3
"""Prepare deterministic clean tiles for stitched segmentation inference.

Unlike augmented training crop generation, this utility never drops edge
regions and never applies augmentation. The final tile position along each axis
is aligned to the image boundary so every original pixel is covered. It is used
for metric computation on validation/test and for clean full-resolution
prediction export on train/validation/test qualitative subsets.

Input layout::

    <eval-root>/<split>/images/*.png
    <eval-root>/<split>/masks/*.png

Output layout::

    <out-root>/<split>/images/*.png
    <out-root>/<split>/masks/*.png
    <out-root>/<split>/tiles_manifest.csv
    <out-root>/<split>/tiles_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean full-coverage tiles for stitched segmentation inference.")
    parser.add_argument("--eval-root", type=Path, required=True,
                        help="Root containing <split>/images and <split>/masks clean PNG pairs.")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=448)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--padding-ignore-label", type=int, default=255,
                        help="Label assigned only to padded mask pixels when an image is smaller than one tile.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def axis_positions(length: int, tile_size: int, stride: int) -> List[int]:
    """Return tile origins with guaranteed complete coverage of one axis."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def pad_tile(image: np.ndarray, mask: np.ndarray, tile_size: int, ignore_label: int) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    pad_bottom = max(0, tile_size - height)
    pad_right = max(0, tile_size - width)
    if pad_bottom == 0 and pad_right == 0:
        return image, mask
    image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=ignore_label)
    return image, mask


def find_mask(image_path: Path, masks_dir: Path) -> Path:
    mask_path = masks_dir / image_path.name
    if not mask_path.is_file():
        raise FileNotFoundError("Mask not found for clean inference image: {}".format(image_path))
    return mask_path


def write_manifest(path: Path, fields: Sequence[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0 or args.stride <= 0:
        raise ValueError("tile-size and stride must be positive.")
    if not 0 <= args.padding_ignore_label <= 255:
        raise ValueError("padding-ignore-label must be in [0, 255].")

    input_root = args.eval_root / args.split
    input_images = input_root / "images"
    input_masks = input_root / "masks"
    if not input_images.is_dir() or not input_masks.is_dir():
        raise FileNotFoundError("Expected clean inference folders: {} and {}".format(input_images, input_masks))

    output_root = args.out_root / args.split
    if output_root.exists() and any(output_root.rglob("*")):
        if not args.overwrite:
            raise FileExistsError("Output split is non-empty: {}. Pass --overwrite.".format(output_root))
        shutil.rmtree(output_root)
    output_images = output_root / "images"
    output_masks = output_root / "masks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_images.glob("*.png"))
    if not image_paths:
        raise ValueError("No clean PNG inference images found in: {}".format(input_images))

    rows: List[Dict[str, object]] = []
    tiles_per_image: Dict[str, int] = {}
    min_coverage = None
    max_coverage = 0
    total_tiles = 0

    for image_path in tqdm(image_paths, desc="Preparing {} tiles".format(args.split)):
        mask_path = find_mask(image_path, input_masks)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError("Unreadable RGB/mask pair: {}, {}".format(image_path, mask_path))
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError("RGB/mask shape mismatch: {} vs {}".format(image_path.name, mask_path.name))

        height, width = image.shape[:2]
        x_positions = axis_positions(width, args.tile_size, args.stride)
        y_positions = axis_positions(height, args.tile_size, args.stride)
        coverage = np.zeros((height, width), dtype=np.uint16)
        image_tile_count = 0

        for y in y_positions:
            for x in x_positions:
                content_height = min(args.tile_size, height - y)
                content_width = min(args.tile_size, width - x)
                image_tile = image[y:y + content_height, x:x + content_width]
                mask_tile = mask[y:y + content_height, x:x + content_width]
                image_tile, mask_tile = pad_tile(image_tile, mask_tile, args.tile_size, args.padding_ignore_label)
                tile_stem = "{}_tile{:04d}_x{}_y{}".format(image_path.stem, image_tile_count, x, y)
                image_out = output_images / (tile_stem + ".png")
                mask_out = output_masks / (tile_stem + ".png")
                cv2.imwrite(str(image_out), image_tile)
                cv2.imwrite(str(mask_out), mask_tile)
                coverage[y:y + content_height, x:x + content_width] += 1
                rows.append({
                    "source_id": image_path.stem,
                    "source_image": str(image_path),
                    "source_mask": str(mask_path),
                    "original_height": height,
                    "original_width": width,
                    "tile_idx": image_tile_count,
                    "x": x,
                    "y": y,
                    "content_height": content_height,
                    "content_width": content_width,
                    "tile_size": args.tile_size,
                    "stride": args.stride,
                    "image_path": str(image_out),
                    "mask_path": str(mask_out),
                })
                image_tile_count += 1

        image_min_coverage = int(coverage.min())
        image_max_coverage = int(coverage.max())
        if image_min_coverage < 1:
            raise RuntimeError("Inference tiling left uncovered pixels in: {}".format(image_path))
        min_coverage = image_min_coverage if min_coverage is None else min(min_coverage, image_min_coverage)
        max_coverage = max(max_coverage, image_max_coverage)
        tiles_per_image[image_path.stem] = image_tile_count
        total_tiles += image_tile_count

    fields = [
        "source_id", "source_image", "source_mask", "original_height", "original_width",
        "tile_idx", "x", "y", "content_height", "content_width", "tile_size", "stride",
        "image_path", "mask_path",
    ]
    manifest_path = output_root / "tiles_manifest.csv"
    write_manifest(manifest_path, fields, rows)
    report = {
        "split": args.split,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "n_source_images": len(image_paths),
        "n_tiles": total_tiles,
        "tiles_per_image_min": min(tiles_per_image.values()),
        "tiles_per_image_max": max(tiles_per_image.values()),
        "tiles_per_image_mean": total_tiles / len(image_paths),
        "pixel_coverage_min": min_coverage,
        "pixel_coverage_max": max_coverage,
        "padding_ignore_label": args.padding_ignore_label,
        "notes": [
            "No image augmentation is applied to clean stitched-inference tiles.",
            "Overlapping tile logits must be stitched to each original image before prediction export or metrics.",
            "For the training subset these tiles are for qualitative clean inference only, not for model fitting.",
        ],
    }
    (output_root / "tiles_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Prepared clean {} tiles: {} from {} images".format(args.split, total_tiles, len(image_paths)))
    print("Tile manifest: {}".format(manifest_path))
    print("Coverage min/max: {}/{}".format(min_coverage, max_coverage))
    print("No augmentations were applied; stitch logits before using the outputs.")


if __name__ == "__main__":
    main()
