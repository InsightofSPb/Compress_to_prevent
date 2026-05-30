#!/usr/bin/env python3
"""Prepare clean and weakly augmented *train-only* tiles for facade segmentation.

This utility is intentionally separate from temporal RGB compression. It reads
only the supervised segmentation training split (RGB inputs plus target masks),
keeps an unmodified tile for every crop, and adds mild augmented copies. Val and
test images are not touched and should be evaluated from the raw split folders.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create clean plus weakly augmented train tiles for segmentation.")
    p.add_argument("--train-root", type=Path, required=True,
                   help="Folder with images/ and masks/ for the original train split.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=448)
    p.add_argument("--stride", type=int, default=224)
    p.add_argument("--min-content-ratio", type=float, default=0.60)
    p.add_argument("--aug-copies", type=int, default=2,
                   help="Number of weakly augmented copies in addition to every clean tile.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    return p.parse_args()


def find_mask(image: Path, masks_dir: Path) -> Path:
    direct = masks_dir / image.name
    if direct.exists():
        return direct
    matches = [p for p in masks_dir.glob(f"{image.stem}.*") if p.suffix.lower() in IMAGE_EXTS]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Exactly one mask is required for {image.name}; found: {matches}")


def pad(array: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    h, w = array.shape[:2]
    bottom, right = max(0, size - h), max(0, size - w)
    if bottom == 0 and right == 0:
        return array
    border = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101
    return cv2.copyMakeBorder(array, 0, bottom, 0, right, border, value=0)


def tiles(image: np.ndarray, mask: np.ndarray, size: int, stride: int,
          min_content_ratio: float) -> Iterator[Tuple[int, int, int, np.ndarray, np.ndarray]]:
    height, width = image.shape[:2]
    idx = 0
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            content_h = min(size, height - y)
            content_w = min(size, width - x)
            if (content_h * content_w) / float(size * size) < min_content_ratio:
                continue
            image_tile = pad(image[y:y + size, x:x + size], size, is_mask=False)
            mask_tile = pad(mask[y:y + size, x:x + size], size, is_mask=True)
            yield idx, x, y, image_tile, mask_tile
            idx += 1


def weak_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.20),
        A.ShiftScaleRotate(
            shift_limit=0.025, scale_limit=0.05, rotate_limit=5,
            border_mode=cv2.BORDER_REFLECT_101, p=0.25,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.25),
        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=5, val_shift_limit=5, p=0.10),
        A.GaussNoise(var_limit=(2.0, 8.0), p=0.05),
    ])


def save_pair(images_dir: Path, masks_dir: Path, stem: str,
              image: np.ndarray, mask: np.ndarray) -> Tuple[Path, Path]:
    image_path = images_dir / f"{stem}.png"
    mask_path = masks_dir / f"{stem}.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    return image_path, mask_path


def save_overlay(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    visible = image.copy()
    foreground = mask > 0
    if visible.ndim == 3:
        color = np.zeros_like(visible)
        color[..., 2] = 255
        visible[foreground] = (0.55 * visible[foreground] + 0.45 * color[foreground]).astype(np.uint8)
    cv2.imwrite(str(path), visible)


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0 or args.stride <= 0 or args.aug_copies < 0:
        raise ValueError("tile-size and stride must be positive; aug-copies must be non-negative.")
    images_src = args.train_root / "images"
    masks_src = args.train_root / "masks"
    if not images_src.is_dir() or not masks_src.is_dir():
        raise FileNotFoundError("Expected <train-root>/images and <train-root>/masks.")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.out_dir}. Pass --overwrite to reuse it.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_images = args.out_dir / "images"
    output_masks = args.out_dir / "masks"
    output_overlays = args.out_dir / "overlays"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        output_overlays.mkdir(parents=True, exist_ok=True)

    transform = weak_transform()
    images = sorted(p for p in images_src.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    records: List[Dict[str, object]] = []
    clean_count = aug_count = 0
    for image_path in tqdm(images, desc="Preparing train tiles"):
        mask_path = find_mask(image_path, masks_src)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError(f"Unreadable input pair: {image_path}, {mask_path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"RGB/mask shape mismatch: {image_path.name} vs {mask_path.name}")

        for tile_idx, x, y, clean_image, clean_mask in tiles(
            image, mask, args.tile_size, args.stride, args.min_content_ratio
        ):
            base = f"{image_path.stem}_tile{tile_idx:04d}_x{x}_y{y}"
            clean_stem = f"{base}_clean"
            out_image, out_mask = save_pair(output_images, output_masks, clean_stem, clean_image, clean_mask)
            if args.save_overlays:
                save_overlay(output_overlays / f"{clean_stem}.png", clean_image, clean_mask)
            records.append({"source_image": str(image_path), "source_mask": str(mask_path),
                            "tile_idx": tile_idx, "x": x, "y": y, "variant": "clean",
                            "image_path": str(out_image), "mask_path": str(out_mask)})
            clean_count += 1

            for aug_idx in range(args.aug_copies):
                augmented = transform(image=clean_image, mask=clean_mask)
                aug_image, aug_mask = augmented["image"], augmented["mask"]
                aug_stem = f"{base}_aug{aug_idx:02d}"
                out_image, out_mask = save_pair(output_images, output_masks, aug_stem, aug_image, aug_mask)
                if args.save_overlays:
                    save_overlay(output_overlays / f"{aug_stem}.png", aug_image, aug_mask)
                records.append({"source_image": str(image_path), "source_mask": str(mask_path),
                                "tile_idx": tile_idx, "x": x, "y": y, "variant": f"aug{aug_idx:02d}",
                                "image_path": str(out_image), "mask_path": str(out_mask)})
                aug_count += 1

    fields = ["source_image", "source_mask", "tile_idx", "x", "y", "variant", "image_path", "mask_path"]
    with (args.out_dir / "train_tiles_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Train input images: {len(images)}")
    print(f"Clean train tiles: {clean_count}")
    print(f"Weakly augmented train tiles: {aug_count}")
    print(f"Manifest: {args.out_dir / 'train_tiles_manifest.csv'}")
    print("Validation and test were not cropped or augmented.")


if __name__ == "__main__":
    main()
