#!/usr/bin/env python3
"""Prepare clean and moderately augmented *train-only* tiles for facade segmentation.

This utility is intentionally separate from temporal RGB compression. It reads
only the supervised segmentation training split (RGB inputs plus target masks),
keeps an unmodified tile for every crop, and adds realistic augmented copies.
Validation and test images are not touched and must be evaluated from raw split
folders.

Every augmented copy is forced to contain a mild photometric transform. Optional
small geometric changes and weak noise/blur can be added on top. The output
manifest records mean image difference from the clean tile; ``--save-diffs``
creates amplified absolute-difference previews for quality control.
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
    p = argparse.ArgumentParser(description="Create clean plus moderate train-only tiles for segmentation.")
    p.add_argument("--train-root", type=Path, required=True,
                   help="Folder with images/ and masks/ for the original train split.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=448)
    p.add_argument("--stride", type=int, default=224)
    p.add_argument("--min-content-ratio", type=float, default=0.60)
    p.add_argument("--aug-copies", type=int, default=2,
                   help="Number of augmented copies in addition to every clean tile.")
    p.add_argument("--min-aug-diff", type=float, default=1.0,
                   help="Retry augmentation when mean absolute RGB difference from clean is below this value.")
    p.add_argument("--max-aug-tries", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--save-diffs", action="store_true",
                   help="Save amplified |aug-clean| previews for visual quality control.")
    return p.parse_args()


def find_mask(image: Path, masks_dir: Path) -> Path:
    direct = masks_dir / image.name
    if direct.exists():
        return direct
    matches = [p for p in masks_dir.glob("{}.*".format(image.stem)) if p.suffix.lower() in IMAGE_EXTS]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("Exactly one mask is required for {}; found: {}".format(image.name, matches))


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


def train_transform() -> A.Compose:
    """A realistic augmentation policy; every returned aug copy changes appearance."""
    return A.Compose([
        # Required appearance variation: illumination/camera response changes are
        # common in street-level temporal data, but the ranges remain moderate.
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.16, contrast_limit=0.16, p=1.0),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=10, p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        ], p=1.0),
        # Mild viewpoint/crop mismatch, with mask transformed jointly.
        A.HorizontalFlip(p=0.25),
        A.ShiftScaleRotate(
            shift_limit=0.035, scale_limit=0.075, rotate_limit=6,
            border_mode=cv2.BORDER_REFLECT_101, p=0.40,
        ),
        # Rare acquisition degradation; no artificial weather or occlusion effects.
        A.OneOf([
            A.GaussNoise(var_limit=(4.0, 16.0), p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ], p=0.12),
    ])


def augment_with_minimum_difference(transform: A.Compose, image: np.ndarray, mask: np.ndarray,
                                    min_diff: float, max_tries: int) -> Tuple[np.ndarray, np.ndarray, float]:
    best_image, best_mask, best_diff = image, mask, -1.0
    for _ in range(max(1, max_tries)):
        augmented = transform(image=image, mask=mask)
        aug_image, aug_mask = augmented["image"], augmented["mask"]
        diff = float(np.abs(aug_image.astype(np.float32) - image.astype(np.float32)).mean())
        if diff > best_diff:
            best_image, best_mask, best_diff = aug_image, aug_mask, diff
        if diff >= min_diff:
            break
    return best_image, best_mask, best_diff


def mask_change_ratio(clean: np.ndarray, augmented: np.ndarray) -> float:
    return float(np.mean(clean != augmented))


def save_pair(images_dir: Path, masks_dir: Path, stem: str,
              image: np.ndarray, mask: np.ndarray) -> Tuple[Path, Path]:
    image_path = images_dir / "{}.png".format(stem)
    mask_path = masks_dir / "{}.png".format(stem)
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


def save_diff_preview(path: Path, clean: np.ndarray, augmented: np.ndarray) -> None:
    diff = cv2.absdiff(clean, augmented).astype(np.float32)
    amplified = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), amplified)


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0 or args.stride <= 0 or args.aug_copies < 0 or args.max_aug_tries <= 0:
        raise ValueError("tile-size, stride and max-aug-tries must be positive; aug-copies must be non-negative.")
    images_src = args.train_root / "images"
    masks_src = args.train_root / "masks"
    if not images_src.is_dir() or not masks_src.is_dir():
        raise FileNotFoundError("Expected <train-root>/images and <train-root>/masks.")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("Output is non-empty: {}. Pass --overwrite to reuse it.".format(args.out_dir))

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_images = args.out_dir / "images"
    output_masks = args.out_dir / "masks"
    output_overlays = args.out_dir / "overlays"
    output_diffs = args.out_dir / "diffs"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        output_overlays.mkdir(parents=True, exist_ok=True)
    if args.save_diffs:
        output_diffs.mkdir(parents=True, exist_ok=True)

    transform = train_transform()
    images = sorted(p for p in images_src.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    records: List[Dict[str, object]] = []
    clean_count = aug_count = near_identical_count = 0
    aug_differences: List[float] = []
    for image_path in tqdm(images, desc="Preparing train tiles"):
        mask_path = find_mask(image_path, masks_src)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError("Unreadable input pair: {}, {}".format(image_path, mask_path))
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError("RGB/mask shape mismatch: {} vs {}".format(image_path.name, mask_path.name))

        for tile_idx, x, y, clean_image, clean_mask in tiles(
            image, mask, args.tile_size, args.stride, args.min_content_ratio
        ):
            base = "{}_tile{:04d}_x{}_y{}".format(image_path.stem, tile_idx, x, y)
            clean_stem = "{}_clean".format(base)
            out_image, out_mask = save_pair(output_images, output_masks, clean_stem, clean_image, clean_mask)
            if args.save_overlays:
                save_overlay(output_overlays / "{}.png".format(clean_stem), clean_image, clean_mask)
            records.append({"source_image": str(image_path), "source_mask": str(mask_path),
                            "tile_idx": tile_idx, "x": x, "y": y, "variant": "clean",
                            "mean_abs_image_diff": "0.000000", "mask_change_ratio": "0.000000",
                            "image_path": str(out_image), "mask_path": str(out_mask)})
            clean_count += 1

            for aug_idx in range(args.aug_copies):
                aug_image, aug_mask, mean_diff = augment_with_minimum_difference(
                    transform, clean_image, clean_mask, args.min_aug_diff, args.max_aug_tries
                )
                aug_stem = "{}_aug{:02d}".format(base, aug_idx)
                out_image, out_mask = save_pair(output_images, output_masks, aug_stem, aug_image, aug_mask)
                changed_mask_ratio = mask_change_ratio(clean_mask, aug_mask)
                if args.save_overlays:
                    save_overlay(output_overlays / "{}.png".format(aug_stem), aug_image, aug_mask)
                if args.save_diffs:
                    save_diff_preview(output_diffs / "{}_diff.png".format(aug_stem), clean_image, aug_image)
                records.append({"source_image": str(image_path), "source_mask": str(mask_path),
                                "tile_idx": tile_idx, "x": x, "y": y, "variant": "aug{:02d}".format(aug_idx),
                                "mean_abs_image_diff": "{:.6f}".format(mean_diff),
                                "mask_change_ratio": "{:.6f}".format(changed_mask_ratio),
                                "image_path": str(out_image), "mask_path": str(out_mask)})
                aug_differences.append(mean_diff)
                near_identical_count += int(mean_diff < args.min_aug_diff)
                aug_count += 1

    fields = ["source_image", "source_mask", "tile_idx", "x", "y", "variant",
              "mean_abs_image_diff", "mask_change_ratio", "image_path", "mask_path"]
    with (args.out_dir / "train_tiles_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    mean_aug_diff = sum(aug_differences) / max(len(aug_differences), 1)
    print("Train input images: {}".format(len(images)))
    print("Clean train tiles: {}".format(clean_count))
    print("Augmented train tiles: {}".format(aug_count))
    print("Mean absolute RGB difference for augmented tiles: {:.4f}".format(mean_aug_diff))
    print("Augmented tiles below requested min diff: {}".format(near_identical_count))
    print("Manifest: {}".format(args.out_dir / "train_tiles_manifest.csv"))
    if args.save_diffs:
        print("Amplified difference previews: {}".format(output_diffs))
    print("Validation and test were not cropped or augmented.")


if __name__ == "__main__":
    main()
