#!/usr/bin/env python3
"""Prepare clean and augmented train-only tiles for facade segmentation.

CutMix pastes an RGB/mask region from another train tile. CutOut sets the
hidden target area to a configurable ignore label. CutBlur changes local RGB
resolution while leaving the target mask unchanged. None of these operations
is used for validation, test, or RGB compression data.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create augmented train-only tiles for segmentation.")
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=448)
    p.add_argument("--stride", type=int, default=224)
    p.add_argument("--min-content-ratio", type=float, default=0.60)
    p.add_argument("--aug-copies", type=int, default=2)
    p.add_argument("--min-aug-diff", type=float, default=1.0)
    p.add_argument("--max-aug-tries", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--save-diffs", action="store_true")
    p.add_argument("--cutmix-p", type=float, default=0.20)
    p.add_argument("--cutmix-area-min", type=float, default=0.08)
    p.add_argument("--cutmix-area-max", type=float, default=0.25)
    p.add_argument("--cutout-p", type=float, default=0.12)
    p.add_argument("--cutout-area-min", type=float, default=0.01)
    p.add_argument("--cutout-area-max", type=float, default=0.04)
    p.add_argument("--cutout-ignore-label", type=int, default=255)
    p.add_argument("--cutblur-p", type=float, default=0.20)
    p.add_argument("--cutblur-area-min", type=float, default=0.08)
    p.add_argument("--cutblur-area-max", type=float, default=0.25)
    p.add_argument("--cutblur-scale-min", type=int, default=2)
    p.add_argument("--cutblur-scale-max", type=int, default=4)
    return p.parse_args()


def validate_args(a: argparse.Namespace) -> None:
    if a.tile_size <= 0 or a.stride <= 0 or a.aug_copies < 0 or a.max_aug_tries <= 0:
        raise ValueError("Invalid tile, stride, aug-copies or max-aug-tries value.")
    for name in ("cutmix_p", "cutout_p", "cutblur_p"):
        if not 0 <= getattr(a, name) <= 1:
            raise ValueError("{} must be between 0 and 1.".format(name))
    for prefix in ("cutmix", "cutout", "cutblur"):
        low, high = getattr(a, prefix + "_area_min"), getattr(a, prefix + "_area_max")
        if not 0 < low <= high < 1:
            raise ValueError("Invalid {} area limits.".format(prefix))
    if not 0 <= a.cutout_ignore_label <= 255:
        raise ValueError("cutout-ignore-label must be in [0, 255].")


def find_mask(image: Path, masks_dir: Path) -> Path:
    direct = masks_dir / image.name
    if direct.exists():
        return direct
    matches = [p for p in masks_dir.glob("{}.*".format(image.stem)) if p.suffix.lower() in IMAGE_EXTS]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("Exactly one mask is required for {}; found {}.".format(image.name, matches))


def load_pair(image_path: Path, masks_dir: Path) -> Tuple[np.ndarray, np.ndarray, Path]:
    mask_path = find_mask(image_path, masks_dir)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise ValueError("Unreadable input pair: {}, {}".format(image_path, mask_path))
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("RGB/mask shape mismatch: {} vs {}".format(image_path.name, mask_path.name))
    return image, mask, mask_path


def pad(array: np.ndarray, size: int, mask: bool) -> np.ndarray:
    h, w = array.shape[:2]
    if h == size and w == size:
        return array
    border = cv2.BORDER_CONSTANT if mask else cv2.BORDER_REFLECT_101
    return cv2.copyMakeBorder(array, 0, size - h, 0, size - w, border, value=0)


def tiles(image: np.ndarray, mask: np.ndarray, size: int, stride: int, ratio: float) -> Iterator[Tuple[int, int, int, np.ndarray, np.ndarray]]:
    index = 0
    height, width = image.shape[:2]
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            content_h, content_w = min(size, height - y), min(size, width - x)
            if content_h * content_w / float(size * size) < ratio:
                continue
            yield index, x, y, pad(image[y:y + size, x:x + size], size, False), pad(mask[y:y + size, x:x + size], size, True)
            index += 1


def base_transform() -> A.Compose:
    return A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.16, contrast_limit=0.16, p=1.0),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=10, p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        ], p=1.0),
        A.HorizontalFlip(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.035, scale_limit=0.075, rotate_limit=6,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.40),
        A.OneOf([A.GaussNoise(var_limit=(4.0, 16.0), p=1.0), A.Blur(blur_limit=3, p=1.0)], p=0.12),
    ])


def random_box(h: int, w: int, low: float, high: float) -> Tuple[int, int, int, int]:
    area = random.uniform(low, high) * h * w
    aspect = random.uniform(0.6, 1.67)
    bw = max(1, min(w, int((area * aspect) ** 0.5)))
    bh = max(1, min(h, int((area / aspect) ** 0.5)))
    x = random.randint(0, w - bw) if bw < w else 0
    y = random.randint(0, h - bh) if bh < h else 0
    return x, y, x + bw, y + bh


def partner_tile(paths: Sequence[Path], current: Path, masks_dir: Path, a: argparse.Namespace) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    other = [p for p in paths if p != current]
    if not other:
        return None
    image, mask, _ = load_pair(random.choice(other), masks_dir)
    choices = list(tiles(image, mask, a.tile_size, a.stride, a.min_content_ratio))
    if not choices:
        return None
    selected = random.choice(choices)
    return selected[3], selected[4]


def augment(clean_image: np.ndarray, clean_mask: np.ndarray, source: Path, images: Sequence[Path],
            masks_dir: Path, transform: A.Compose, a: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, float, str, float, float, float]:
    best = None
    for _ in range(a.max_aug_tries):
        result = transform(image=clean_image, mask=clean_mask)
        image, mask = result["image"], result["mask"]
        ops = ["base_photo"]
        mix_ratio = out_ratio = blur_ratio = 0.0
        if random.random() < a.cutmix_p:
            partner = partner_tile(images, source, masks_dir, a)
            if partner is not None:
                x0, y0, x1, y1 = random_box(image.shape[0], image.shape[1], a.cutmix_area_min, a.cutmix_area_max)
                image, mask = image.copy(), mask.copy()
                image[y0:y1, x0:x1] = partner[0][y0:y1, x0:x1]
                mask[y0:y1, x0:x1] = partner[1][y0:y1, x0:x1]
                mix_ratio = (x1 - x0) * (y1 - y0) / float(image.shape[0] * image.shape[1])
                ops.append("cutmix")
        if random.random() < a.cutblur_p:
            x0, y0, x1, y1 = random_box(image.shape[0], image.shape[1], a.cutblur_area_min, a.cutblur_area_max)
            crop = image[y0:y1, x0:x1]
            factor = random.randint(a.cutblur_scale_min, a.cutblur_scale_max)
            low = cv2.resize(crop, (max(1, crop.shape[1] // factor), max(1, crop.shape[0] // factor)), interpolation=cv2.INTER_AREA)
            image = image.copy()
            image[y0:y1, x0:x1] = cv2.resize(low, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
            blur_ratio = (x1 - x0) * (y1 - y0) / float(image.shape[0] * image.shape[1])
            ops.append("cutblur")
        if random.random() < a.cutout_p:
            x0, y0, x1, y1 = random_box(image.shape[0], image.shape[1], a.cutout_area_min, a.cutout_area_max)
            image, mask = image.copy(), mask.copy()
            image[y0:y1, x0:x1] = np.rint(image.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
            mask[y0:y1, x0:x1] = a.cutout_ignore_label
            out_ratio = (x1 - x0) * (y1 - y0) / float(image.shape[0] * image.shape[1])
            ops.append("cutout_ignore{}".format(a.cutout_ignore_label))
        diff = float(np.abs(image.astype(np.float32) - clean_image.astype(np.float32)).mean())
        candidate = image, mask, diff, "+".join(ops), mix_ratio, out_ratio, blur_ratio
        if best is None or diff > best[2]:
            best = candidate
        if diff >= a.min_aug_diff:
            return candidate
    return best


def save_pair(images_dir: Path, masks_dir: Path, name: str, image: np.ndarray, mask: np.ndarray) -> Tuple[Path, Path]:
    image_path, mask_path = images_dir / (name + ".png"), masks_dir / (name + ".png")
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    return image_path, mask_path


def save_overlay(path: Path, image: np.ndarray, mask: np.ndarray, ignore: int) -> None:
    out = image.copy()
    foreground, ignored = (mask > 0) & (mask != ignore), mask == ignore
    out[foreground] = (0.55 * out[foreground] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
    out[ignored] = (0.4 * out[ignored] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.imwrite(str(path), out)


def main() -> None:
    a = parse_args()
    validate_args(a)
    source_images, source_masks = a.train_root / "images", a.train_root / "masks"
    if not source_images.is_dir() or not source_masks.is_dir():
        raise FileNotFoundError("Expected <train-root>/images and <train-root>/masks.")
    if a.out_dir.exists() and any(a.out_dir.iterdir()) and not a.overwrite:
        raise FileExistsError("Output is non-empty; pass --overwrite.")
    random.seed(a.seed)
    np.random.seed(a.seed)
    output_images, output_masks = a.out_dir / "images", a.out_dir / "masks"
    overlays, diffs = a.out_dir / "overlays", a.out_dir / "diffs"
    for folder in (output_images, output_masks):
        folder.mkdir(parents=True, exist_ok=True)
    if a.save_overlays:
        overlays.mkdir(parents=True, exist_ok=True)
    if a.save_diffs:
        diffs.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in source_images.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    transform, rows, op_counts, diffs_values = base_transform(), [], {"cutmix": 0, "cutout": 0, "cutblur": 0}, []
    clean_count = aug_count = below_min = 0
    for image_path in tqdm(images, desc="Preparing train tiles"):
        image, mask, mask_path = load_pair(image_path, source_masks)
        for index, x, y, clean_image, clean_mask in tiles(image, mask, a.tile_size, a.stride, a.min_content_ratio):
            base = "{}_tile{:04d}_x{}_y{}".format(image_path.stem, index, x, y)
            image_out, mask_out = save_pair(output_images, output_masks, base + "_clean", clean_image, clean_mask)
            rows.append([str(image_path), str(mask_path), index, x, y, "clean", "clean", "0", "0", "0", "0", "0", str(image_out), str(mask_out)])
            clean_count += 1
            for number in range(a.aug_copies):
                aug_image, aug_mask, diff, ops, mix, out, blur = augment(clean_image, clean_mask, image_path, images, source_masks, transform, a)
                stem = "{}_aug{:02d}".format(base, number)
                image_out, mask_out = save_pair(output_images, output_masks, stem, aug_image, aug_mask)
                if a.save_overlays:
                    save_overlay(overlays / (stem + ".png"), aug_image, aug_mask, a.cutout_ignore_label)
                if a.save_diffs:
                    cv2.imwrite(str(diffs / (stem + "_diff.png")), np.clip(cv2.absdiff(clean_image, aug_image) * 4, 0, 255))
                mask_diff = float(np.mean(clean_mask != aug_mask))
                rows.append([str(image_path), str(mask_path), index, x, y, "aug{:02d}".format(number), ops, diff, mask_diff, mix, out, blur, str(image_out), str(mask_out)])
                for key in op_counts:
                    op_counts[key] += int(key in ops)
                diffs_values.append(diff)
                below_min += int(diff < a.min_aug_diff)
                aug_count += 1
    fields = ["source_image", "source_mask", "tile_idx", "x", "y", "variant", "augmentation_ops", "mean_abs_image_diff", "mask_change_ratio", "cutmix_area_ratio", "cutout_ignore_ratio", "cutblur_area_ratio", "image_path", "mask_path"]
    with (a.out_dir / "train_tiles_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)
    print("Clean train tiles: {}".format(clean_count))
    print("Augmented train tiles: {}".format(aug_count))
    print("Mean augmented RGB difference: {:.4f}".format(sum(diffs_values) / max(len(diffs_values), 1)))
    print("Augmented tiles below requested min diff: {}".format(below_min))
    print("Applied special augmentations: {}".format(op_counts))
    print("CutOut uses ignore label {}; verify your loss ignore_index.".format(a.cutout_ignore_label))


if __name__ == "__main__":
    main()
