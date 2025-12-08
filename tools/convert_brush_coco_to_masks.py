"""
Convert Label Studio "Brush labels to COCO" exports to per-pixel masks for LPOSS/MMSegmentation.

Features
- Maps category names to stable label IDs using the provided palette.
- Renders polygon or RLE segmentations to PNG masks (uint8) with background=0.
- Handles filenames with prefixed hashes (e.g., "abcd-IMG_123.jpg") by trying both hashed
  and plain versions when locating the image.
- Optionally writes color overlays for quick visual validation while preserving image size.

Example
-------
python tools/convert_brush_coco_to_masks.py \
    --coco-json /home/sasha/LPOSS/datasets/SPb_facades/annotations/result_coco.json \
    --images-root /home/sasha/LPOSS/datasets/SPb_facades/raw_images \
    --masks-out /home/sasha/LPOSS/datasets/SPb_facades/semantic_masks \
    --overlays-out /home/sasha/LPOSS/datasets/SPb_facades/overlays
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

try:
    from pycocotools import mask as mask_utils
except Exception as exc:  # pragma: no cover - import guard
    mask_utils = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


LOGGER = logging.getLogger("convert_coco")

# Label Studio palette used by the user (name -> (id, rgb))
LABELS = [
    ("BACKGROUND", 0, (0x00, 0x00, 0x00)),
    ("CRACK", 1, (0xE5, 0x39, 0x35)),
    ("SPALLING", 2, (0x1E, 0x88, 0xE5)),
    ("DELAMINATION", 3, (0x43, 0xA0, 0x47)),
    ("MISSING_ELEMENT", 4, (0xFB, 0x8C, 0x00)),
    ("WATER_STAIN", 5, (0x8E, 0x24, 0xAA)),
    ("EFFLORESCENCE", 6, (0xFD, 0xD8, 0x35)),
    ("CORROSION", 7, (0x00, 0xAC, 0xC1)),
    ("ORNAMENT_INTACT", 8, (0x9E, 0x9E, 0x9E)),
    ("REPAIRS", 9, (0x4E, 0x9E, 0x9E)),
    ("TEXT_OR_IMAGES", 10, (0x8E, 0x7E, 0x47)),
]

NAME_TO_LABEL_ID: Dict[str, int] = {name: idx for name, idx, _ in LABELS}
CATEGORY_COLOR: Dict[int, Tuple[int, int, int]] = {idx: color for _, idx, color in LABELS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", type=Path, required=True, help="Path to COCO JSON export.")
    parser.add_argument( "--images-root", type=Path, required=True, help="Directory containing the raw images." )
    parser.add_argument( "--masks-out", type=Path, required=True, help="Output directory for semantic masks (PNG)." )
    parser.add_argument( "--overlays-out", type=Path, default=None, help="Optional directory to store color overlays for visual QC.", )
    parser.add_argument( "--alpha", type=float, default=0.45, help="Alpha used when blending overlay masks onto the RGB images.", )
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def collect_annotations(annotations: Iterable[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations:
        grouped[ann["image_id"]].append(ann)
    return grouped


def map_categories(categories: Iterable[dict]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for cat in categories:
        name = cat["name"].strip()
        if name not in NAME_TO_LABEL_ID:
            raise ValueError(f"Unknown category '{name}'. Update LABELS mapping to include it.")
        mapping[cat["id"]] = NAME_TO_LABEL_ID[name]
    return mapping


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        for poly in segmentation:
            if len(poly) < 6:
                continue
            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(xy, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)

    if isinstance(segmentation, dict):
        if mask_utils is None:
            raise ImportError(
                "pycocotools is required to decode RLE masks. "
                "Install it with `pip install pycocotools` or `pip install pycocotools-windows` on Windows."
            ) from _IMPORT_ERROR
        rle = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded.sum(axis=2)
        return (decoded > 0).astype(np.uint8)

    raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")


def find_image_path(images_root: Path, file_name: str) -> Path | None:
    direct = images_root / file_name
    if direct.exists():
        return direct

    if "-" in file_name:
        tail = file_name.split("-", 1)[1]
        candidate = images_root / tail
        if candidate.exists():
            return candidate

    tail_only = Path(file_name).name
    fallback = list(images_root.glob(f"*{tail_only}"))
    if fallback:
        return fallback[0]

    return None


def build_overlay(mask: np.ndarray, image: Image.Image, alpha: float) -> Image.Image:
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    for label_id, color in CATEGORY_COLOR.items():
        region = mask == label_id
        if not np.any(region):
            continue
        overlay[region] = (*color, int(alpha * 255))
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    blended = Image.alpha_composite(image.convert("RGBA"), overlay_img)
    return blended.convert("RGB")


def convert(args: argparse.Namespace) -> None:
    coco = load_coco(args.coco_json)
    annotations_by_image = collect_annotations(coco.get("annotations", []))
    category_map = map_categories(coco.get("categories", []))

    ensure_dirs(args.masks_out, args.overlays_out)
    images = coco.get("images", [])
    
    for img_entry in tqdm(images, desc="Обработка изображений", unit="img", total=len(images)):
        image_id = img_entry["id"]
        file_name = img_entry["file_name"]
        width, height = img_entry["width"], img_entry["height"]

        img_path = find_image_path(args.images_root, file_name)
        if img_path is None:
            LOGGER.warning("Image not found: %s (id=%s)", file_name, image_id)
            continue

        image = Image.open(img_path).convert("RGB")
        if image.size != (width, height):
            LOGGER.warning(
                "Size mismatch for %s: JSON has (%d, %d), image is (%d, %d)",
                file_name,
                width,
                height,
                image.size[0],
                image.size[1],
            )
            width, height = image.size

        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in annotations_by_image.get(image_id, []):
            label_id = category_map[ann["category_id"]]
            seg = ann.get("segmentation")
            if seg is None:
                continue
            ann_mask = decode_segmentation(seg, height, width)
            mask[ann_mask > 0] = label_id

        mask_path = args.masks_out / (Path(file_name).stem + ".png")
        Image.fromarray(mask, mode="L").save(mask_path)

        if args.overlays_out:
            overlay_path = args.overlays_out / (Path(file_name).stem + "_overlay.jpg")
            overlay_img = build_overlay(mask, image, args.alpha)
            overlay_img.save(overlay_path, quality=95)

    LOGGER.info("Conversion complete: %s", args.masks_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    convert(parse_args())
