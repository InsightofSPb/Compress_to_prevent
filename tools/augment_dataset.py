import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from dataset_ops import (
    apply_cutmix,
    apply_mixup,
    apply_zoom,
    build_transforms,
    generate_tiles,
)


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_palette(config: Dict) -> np.ndarray:
    palette = config.get("palette")
    if palette:
        return np.array(palette, dtype=np.uint8)

    num_colors = 256
    rng = np.random.default_rng(config.get("seed", 42))
    return rng.integers(0, 255, size=(num_colors, 3), dtype=np.uint8)



def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(mask)
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in unique_labels:
        color_mask[mask == label] = palette[int(label) % len(palette)]
    return color_mask


def create_overlay(image: np.ndarray, mask: np.ndarray, palette: np.ndarray, alpha: float) -> np.ndarray:
    color_mask = colorize_mask(mask, palette)
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)



def load_mask_path(image_path: Path, mask_dir: Path, mapping: Dict[str, str]) -> Path:
    mapped = mapping.get(image_path.name)
    if mapped is None:
        mask_path = mask_dir / image_path.name
    else:
        mapped_path = Path(mapped)
        mask_path = mapped_path if mapped_path.is_absolute() else mask_dir / mapped
    return mask_path


def load_image_mask(
    image_path: Path, mask_dir: Path, mapping: Dict[str, str]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    mask_path = load_mask_path(image_path, mask_dir, mapping)
    if not mask_path.exists():
        return None
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return image, mask


def validate_input_dirs(image_dir: Path, mask_dir: Path) -> None:
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}. Update paths.images in the config."
        )
    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Mask directory not found: {mask_dir}. Update paths.masks in the config."
        )
    if not any(image_dir.iterdir()):
        raise FileNotFoundError(
            f"Image directory {image_dir} is empty. Provide input images before augmenting."
        )


def sample_partner(image_paths: List[Path], exclude: Path) -> Path:
    candidates = [p for p in image_paths if p != exclude]
    return random.choice(candidates) if candidates else exclude


def save_triplet(
    base_name: str,
    tile_idx: Optional[int],
    idx: int,
    image: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    output_dirs: Dict[str, Path],
    cfg: Dict,
) -> None:
    suffix = cfg.get("suffix", "aug")
    tile_suffix = f"_tile{tile_idx}" if tile_idx is not None else ""
    stem = f"{base_name}{tile_suffix}_{suffix}{idx}"
    image_name = f"{stem}.{cfg.get('image_format', 'png')}"
    mask_name = f"{stem}.{cfg.get('mask_format', 'png')}"
    overlay_name = f"{stem}.{cfg.get('overlay_format', 'png')}"

    cv2.imwrite(str(output_dirs["images"] / image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dirs["masks"] / mask_name), mask)
    cv2.imwrite(str(output_dirs["overlays"] / overlay_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def augment_dataset(config: Dict) -> None:
    paths_cfg = config.get("paths", {})
    image_dir = Path(paths_cfg.get("images"))
    mask_dir = Path(paths_cfg.get("masks"))
    mapping_path = paths_cfg.get("pairs")
    mapping: Dict[str, str] = {}
    if mapping_path:
        mapping = load_config(Path(mapping_path))
        if not isinstance(mapping, dict):
            raise ValueError("pairs mapping file must contain a dictionary of image->mask names")
    validate_input_dirs(image_dir, mask_dir)
    output_dirs = {
        "images": Path(paths_cfg.get("output_images")),
        "masks": Path(paths_cfg.get("output_masks")),
        "overlays": Path(paths_cfg.get("output_overlays")),
    }
    ensure_dirs(*output_dirs.values())

    aug_cfg = config.get("augmentations", {})
    base_transform = build_transforms(aug_cfg)
    palette = build_palette(config)

    tiling_cfg = config.get("tiling", {})
    tile_enabled = tiling_cfg.get("enabled", False)
    tile_h = tiling_cfg.get("height") or aug_cfg.get("size", {}).get("resize", {}).get("height")
    tile_w = tiling_cfg.get("width") or aug_cfg.get("size", {}).get("resize", {}).get("width")
    stride_h = tiling_cfg.get("stride_h", tile_h)
    stride_w = tiling_cfg.get("stride_w", tile_w)
    pad_mode = tiling_cfg.get("pad_mode", "reflect")
    min_content_ratio = tiling_cfg.get("min_content_ratio", 0.0)

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    image_paths = sorted(image_dir.glob("*"))
    num_aug = config.get("augmentations_per_image", 1)

    mixup_cfg = aug_cfg.get("mixup", {})
    cutmix_cfg = aug_cfg.get("cutmix", {})
    zoom_cfg = aug_cfg.get("zoom", {})

    progress = tqdm(image_paths, desc="Augmenting images")
    for image_path in progress:
        loaded = load_image_mask(image_path, mask_dir, mapping)
        if loaded is None:
            expected_mask = load_mask_path(image_path, mask_dir, mapping)
            progress.write(
                "Skipping {img}: missing or unreadable mask (expected at {mask}). "
                "Ensure the mask file exists or provide a name mapping via paths.pairs."
            .format(img=image_path.name, mask=expected_mask)
            )
            continue
        image, mask = loaded
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if tile_enabled and tile_h is not None and tile_w is not None:
            tiles = list(
                generate_tiles(
                    image,
                    mask,
                    tile_h,
                    tile_w,
                    stride_h,
                    stride_w,
                    pad_mode,
                    min_content_ratio,
                )
            )
        else:
            tiles = [(None, 0, 0, image, mask)]

        for tile_idx, _x, _y, tile_img, tile_mask in tiles:
            for idx in range(num_aug):
                transformed = base_transform(image=tile_img, mask=tile_mask)
                aug_img, aug_mask = transformed["image"], transformed["mask"]
                aug_img, aug_mask = apply_zoom(aug_img, aug_mask, zoom_cfg)

                partner_img, partner_mask = None, None
                if mixup_cfg.get("p", 0) > 0 and random.random() < mixup_cfg["p"]:
                    partner_path = sample_partner(image_paths, image_path)
                    partner_loaded = load_image_mask(partner_path, mask_dir, mapping)
                    if partner_loaded:
                        partner_image, partner_mask = partner_loaded
                        partner_image = cv2.cvtColor(partner_image, cv2.COLOR_BGR2RGB)
                        partner = base_transform(image=partner_image, mask=partner_mask)
                        aug_img, aug_mask = apply_mixup(
                            aug_img,
                            aug_mask,
                            partner["image"],
                            partner["mask"],
                            mixup_cfg.get("alpha", 0.4),
                        )

                if cutmix_cfg.get("p", 0) > 0 and random.random() < cutmix_cfg["p"]:
                    if partner_img is None:
                        partner_path = sample_partner(image_paths, image_path)
                        partner_loaded = load_image_mask(partner_path, mask_dir, mapping)
                        if partner_loaded:
                            partner_image, partner_mask = partner_loaded
                            partner_image = cv2.cvtColor(partner_image, cv2.COLOR_BGR2RGB)
                            partner = base_transform(image=partner_image, mask=partner_mask)
                            partner_img, partner_mask = partner["image"], partner["mask"]
                    if partner_img is not None:
                        aug_img, aug_mask = apply_cutmix(
                            aug_img,
                            aug_mask,
                            partner_img,
                            partner_mask,
                            cutmix_cfg.get("alpha", 1.0),
                        )

                overlay = create_overlay(
                    aug_img,
                    aug_mask,
                    palette,
                    config.get("overlay_alpha", 0.45),
                )
                save_triplet(
                    image_path.stem,
                    tile_idx,
                    idx,
                    aug_img,
                    aug_mask,
                    overlay,
                    output_dirs,
                    config.get("save", {}),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline dataset augmentation for facades")
    parser.add_argument(
        "-c", "--config", default="configs/augmentation.yaml", help="Path to augmentation config"
    )
    args = parser.parse_args()
    config = load_config(Path(args.config))
    augment_dataset(config)


if __name__ == "__main__":
    main()
