import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import yaml
from tqdm import tqdm


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


def apply_mixup(img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, mask2: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    mix_mask = np.random.beta(alpha, alpha, size=mask1.shape)
    mix_mask_img = mix_mask[..., None]
    mixed_img = (img1 * mix_mask_img + img2 * (1 - mix_mask_img)).astype(np.uint8)
    mixed_mask = np.where(mix_mask >= 0.5, mask1, mask2).astype(mask1.dtype)
    return mixed_img, mixed_mask


def apply_cutmix(img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, mask2: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img1.shape[:2]
    cut_ratio = np.random.beta(alpha, alpha)
    cut_w = int(w * np.sqrt(1 - cut_ratio))
    cut_h = int(h * np.sqrt(1 - cut_ratio))

    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    img = img1.copy()
    mask = mask1.copy()
    img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
    return img, mask


def build_transforms(config: Dict) -> A.Compose:
    transforms: List[A.BasicTransform] = []
    size_cfg = config.get("size", {})
    if "resize" in size_cfg:
        resize_cfg = size_cfg["resize"]
        transforms.append(
            A.Resize(
                resize_cfg.get("height"),
                resize_cfg.get("width"),
                interpolation=cv2.INTER_LINEAR,
            )
        )
    if "random_crop" in size_cfg and size_cfg["random_crop"].get("p", 0) > 0:
        crop_cfg = size_cfg["random_crop"]
        transforms.append(
            A.RandomCrop(
                height=crop_cfg.get("height"),
                width=crop_cfg.get("width"),
                p=crop_cfg.get("p", 1.0),
            )
        )

    geo_cfg = config.get("geometric", {})
    if geo_cfg.get("horizontal_flip", 0) > 0:
        transforms.append(A.HorizontalFlip(p=geo_cfg.get("horizontal_flip", 0)))
    if geo_cfg.get("vertical_flip", 0) > 0:
        transforms.append(A.VerticalFlip(p=geo_cfg.get("vertical_flip", 0)))
    if geo_cfg.get("random_rotate90", 0) > 0:
        transforms.append(A.RandomRotate90(p=geo_cfg.get("random_rotate90", 0)))
    if geo_cfg.get("shift_scale_rotate"):
        ssr = geo_cfg["shift_scale_rotate"]
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=ssr.get("shift_limit", 0.05),
                scale_limit=ssr.get("scale_limit", 0.1),
                rotate_limit=ssr.get("rotate_limit", 15),
                border_mode=cv2.BORDER_REFLECT_101,
                p=ssr.get("p", 0.5),
            )
        )

    photo_cfg = config.get("photometric", {})
    if photo_cfg.get("brightness_contrast", 0) > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=photo_cfg.get("brightness_limit", 0.2),
                contrast_limit=photo_cfg.get("contrast_limit", 0.2),
                p=photo_cfg.get("brightness_contrast", 0),
            )
        )
    if photo_cfg.get("hue_sat", 0) > 0:
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=photo_cfg.get("hue_shift_limit", 10),
                sat_shift_limit=photo_cfg.get("sat_shift_limit", 10),
                val_shift_limit=photo_cfg.get("val_shift_limit", 10),
                p=photo_cfg.get("hue_sat", 0),
            )
        )
    if photo_cfg.get("rgb_shift", 0) > 0:
        transforms.append(
            A.RGBShift(
                r_shift_limit=photo_cfg.get("r_shift", 10),
                g_shift_limit=photo_cfg.get("g_shift", 10),
                b_shift_limit=photo_cfg.get("b_shift", 10),
                p=photo_cfg.get("rgb_shift", 0),
            )
        )
    if photo_cfg.get("gaussian_noise", 0) > 0:
        transforms.append(A.GaussNoise(var_limit=photo_cfg.get("noise_var", (10.0, 50.0)), p=photo_cfg.get("gaussian_noise", 0)))
    if photo_cfg.get("blur", 0) > 0:
        transforms.append(A.Blur(blur_limit=photo_cfg.get("blur_limit", 3), p=photo_cfg.get("blur", 0)))
    if photo_cfg.get("clahe", 0) > 0:
        transforms.append(A.CLAHE(clip_limit=photo_cfg.get("clip_limit", 2.0), p=photo_cfg.get("clahe", 0)))

    weather_cfg = config.get("weather", {})
    if weather_cfg.get("rain", 0) > 0:
        transforms.append(A.RandomRain(p=weather_cfg.get("rain", 0)))
    if weather_cfg.get("snow", 0) > 0:
        transforms.append(A.RandomSnow(p=weather_cfg.get("snow", 0)))
    if weather_cfg.get("fog", 0) > 0:
        transforms.append(A.RandomFog(p=weather_cfg.get("fog", 0)))
    if weather_cfg.get("sun_flare", 0) > 0:
        transforms.append(A.RandomSunFlare(p=weather_cfg.get("sun_flare", 0)))
    if weather_cfg.get("shadow", 0) > 0:
        transforms.append(A.RandomShadow(p=weather_cfg.get("shadow", 0)))

    cutout_cfg = config.get("cutout")
    if cutout_cfg and cutout_cfg.get("p", 0) > 0:
        transforms.append(
            A.Cutout(
                num_holes=cutout_cfg.get("num_holes", 8),
                max_h_size=cutout_cfg.get("max_h_size", 32),
                max_w_size=cutout_cfg.get("max_w_size", 32),
                fill_value=cutout_cfg.get("fill_value", 0),
                p=cutout_cfg.get("p", 0),
            )
        )

    return A.Compose(transforms, additional_targets={"mask": "mask"})


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


def sample_partner(image_paths: List[Path], exclude: Path) -> Path:
    candidates = [p for p in image_paths if p != exclude]
    return random.choice(candidates) if candidates else exclude


def save_triplet(
    base_name: str,
    idx: int,
    image: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    output_dirs: Dict[str, Path],
    cfg: Dict,
) -> None:
    suffix = cfg.get("suffix", "aug")
    stem = f"{base_name}_{suffix}{idx}"
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
    output_dirs = {
        "images": Path(paths_cfg.get("output_images")),
        "masks": Path(paths_cfg.get("output_masks")),
        "overlays": Path(paths_cfg.get("output_overlays")),
    }
    ensure_dirs(*output_dirs.values())

    aug_cfg = config.get("augmentations", {})
    base_transform = build_transforms(aug_cfg)
    palette = build_palette(config)

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    image_paths = sorted(image_dir.glob("*"))
    num_aug = config.get("augmentations_per_image", 1)

    mixup_cfg = aug_cfg.get("mixup", {})
    cutmix_cfg = aug_cfg.get("cutmix", {})

    progress = tqdm(image_paths, desc="Augmenting images")
    for image_path in progress:
        loaded = load_image_mask(image_path, mask_dir, mapping)
        if loaded is None:
            expected_mask = load_mask_path(image_path, mask_dir, mapping)
            progress.write(
                f"Skipping {image_path.name}: missing or unreadable mask (expected at {expected_mask})"
            )
            continue
        image, mask = loaded
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for idx in range(num_aug):
            transformed = base_transform(image=image, mask=mask)
            aug_img, aug_mask = transformed["image"], transformed["mask"]

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
