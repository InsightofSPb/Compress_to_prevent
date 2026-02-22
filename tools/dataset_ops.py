import random
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np


def pad_tile(
    tile: np.ndarray, target_h: int, target_w: int, is_mask: bool, pad_mode: str
) -> np.ndarray:
    h, w = tile.shape[:2]
    if h == target_h and w == target_w:
        return tile

    pad_bottom = max(0, target_h - h)
    pad_right = max(0, target_w - w)
    border_type = cv2.BORDER_REFLECT if pad_mode == "reflect" else cv2.BORDER_CONSTANT
    pad_value = 0 if is_mask else 0

    return cv2.copyMakeBorder(
        tile,
        0,
        pad_bottom,
        0,
        pad_right,
        border_type,
        value=pad_value,
    )


def generate_tiles(
    image: np.ndarray,
    mask: np.ndarray,
    tile_h: int,
    tile_w: int,
    stride_h: int,
    stride_w: int,
    pad_mode: str = "reflect",
    min_content_ratio: float = 0.0,
):
    tile_idx = 0
    h, w = image.shape[:2]
    for y in range(0, h, stride_h):
        for x in range(0, w, stride_w):
            content_h = min(tile_h, h - y)
            content_w = min(tile_w, w - x)
            total_area = tile_h * tile_w
            if total_area > 0:
                content_ratio = (content_h * content_w) / total_area
                if content_ratio < min_content_ratio:
                    continue
            img_tile = image[y : y + tile_h, x : x + tile_w]
            mask_tile = mask[y : y + tile_h, x : x + tile_w]
            img_tile = pad_tile(img_tile, tile_h, tile_w, is_mask=False, pad_mode=pad_mode)
            mask_tile = pad_tile(mask_tile, tile_h, tile_w, is_mask=True, pad_mode=pad_mode)
            yield tile_idx, x, y, img_tile, mask_tile
            tile_idx += 1


def apply_mixup(
    img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, mask2: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    mix_mask = np.random.beta(alpha, alpha, size=mask1.shape)
    mix_mask_img = mix_mask[..., None]
    mixed_img = (img1 * mix_mask_img + img2 * (1 - mix_mask_img)).astype(np.uint8)
    mixed_mask = np.where(mix_mask >= 0.5, mask1, mask2).astype(mask1.dtype)
    return mixed_img, mixed_mask


def apply_cutmix(
    img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, mask2: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
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


def _sample_zoom_crop(
    height: int,
    width: int,
    scale_range: Tuple[float, float],
    ratio_range: Tuple[float, float],
) -> Tuple[int, int]:
    scale = random.uniform(*scale_range)
    ratio = random.uniform(*ratio_range)
    crop_h = int(round(height * np.sqrt(scale / ratio)))
    crop_w = int(round(width * np.sqrt(scale * ratio)))
    crop_h = int(np.clip(crop_h, 1, height))
    crop_w = int(np.clip(crop_w, 1, width))
    return crop_h, crop_w


def apply_zoom(
    image: np.ndarray,
    mask: np.ndarray,
    zoom_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    if not zoom_cfg.get("enabled", False):
        return image, mask
    if random.random() >= zoom_cfg.get("p", 0.0):
        return image, mask

    h, w = image.shape[:2]
    scale_range = (zoom_cfg.get("scale_min", 0.7), zoom_cfg.get("scale_max", 1.0))
    ratio_range = (zoom_cfg.get("ratio_min", 0.9), zoom_cfg.get("ratio_max", 1.1))
    crop_h, crop_w = _sample_zoom_crop(h, w, scale_range, ratio_range)

    y0 = random.randint(0, h - crop_h) if h > crop_h else 0
    x0 = random.randint(0, w - crop_w) if w > crop_w else 0

    non_background = np.argwhere(mask > 0)
    damage_center_prob = zoom_cfg.get("damage_center_prob", 0.0)
    if non_background.size > 0 and random.random() < damage_center_prob:
        cy, cx = non_background[np.random.randint(len(non_background))]
        y0 = int(np.clip(cy - crop_h // 2, 0, h - crop_h))
        x0 = int(np.clip(cx - crop_w // 2, 0, w - crop_w))

    crop_img = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    crop_mask = mask[y0 : y0 + crop_h, x0 : x0 + crop_w]

    zoom_img = cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_LINEAR)
    zoom_mask = cv2.resize(crop_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return zoom_img, zoom_mask


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
        transforms.append(
            A.GaussNoise(var_limit=photo_cfg.get("noise_var", (10.0, 50.0)), p=photo_cfg.get("gaussian_noise", 0))
        )
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
