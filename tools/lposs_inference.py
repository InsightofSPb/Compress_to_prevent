import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from mmcv import Config
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LPOSS/MaskCLIP inference on a folder of images."
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "--gt-images",
        "--gt_images",
        dest="gt_images",
        default=None,
        help="Optional directory with ground-truth masks to overlay.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for overlay visualizations.",
    )
    parser.add_argument(
        "--lposs-config",
        required=True,
        help="Path to the LPOSS Hydra config (e.g., configs/lposs.yaml).",
    )
    parser.add_argument(
        "--lposs-checkpoint",
        required=True,
        help="Path to the LPOSS checkpoint (.pth).",
    )
    parser.add_argument(
        "--lposs-dataset-config",
        required=True,
        help="Path to the dataset config with classes/palette.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string (default: cuda:0).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size for per-tile predictions (default: 512).",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=0,
        help="Overlap (in pixels) between tiles (default: 0).",
    )
    return parser.parse_args()


def _load_hydra_cfg(config_path: Path):
    config_dir = str(config_path.parent)
    config_name = config_path.stem
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def _get_nested(obj, keys: Sequence[str], default=None):
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if hasattr(cur, "get"):
            cur = cur.get(key, None)
        elif isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def extract_class_names(dataset_cfg) -> List[str]:
    candidates = []
    candidates.append(dataset_cfg.get("classes", None))
    metainfo = dataset_cfg.get("metainfo", None) or {}
    if isinstance(metainfo, dict):
        candidates.append(metainfo.get("classes", None))

    for path in [
        ("data", "train", "dataset", "classes"),
        ("data", "train", "dataset", "metainfo", "classes"),
        ("train_dataloader", "dataset", "classes"),
        ("train_dataloader", "dataset", "metainfo", "classes"),
        ("val_dataloader", "dataset", "classes"),
        ("val_dataloader", "dataset", "metainfo", "classes"),
    ]:
        node = dataset_cfg
        ok = True
        for key in path:
            if not hasattr(node, "get"):
                ok = False
                break
            node = node.get(key, None)
            if node is None:
                ok = False
                break
        if ok:
            candidates.append(node)

    for cand in candidates:
        if isinstance(cand, (list, tuple)) and cand and isinstance(cand[0], str):
            return list(cand)

    raise ValueError(
        "Could not infer `classes` from dataset config. "
        "Ensure it defines `classes = [...]` or `metainfo = dict(classes=[...])`."
    )


def extract_palette(dataset_cfg) -> List[List[int]]:
    candidates = []
    candidates.append(dataset_cfg.get("palette", None))
    metainfo = dataset_cfg.get("metainfo", None) or {}
    if isinstance(metainfo, dict):
        candidates.append(metainfo.get("palette", None))

    for path in [
        ("data", "train", "dataset", "palette"),
        ("data", "train", "dataset", "metainfo", "palette"),
        ("train_dataloader", "dataset", "palette"),
        ("train_dataloader", "dataset", "metainfo", "palette"),
        ("val_dataloader", "dataset", "palette"),
        ("val_dataloader", "dataset", "metainfo", "palette"),
    ]:
        node = dataset_cfg
        ok = True
        for key in path:
            if not hasattr(node, "get"):
                ok = False
                break
            node = node.get(key, None)
            if node is None:
                ok = False
                break
        if ok:
            candidates.append(node)

    for cand in candidates:
        if isinstance(cand, (list, tuple)) and cand and isinstance(cand[0], (list, tuple)):
            return [list(c) for c in cand]

    raise ValueError(
        "Could not infer `palette` from dataset config. "
        "Ensure it defines `palette = [...]` or `metainfo = dict(palette=[...])`."
    )


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state" in state:
        state = state["state"]

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        LOGGER.warning("Checkpoint load missing=%s unexpected=%s", missing, unexpected)


class IdentityHead(nn.Module):
    def forward(self, feats: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        return feats


class FineTuneWrapper(nn.Module):
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.mixer = IdentityHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone = getattr(self.base_model, "clip_backbone", None) or self.base_model

        logits = None
        feats = None

        try:
            out = backbone(x, return_feat=True)
            if isinstance(out, tuple) and len(out) >= 2:
                logits, feats = out[0], out[1]
            else:
                logits, feats = out, None
        except TypeError:
            out = backbone(x)
            if isinstance(out, tuple):
                logits = out[0]
                feats = out[1] if len(out) > 1 else None
            else:
                logits, feats = out, None

        if feats is None:
            feats = logits

        feats = self.mixer(feats, logits)

        decode_head = getattr(self.base_model, "decode_head", None)
        if decode_head is None and hasattr(self.base_model, "clip_backbone"):
            decode_head = getattr(self.base_model.clip_backbone, "decode_head", None)
        if decode_head is None:
            raise AttributeError("No decode_head found in base_model")

        mixed_logits = decode_head.cls_seg(feats)

        if mixed_logits.shape[-2:] != logits.shape[-2:]:
            mixed_logits = F.interpolate(
                mixed_logits, size=logits.shape[-2:], mode="bilinear", align_corners=False
            )

        return (logits + mixed_logits) / 2.0


def _infer_patch_size(model: nn.Module) -> int:
    candidates = []
    for obj in [model, getattr(model, "base_model", None), getattr(model, "clip_backbone", None)]:
        if obj is None:
            continue
        clip_backbone = getattr(obj, "clip_backbone", None)
        if clip_backbone is not None:
            value = getattr(clip_backbone, "patch_size", None)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        for attr in ("patch_size", "vit_patch_size", "dino_patch_size"):
            value = getattr(obj, attr, None)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        candidates.append(obj)

    for obj in candidates:
        backbone = getattr(obj, "backbone", None)
        if backbone is None:
            continue
        visual = getattr(backbone, "visual", None)
        if visual is not None:
            value = getattr(visual, "patch_size", None)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)

    return 14


def build_lposs_inferencer(
    config_path: Path,
    checkpoint_path: Path,
    dataset_config: Path,
    device: str,
) -> Tuple[nn.Module, List[str], List[List[int]], Tuple[List[float], List[float], bool, int]]:
    cfg = _load_hydra_cfg(config_path)
    dataset_cfg = Config.fromfile(str(dataset_config))
    class_names = extract_class_names(dataset_cfg)
    palette = extract_palette(dataset_cfg)

    base_model = build_model(cfg.model, class_names=class_names)
    seg_model = FineTuneWrapper(base_model)
    load_checkpoint(seg_model, checkpoint_path)

    dp = _get_nested(cfg, ["model", "data_preprocessor"], default=None) or _get_nested(
        cfg, ["data_preprocessor"], default=None
    )
    mean = _get_nested(dp, ["mean"], default=[123.675, 116.28, 103.53])
    std = _get_nested(dp, ["std"], default=[58.395, 57.12, 57.375])
    bgr_to_rgb = bool(_get_nested(dp, ["bgr_to_rgb"], default=True))

    mean = [float(x) for x in list(mean)]
    std = [float(x) for x in list(std)]

    seg_model = seg_model.to(device)
    seg_model.eval()

    patch_size = _infer_patch_size(seg_model)

    return seg_model, class_names, palette, (mean, std, bgr_to_rgb, patch_size)


def lposs_predict_map(
    image: np.ndarray,
    seg_model: nn.Module,
    norm_params: Tuple[List[float], List[float], bool, int],
) -> np.ndarray:
    mean, std, bgr_to_rgb, patch_size = norm_params

    img = image
    if bgr_to_rgb is False:
        img = img[..., ::-1].copy()

    height, width = img.shape[:2]
    x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t

    patch_size = max(1, int(patch_size))
    pad_h = (patch_size - (height % patch_size)) % patch_size
    pad_w = (patch_size - (width % patch_size)) % patch_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    device = next(seg_model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        logits = seg_model(x)
        logits = logits[:, :, :height, :width]
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
        probs = torch.softmax(logits, dim=1)

    probs = probs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return probs


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    palette: List[List[int]],
    ignore_index: int = 255,
) -> np.ndarray:
    overlay = image.copy()
    color_mask = np.zeros_like(image)

    for idx, color in enumerate(palette):
        color_mask[mask == idx] = color

    valid = mask != ignore_index
    overlay[valid] = (0.6 * overlay[valid] + 0.4 * color_mask[valid]).astype(np.uint8)
    return overlay


def add_legend(
    image: np.ndarray,
    class_names: List[str],
    palette: List[List[int]],
    font_scale: float = 0.45,
    line_height: int = 20,
    padding: int = 10,
) -> np.ndarray:
    legend_width = 260
    height, width = image.shape[:2]
    canvas = np.zeros((height, width + legend_width, 3), dtype=image.dtype)
    canvas[:, :width] = image
    canvas[:, width:] = 30

    y = padding + line_height
    for idx, name in enumerate(class_names):
        if y + line_height > height - padding:
            break
        color = palette[idx]
        cv2.rectangle(
            canvas,
            (width + padding, y - line_height + 4),
            (width + padding + 16, y + 4),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            name,
            (width + padding + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return canvas


def resolve_image_list(images_path: Path) -> List[Path]:
    if images_path.is_file():
        return [images_path]

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [p for p in images_path.rglob("*") if p.suffix.lower() in supported]
    return sorted(paths)


def resolve_gt_path(gt_dir: Path, image_path: Path) -> Optional[Path]:
    candidate = gt_dir / image_path.name
    if candidate.exists():
        return candidate

    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        alt = gt_dir / f"{image_path.stem}{ext}"
        if alt.exists():
            return alt
    return None


def load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int64)


def iter_tiles(
    image: np.ndarray,
    tile_size: int,
    overlap: int,
) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    height, width = image.shape[:2]
    stride = max(1, tile_size - overlap)
    tiles = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            tile = image[y:y_end, x:x_end]
            tiles.append(((x, y), tile))
    return tiles


def main() -> None:
    args = parse_args()

    images_path = Path(args.images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.gt_images) if args.gt_images else None

    seg_model, class_names, palette, norm_params = build_lposs_inferencer(
        config_path=Path(args.lposs_config),
        checkpoint_path=Path(args.lposs_checkpoint),
        dataset_config=Path(args.lposs_dataset_config),
        device=args.device,
    )

    LOGGER.info("Classes: %s", class_names)
    LOGGER.info("Palette size: %d", len(palette))

    ignore_index = 255
    dataset_cfg = Config.fromfile(str(args.lposs_dataset_config))
    ignore_index = dataset_cfg.get("ignore_index", ignore_index)

    image_paths = resolve_image_list(images_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_path}")

    overlay_dir = out_dir / "overlays"
    tiles_dir = out_dir / "tiles"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Running inference", unit="image"):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            LOGGER.warning("Skipping unreadable image: %s", image_path)
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        probs = lposs_predict_map(image_rgb, seg_model, norm_params)
        pred_mask = probs.argmax(axis=-1).astype(np.int64)

        pred_overlay = overlay_mask(image_rgb, pred_mask, palette, ignore_index)
        pred_overlay = add_legend(pred_overlay, class_names, palette)
        pred_out = overlay_dir / f"{image_path.stem}_pred{image_path.suffix}"
        cv2.imwrite(str(pred_out), cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))

        if gt_dir is not None:
            gt_path = resolve_gt_path(gt_dir, image_path)
            if gt_path is None:
                LOGGER.warning("GT mask not found for %s", image_path.name)
            else:
                gt_mask = load_mask(gt_path)
                if gt_mask.shape[:2] != image_rgb.shape[:2]:
                    gt_mask = cv2.resize(
                        gt_mask,
                        (image_rgb.shape[1], image_rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                gt_overlay = overlay_mask(image_rgb, gt_mask, palette, ignore_index)
                gt_overlay = add_legend(gt_overlay, class_names, palette)
                gt_out = overlay_dir / f"{image_path.stem}_gt{image_path.suffix}"
                cv2.imwrite(str(gt_out), cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))

        tile_root = tiles_dir / image_path.stem
        tile_root.mkdir(parents=True, exist_ok=True)
        tiles = iter_tiles(image_rgb, args.tile_size, args.tile_overlap)
        for (x, y), tile in tiles:
            tile_probs = lposs_predict_map(tile, seg_model, norm_params)
            tile_pred = tile_probs.argmax(axis=-1).astype(np.int64)
            tile_overlay = overlay_mask(tile, tile_pred, palette, ignore_index)
            tile_overlay = add_legend(tile_overlay, class_names, palette)
            tile_out = tile_root / f"{image_path.stem}_x{x}_y{y}_pred{image_path.suffix}"
            cv2.imwrite(str(tile_out), cv2.cvtColor(tile_overlay, cv2.COLOR_RGB2BGR))

    LOGGER.info("Inference complete. Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()