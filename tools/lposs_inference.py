import argparse
import json
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

STRUCTURAL_DAMAGE = {"CRACK", "SPALLING", "DELAMINATION", "MISSING_ELEMENT"}
SURFACE_STAIN = {"WATER_STAIN", "EFFLORESCENCE", "CORROSION"}
HUMAN_ACTIVITY = {"TEXT_OR_IMAGES", "REPAIRS"}

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
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Compute metrics from GT masks (requires --gt-images).",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Optional path to write metrics JSON summary.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Print prediction/GT distributions for first N full images.",
    )
    parser.add_argument(
        "--sanity-n",
        type=int,
        default=1,
        help="How many full images to print sanity stats for (default: 1).",
    )
    return parser.parse_args()


def _load_hydra_cfg(config_path: Path):
    config_dir = str(config_path.parent)
    config_name = config_path.stem
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


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
) -> Tuple[nn.Module, List[str], List[List[int]], int]:
    cfg = _load_hydra_cfg(config_path)
    dataset_cfg = Config.fromfile(str(dataset_config))
    class_names = extract_class_names(dataset_cfg)
    palette = extract_palette(dataset_cfg)

    base_model = build_model(cfg.model, class_names=class_names)
    seg_model = FineTuneWrapper(base_model)
    load_checkpoint(seg_model, checkpoint_path)

    seg_model = seg_model.to(device)
    seg_model.eval()

    patch_size = _infer_patch_size(seg_model)

    return seg_model, class_names, palette, patch_size


def lposs_predict_map(
    image: np.ndarray,
    seg_model: nn.Module,
    patch_size: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    x = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

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


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0)


def _compute_group_metrics(confusion: np.ndarray, group_indices: List[int]) -> dict:
    tp = confusion[group_indices, :][:, group_indices].sum()
    fp = confusion[:, group_indices].sum() - tp
    fn = confusion[group_indices, :].sum() - tp
    iou = _safe_divide(tp, tp + fp + fn)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    acc = _safe_divide(tp, tp + fn)
    return {"iou": float(iou), "f1": float(f1), "accuracy": float(acc)}


def _compute_metrics_from_confusion(confusion: np.ndarray, class_names: List[str]) -> dict:
    true_positives = np.diag(confusion)
    false_positives = confusion.sum(axis=0) - true_positives
    false_negatives = confusion.sum(axis=1) - true_positives

    class_iou = _safe_divide(true_positives, true_positives + false_positives + false_negatives)
    class_f1 = _safe_divide(2 * true_positives, 2 * true_positives + false_positives + false_negatives)
    class_acc = _safe_divide(true_positives, true_positives + false_negatives)

    class_metrics = {}
    for idx, name in enumerate(class_names):
        class_metrics[name] = {
            "iou": float(class_iou[idx]),
            "f1": float(class_f1[idx]),
            "accuracy": float(class_acc[idx]),
        }

    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    group_indices = {
        "STRUCTURAL_DAMAGE": [name_to_idx[c] for c in STRUCTURAL_DAMAGE if c in name_to_idx],
        "SURFACE_STAIN": [name_to_idx[c] for c in SURFACE_STAIN if c in name_to_idx],
        "HUMAN_ACTIVITY": [name_to_idx[c] for c in HUMAN_ACTIVITY if c in name_to_idx],
    }

    group_metrics = {}
    for group_name, indices in group_indices.items():
        if indices:
            group_metrics[group_name] = _compute_group_metrics(confusion, indices)

    return {
        "mIoU": float(np.nanmean(class_iou)),
        "mF1": float(np.nanmean(class_f1)),
        "mAcc": float(np.nanmean(class_acc)),
        "class_metrics": class_metrics,
        "group_metrics": group_metrics,
    }


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

    if args.eval and not args.gt_images:
        raise ValueError("--eval requires --gt-images")

    images_path = Path(args.images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.gt_images) if args.gt_images else None

    seg_model, class_names, palette, patch_size = build_lposs_inferencer(
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

    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    num_images_with_gt = 0
    sanity_printed = 0

    for image_path in tqdm(image_paths, desc="Running inference", unit="image"):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            LOGGER.warning("Skipping unreadable image: %s", image_path)
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        probs = lposs_predict_map(image_rgb, seg_model, patch_size)
        pred_mask = probs.argmax(axis=-1).astype(np.int64)

        pred_overlay = overlay_mask(image_rgb, pred_mask, palette, ignore_index)
        pred_overlay = add_legend(pred_overlay, class_names, palette)
        pred_out = overlay_dir / f"{image_path.stem}_pred{image_path.suffix}"
        cv2.imwrite(str(pred_out), cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))

        gt_mask_for_eval = None
        if gt_dir is not None:
            gt_path = resolve_gt_path(gt_dir, image_path)
            if gt_path is None:
                LOGGER.warning("GT mask not found for %s", image_path.name)
            else:
                gt_mask = load_mask(gt_path)
                if gt_mask.shape[:2] != image_rgb.shape[:2]:
                    LOGGER.warning(
                        "Resizing GT mask for %s from %s to %s",
                        image_path.name,
                        gt_mask.shape[:2],
                        image_rgb.shape[:2],
                    )
                    gt_mask = cv2.resize(
                        gt_mask,
                        (image_rgb.shape[1], image_rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                gt_mask_for_eval = gt_mask
                gt_overlay = overlay_mask(image_rgb, gt_mask, palette, ignore_index)
                gt_overlay = add_legend(gt_overlay, class_names, palette)
                gt_out = overlay_dir / f"{image_path.stem}_gt{image_path.suffix}"
                cv2.imwrite(str(gt_out), cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))

        if args.eval:
            if gt_mask_for_eval is None:
                LOGGER.warning("Skipping metrics for %s: GT mask missing", image_path.name)
            else:
                valid = gt_mask_for_eval != ignore_index
                if np.any(valid):
                    gt_flat = gt_mask_for_eval[valid].reshape(-1)
                    pred_flat = pred_mask[valid].reshape(-1)
                    combined = gt_flat * num_classes + pred_flat
                    hist = np.bincount(combined, minlength=num_classes * num_classes)
                    confusion += hist.reshape(num_classes, num_classes)
                num_images_with_gt += 1

        if args.sanity_check and sanity_printed < max(0, args.sanity_n):
            pred_unique, pred_counts = np.unique(pred_mask, return_counts=True)
            pred_dist = {class_names[int(i)]: int(c) for i, c in zip(pred_unique, pred_counts)}
            LOGGER.info("[SANITY] %s pred distribution: %s", image_path.name, pred_dist)
            if gt_mask_for_eval is not None:
                gt_unique, gt_counts = np.unique(gt_mask_for_eval, return_counts=True)
                gt_dist = {
                    (class_names[int(i)] if int(i) < len(class_names) else f"IGNORE_{int(i)}"): int(c)
                    for i, c in zip(gt_unique, gt_counts)
                }
                ignored_frac = float((gt_mask_for_eval == ignore_index).mean())
                LOGGER.info("[SANITY] %s gt distribution: %s", image_path.name, gt_dist)
                LOGGER.info("[SANITY] %s gt ignored_frac: %.4f", image_path.name, ignored_frac)
            sanity_printed += 1

        tile_root = tiles_dir / image_path.stem
        tile_root.mkdir(parents=True, exist_ok=True)
        tiles = iter_tiles(image_rgb, args.tile_size, args.tile_overlap)
        for (x, y), tile in tiles:
            tile_probs = lposs_predict_map(tile, seg_model, patch_size)
            tile_pred = tile_probs.argmax(axis=-1).astype(np.int64)
            tile_overlay = overlay_mask(tile, tile_pred, palette, ignore_index)
            tile_overlay = add_legend(tile_overlay, class_names, palette)
            tile_out = tile_root / f"{image_path.stem}_x{x}_y{y}_pred{image_path.suffix}"
            cv2.imwrite(str(tile_out), cv2.cvtColor(tile_overlay, cv2.COLOR_RGB2BGR))

    if args.eval:
        metrics = _compute_metrics_from_confusion(confusion, class_names)
        LOGGER.info(
            "Evaluation summary — mIoU: %.4f | mF1: %.4f | mAcc: %.4f",
            metrics["mIoU"],
            metrics["mF1"],
            metrics["mAcc"],
        )
        sorted_by_iou = sorted(
            metrics["class_metrics"].items(), key=lambda item: item[1]["iou"]
        )
        worst_k = min(5, len(sorted_by_iou))
        LOGGER.info(
            "Worst-%d classes by IoU: %s",
            worst_k,
            [
                {"class": name, "iou": vals["iou"], "f1": vals["f1"], "accuracy": vals["accuracy"]}
                for name, vals in sorted_by_iou[:worst_k]
            ],
        )

        if args.metrics_out:
            summary = {
                "num_images_total": len(image_paths),
                "num_images_with_gt": num_images_with_gt,
                "checkpoint": str(args.lposs_checkpoint),
                "dataset_config": str(args.lposs_dataset_config),
                "mIoU": metrics["mIoU"],
                "mF1": metrics["mF1"],
                "mAcc": metrics["mAcc"],
                "class_metrics": metrics["class_metrics"],
                "group_metrics": metrics["group_metrics"],
            }
            metrics_out_path = Path(args.metrics_out)
            metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            LOGGER.info("Saved metrics JSON to %s", metrics_out_path)

    LOGGER.info("Inference complete. Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
