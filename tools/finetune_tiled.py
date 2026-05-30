"""Single-GPU LPOSS/MaskCLIP fine-tuning with stitched tiled validation.

Training reads pre-generated augmented train tiles. Validation reads clean tiles
from a deterministic tiling manifest, runs the model tile-wise, averages logits
in overlapping regions, then computes loss and semantic segmentation metrics on
the reconstructed full-resolution validation images.

Optionally, online train metrics are accumulated from the same forward passes
used for optimization. They add no second train inference pass and are intended
for diagnostics only: train predictions are produced on augmented samples in
training mode before the current optimizer update.

A separate fixed clean train probe may be evaluated after each epoch. It is
intentionally small and is used only for interpretable visual diagnostics:
fixed examples plus best/worst damage-union predictions are saved per epoch.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Make repository-local packages (in particular ./mmseg) importable before
# importing them. This removes the need to export PYTHONPATH manually.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2  # noqa: E402
import mmcv  # type: ignore  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from mmseg.datasets import build_dataloader, build_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

import finetune as common  # noqa: E402
from helpers.logger import get_logger  # noqa: E402
from models import build_model  # noqa: E402

DAMAGE_CLASS_NAMES = (
    "CRACK",
    "SPALLING",
    "DELAMINATION",
    "MISSING_ELEMENT",
    "WATER_STAIN",
    "EFFLORESCENCE",
    "CORROSION",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LPOSS with stitched tiled validation")
    parser.add_argument("config")
    parser.add_argument("--train-dataset-config", required=True)
    parser.add_argument("--val-dataset-config", required=True)
    parser.add_argument("--val-tiles-manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--unfreeze-depth", type=int, default=0, help="-1 for full MaskCLIP unfreeze")
    parser.add_argument("--mix-strategy", choices=["add", "concat", "replace"], default="add")
    parser.add_argument("--use-embedding-mixer", action="store_true")
    parser.add_argument("--loss-mode", choices=["ce_weighted", "ce_dice", "focal_dice"], default="ce_weighted")
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weights", default="auto", help='"auto", "none", or comma-separated values')
    parser.add_argument("--class-weight-mode", choices=["inverse", "median_freq", "effective_num"], default="inverse")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--val-save-visualizations",
        type=int,
        default=10,
        help="Number of stitched validation visualizations saved per epoch; use -1 to save all.",
    )
    parser.add_argument(
        "--select-best-by",
        choices=[
            "val_loss", "mIoU", "mF1", "mIoU_no_background", "mF1_no_background",
            "DAMAGE_MACRO_MIOU", "DAMAGE_MACRO_MF1",
        ],
        default="val_loss",
    )
    parser.add_argument(
        "--log-online-train-metrics",
        action="store_true",
        help=(
            "Accumulate per-class train metrics from optimization forward passes. "
            "No additional inference pass is run; metrics are diagnostic only."
        ),
    )
    parser.add_argument(
        "--train-probe-split",
        type=Path,
        default=None,
        help=(
            "Optional MMSeg split .txt listing a small fixed clean-train probe. "
            "The probe is evaluated after each epoch for visualization only."
        ),
    )
    parser.add_argument("--train-probe-batch-size", type=int, default=32)
    parser.add_argument("--train-probe-fixed-count", type=int, default=20)
    parser.add_argument("--train-probe-top-k", type=int, default=20)
    parser.add_argument(
        "--train-probe-min-damage-ratio",
        type=float,
        default=0.01,
        help="Minimum GT fraction of damage-class pixels for best/worst probe ranking.",
    )
    parser.add_argument("--train-probe-seed", type=int, default=42)
    return parser.parse_args()


def read_tile_manifest(path: Path) -> List[Dict[str, str]]:
    required = {
        "source_id", "source_image", "source_mask", "original_height", "original_width",
        "x", "y", "content_height", "content_width", "image_path",
    }
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError("Validation tile manifest is missing columns: {}".format(sorted(missing)))
        rows = [{str(key): (value or "") for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError("Validation tile manifest is empty: {}".format(path))
    return rows


def group_tiles(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    for source_rows in grouped.values():
        source_rows.sort(key=lambda row: int(row["tile_idx"]))
    return dict(grouped)


def build_loaders_and_val_dataset(
    train_cfg_path: str, val_cfg_path: str, batch_size: int, workers: int
):
    train_cfg = mmcv.Config.fromfile(train_cfg_path)
    train_dataset = build_dataset(train_cfg.data.train)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=workers,
        dist=False,
        shuffle=True,
        persistent_workers=workers > 0,
        pin_memory=True,
    )
    val_cfg = mmcv.Config.fromfile(val_cfg_path)
    val_dataset = build_dataset(val_cfg.data.val, dict(test_mode=False))
    return train_loader, val_dataset


def build_train_probe_dataset(train_cfg_path: str, split_path: Optional[Path]):
    if split_path is None:
        return None
    if not split_path.is_file():
        raise FileNotFoundError("Train probe split file does not exist: {}".format(split_path))
    train_cfg = mmcv.Config.fromfile(train_cfg_path)
    probe_cfg = train_cfg.data.train.copy()
    probe_cfg["split"] = str(split_path)
    return build_dataset(probe_cfg, dict(test_mode=False))


def tile_dataset_index(val_dataset) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for position, info in enumerate(val_dataset.img_infos):
        filename = info.get("filename") or info.get("img_path")
        name = Path(filename).name
        if name in index:
            raise ValueError("Duplicate validation tile filename in dataset: {}".format(name))
        index[name] = position
    return index


def source_target(mask_path: Path) -> torch.Tensor:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError("Could not read full-resolution validation mask: {}".format(mask_path))
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(mask.astype(np.int64))


def add_task_metrics(metrics: dict, class_names: Sequence[str]) -> dict:
    class_metrics = metrics.get("class_metrics", {})
    foreground_names = [name for name in class_names if name != "BACKGROUND" and name in class_metrics]
    if foreground_names:
        metrics["mIoU_no_background"] = float(np.mean([class_metrics[name]["iou"] for name in foreground_names]))
        metrics["mF1_no_background"] = float(np.mean([class_metrics[name]["f1"] for name in foreground_names]))
        metrics["mAcc_no_background"] = float(np.mean([class_metrics[name]["accuracy"] for name in foreground_names]))

    damage_names = [name for name in DAMAGE_CLASS_NAMES if name in class_metrics]
    if damage_names:
        metrics["DAMAGE_MACRO_MIOU"] = float(np.mean([class_metrics[name]["iou"] for name in damage_names]))
        metrics["DAMAGE_MACRO_MF1"] = float(np.mean([class_metrics[name]["f1"] for name in damage_names]))
        metrics["DAMAGE_MACRO_MACC"] = float(np.mean([class_metrics[name]["accuracy"] for name in damage_names]))
        metrics["damage_macro_classes"] = damage_names
    return metrics


def log_metric_breakdown(logger, title: str, metrics: dict, class_names: Sequence[str]) -> None:
    logger.info(
        "%s summary — mIoU=%.4f | mF1=%.4f | mIoU_no_background=%.4f | "
        "DAMAGE_MACRO_MIOU=%.4f | DAMAGE_MACRO_MF1=%.4f",
        title,
        metrics.get("mIoU", 0.0),
        metrics.get("mF1", 0.0),
        metrics.get("mIoU_no_background", 0.0),
        metrics.get("DAMAGE_MACRO_MIOU", 0.0),
        metrics.get("DAMAGE_MACRO_MF1", 0.0),
    )
    logger.info("%s per-class metrics:", title)
    class_metrics = metrics.get("class_metrics", {})
    for name in class_names:
        values = class_metrics.get(name, {})
        logger.info(
            "  %-18s IoU=%.4f | F1=%.4f | Acc=%.4f",
            name,
            values.get("iou", 0.0),
            values.get("f1", 0.0),
            values.get("accuracy", 0.0),
        )
    logger.info("%s grouped metrics:", title)
    for group_name, values in metrics.get("group_metrics", {}).items():
        logger.info(
            "  %-18s IoU=%.4f | F1=%.4f | Acc=%.4f",
            group_name,
            values.get("iou", 0.0),
            values.get("f1", 0.0),
            values.get("accuracy", 0.0),
        )


def save_stitched_visualization(
    source_image: Path,
    source_mask: Path,
    prediction: np.ndarray,
    palette,
    class_names,
    output_path: Path,
    ignore_index: int,
) -> None:
    image = mmcv.imread(str(source_image))
    gt = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise ValueError("Could not read visualization GT: {}".format(source_mask))
    if gt.ndim == 3:
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    if image.shape[:2] != prediction.shape or gt.shape != prediction.shape:
        raise ValueError("Visualization shape mismatch for {}".format(source_image))
    gt_overlay = common.overlay_mask(image, gt, palette, ignore_index=ignore_index)
    pred_overlay = common.overlay_mask(image, prediction, palette, ignore_index=ignore_index)
    canvas = np.concatenate([image, gt_overlay, pred_overlay], axis=1)
    canvas = common.draw_legend(canvas, class_names, palette)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mmcv.imwrite(canvas, str(output_path))


def resolve_dataset_image_path(dataset, idx: int) -> Path:
    info = dataset.img_infos[idx]
    filename = info.get("filename") or info.get("img_path")
    path = Path(filename)
    if path.is_absolute() and path.is_file():
        return path
    candidate = Path(dataset.img_dir) / path
    if candidate.is_file():
        return candidate
    candidate = Path(dataset.img_dir) / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("Could not resolve dataset image at index {}: {}".format(idx, filename))


def save_probe_visualization(
    dataset,
    idx: int,
    prediction: np.ndarray,
    palette,
    class_names,
    output_path: Path,
    ignore_index: int,
    title: str,
) -> None:
    image_path = resolve_dataset_image_path(dataset, idx)
    image = mmcv.imread(str(image_path))
    target = dataset.get_gt_seg_map_by_idx(idx)
    if target.ndim == 3:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    gt_overlay = common.overlay_mask(image, target, palette, ignore_index=ignore_index)
    pred_overlay = common.overlay_mask(image, prediction, palette, ignore_index=ignore_index)
    canvas = np.concatenate([image, gt_overlay, pred_overlay], axis=1)
    canvas = common.draw_legend(canvas, class_names, palette)
    cv2.putText(canvas, title, (10, canvas.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mmcv.imwrite(canvas, str(output_path))


def damage_binary_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    damage_indices: Sequence[int],
    ignore_index: int,
) -> Tuple[float, float]:
    valid = target != ignore_index
    denominator = int(valid.sum())
    if denominator == 0:
        return 0.0, 0.0
    gt_damage = np.isin(target, damage_indices) & valid
    pred_damage = np.isin(prediction, damage_indices) & valid
    intersection = int(np.logical_and(gt_damage, pred_damage).sum())
    union = int(np.logical_or(gt_damage, pred_damage).sum())
    gt_ratio = float(gt_damage.sum() / denominator)
    score = float(intersection / union) if union > 0 else 0.0
    return score, gt_ratio


@torch.no_grad()
def evaluate_train_probe(
    model: nn.Module,
    probe_dataset,
    class_names: List[str],
    ignore_index: int,
    output_dir: Path,
    batch_size: int,
    fixed_indices: Sequence[int],
    top_k: int,
    min_damage_ratio: float,
) -> dict:
    model.eval()
    device = next(model.parameters()).device
    num_classes = len(class_names)
    palette = probe_dataset.PALETTE
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    damage_indices = [name_to_idx[name] for name in DAMAGE_CLASS_NAMES if name in name_to_idx]
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    results: List[Dict[str, object]] = []

    for start in tqdm(range(0, len(probe_dataset), batch_size), desc="Clean train probe"):
        indices = list(range(start, min(start + batch_size, len(probe_dataset))))
        samples = [probe_dataset[idx] for idx in indices]
        images = []
        for sample in samples:
            tensor = sample["img"].data
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            images.append(tensor)
        image_batch = torch.stack(images, dim=0).to(device)
        logits, _ = model(image_batch)
        logits = F.interpolate(logits, size=image_batch.shape[-2:], mode="bilinear", align_corners=False)
        predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        for local_idx, dataset_idx in enumerate(indices):
            target_np = probe_dataset.get_gt_seg_map_by_idx(dataset_idx)
            if target_np.ndim == 3:
                target_np = cv2.cvtColor(target_np, cv2.COLOR_BGR2GRAY)
            target = torch.from_numpy(target_np.astype(np.int64))
            prediction = torch.from_numpy(predictions[local_idx].astype(np.int64))
            confusion = common._accumulate_confusion(
                confusion, prediction, target, num_classes=num_classes, ignore_index=ignore_index
            )
            damage_iou, damage_ratio = damage_binary_iou(
                predictions[local_idx], target_np, damage_indices, ignore_index
            )
            results.append(
                {
                    "idx": dataset_idx,
                    "damage_union_iou": damage_iou,
                    "gt_damage_ratio": damage_ratio,
                    "prediction": predictions[local_idx],
                    "image_name": resolve_dataset_image_path(probe_dataset, dataset_idx).name,
                }
            )

    metrics = add_task_metrics(common._compute_metrics_from_confusion(confusion, class_names), class_names)
    result_by_idx = {int(item["idx"]): item for item in results}

    for idx in fixed_indices:
        item = result_by_idx[idx]
        save_probe_visualization(
            probe_dataset, idx, item["prediction"], palette, class_names,
            output_dir / "fixed" / "{:04d}_{}".format(idx, item["image_name"]),
            ignore_index,
            "fixed | damage_union_iou={:.4f} | gt_damage_ratio={:.4f}".format(
                item["damage_union_iou"], item["gt_damage_ratio"]
            ),
        )

    eligible = [item for item in results if float(item["gt_damage_ratio"]) >= min_damage_ratio]
    eligible.sort(key=lambda item: float(item["damage_union_iou"]), reverse=True)
    effective_k = min(top_k, len(eligible))
    best = eligible[:effective_k]
    worst = list(reversed(eligible[-effective_k:])) if effective_k else []
    for rank, item in enumerate(best, start=1):
        save_probe_visualization(
            probe_dataset, int(item["idx"]), item["prediction"], palette, class_names,
            output_dir / "best_damage" / "{:02d}_{}".format(rank, item["image_name"]),
            ignore_index,
            "best_damage rank={} | iou={:.4f} | gt_ratio={:.4f}".format(
                rank, item["damage_union_iou"], item["gt_damage_ratio"]
            ),
        )
    for rank, item in enumerate(worst, start=1):
        save_probe_visualization(
            probe_dataset, int(item["idx"]), item["prediction"], palette, class_names,
            output_dir / "worst_damage" / "{:02d}_{}".format(rank, item["image_name"]),
            ignore_index,
            "worst_damage rank={} | iou={:.4f} | gt_ratio={:.4f}".format(
                rank, item["damage_union_iou"], item["gt_damage_ratio"]
            ),
        )

    report = {
        "n_probe_tiles": len(probe_dataset),
        "n_damage_positive_ranked_tiles": len(eligible),
        "min_gt_damage_ratio_for_ranking": min_damage_ratio,
        "fixed_indices": list(fixed_indices),
        "top_k_requested": top_k,
        "metrics": metrics,
        "best_damage": [
            {"idx": int(item["idx"]), "image_name": item["image_name"],
             "damage_union_iou": float(item["damage_union_iou"]),
             "gt_damage_ratio": float(item["gt_damage_ratio"])}
            for item in best
        ],
        "worst_damage": [
            {"idx": int(item["idx"]), "image_name": item["image_name"],
             "damage_union_iou": float(item["damage_union_iou"]),
             "gt_damage_ratio": float(item["gt_damage_ratio"])}
            for item in worst
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


@torch.no_grad()
def evaluate_stitched_tiles(
    model: nn.Module,
    val_dataset,
    tiles_manifest: Path,
    class_names: List[str],
    loss_cfg: Dict,
    class_weights: Optional[torch.Tensor],
    ignore_index: int,
    viz_dir: Optional[Path] = None,
    max_visualizations: int = 10,
) -> Tuple[dict, float]:
    model.eval()
    device = next(model.parameters()).device
    num_classes = len(class_names)
    palette = val_dataset.PALETTE
    tile_index = tile_dataset_index(val_dataset)
    grouped = group_tiles(read_tile_manifest(tiles_manifest))
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    loss_values: List[float] = []
    first_prediction_dist = None

    source_ids = sorted(grouped)
    for source_number, source_id in enumerate(tqdm(source_ids, desc="Stitched validation")):
        rows = grouped[source_id]
        height = int(rows[0]["original_height"])
        width = int(rows[0]["original_width"])
        source_image = Path(rows[0]["source_image"])
        source_mask = Path(rows[0]["source_mask"])
        logits_sum = torch.zeros((num_classes, height, width), dtype=torch.float32)
        coverage = torch.zeros((height, width), dtype=torch.float32)

        for row in rows:
            tile_name = Path(row["image_path"]).name
            if tile_name not in tile_index:
                raise KeyError("Tile from manifest is absent from val dataset: {}".format(tile_name))
            sample = val_dataset[tile_index[tile_name]]
            image_tensor = sample["img"].data
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)
            logits, _ = model(image_tensor)
            logits = F.interpolate(logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
            logits = logits[0].detach().cpu()

            x = int(row["x"])
            y = int(row["y"])
            content_h = int(row["content_height"])
            content_w = int(row["content_width"])
            logits_sum[:, y:y + content_h, x:x + content_w] += logits[:, :content_h, :content_w]
            coverage[y:y + content_h, x:x + content_w] += 1.0

        if float(coverage.min()) < 1.0:
            raise RuntimeError("Stitched validation has uncovered pixels in source image: {}".format(source_id))
        stitched_logits = logits_sum / coverage.unsqueeze(0)
        target = source_target(source_mask)
        if tuple(target.shape) != (height, width):
            raise ValueError("Manifest/full mask size mismatch for source: {}".format(source_id))

        cpu_weights = class_weights.detach().cpu() if class_weights is not None else None
        loss = common.compute_loss(
            stitched_logits.unsqueeze(0), target.unsqueeze(0), loss_cfg,
            class_weights=cpu_weights, ignore_index=ignore_index,
        )
        loss_values.append(float(loss.item()))
        prediction = stitched_logits.argmax(dim=0)
        confusion = common._accumulate_confusion(
            confusion, prediction, target, num_classes=num_classes, ignore_index=ignore_index
        )
        if first_prediction_dist is None:
            values, counts = np.unique(prediction.numpy(), return_counts=True)
            first_prediction_dist = dict(zip(values.tolist(), counts.tolist()))

        if viz_dir is not None and (max_visualizations < 0 or source_number < max_visualizations):
            save_stitched_visualization(
                source_image, source_mask, prediction.numpy(), palette, class_names,
                viz_dir / "{}_stitched.png".format(source_id), ignore_index,
            )

    metrics = add_task_metrics(common._compute_metrics_from_confusion(confusion, class_names), class_names)
    val_loss = sum(loss_values) / max(len(loss_values), 1)
    if first_prediction_dist is not None:
        print("[VAL DEBUG] First stitched source class dist:", first_prediction_dist)
    return metrics, val_loss


def selection_score(metrics: dict, val_loss: float, mode: str) -> Tuple[str, float]:
    if mode == "val_loss":
        return "val_loss", float(val_loss)
    metric = float(metrics.get(mode, 0.0))
    # common.save_checkpoint retains the smallest score, hence negate metrics.
    return "neg_{}".format(mode), -metric


def main() -> None:
    args = parse_args()
    if args.train_probe_batch_size <= 0 or args.train_probe_fixed_count < 0 or args.train_probe_top_k < 0:
        raise ValueError("Invalid train-probe batch/fixed/top-k value.")
    if not 0.0 <= args.train_probe_min_damage_ratio <= 1.0:
        raise ValueError("train-probe-min-damage-ratio must be in [0, 1].")

    config_dir = PROJECT_ROOT / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=args.config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = "finetune_tiled_{}_lr{}_depth{}_bs{}".format(
        timestamp, args.learning_rate, args.unfreeze_depth, args.batch_size
    )
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(cfg)
    logger.info("Starting fine-tuning with stitched tiled validation")
    logger.info("Validation tiles manifest: %s", args.val_tiles_manifest)
    logger.info("Online train metrics enabled: %s", args.log_online_train_metrics)

    train_loader, val_dataset = build_loaders_and_val_dataset(
        args.train_dataset_config, args.val_dataset_config, args.batch_size, args.num_workers
    )
    probe_dataset = build_train_probe_dataset(args.train_dataset_config, args.train_probe_split)
    if probe_dataset is not None:
        logger.info("Clean train probe enabled: %d tiles from %s", len(probe_dataset), args.train_probe_split)

    class_names = list(train_loader.dataset.CLASSES)
    num_classes = len(class_names)
    common.validate_class_coverage(train_loader.dataset, val_dataset, num_classes, logger)
    if probe_dataset is not None:
        common.validate_class_coverage(train_loader.dataset, probe_dataset, num_classes, logger)
    ignore_index = getattr(train_loader.dataset, "ignore_index", 255)
    logger.info("Class names (index -> name): %s", {index: name for index, name in enumerate(class_names)})
    logger.info("Ignore index: %d", ignore_index)
    logger.info("Damage macro classes: %s", DAMAGE_CLASS_NAMES)

    fixed_probe_indices: List[int] = []
    if probe_dataset is not None and args.train_probe_fixed_count > 0:
        rng = random.Random(args.train_probe_seed)
        fixed_probe_indices = sorted(
            rng.sample(range(len(probe_dataset)), min(args.train_probe_fixed_count, len(probe_dataset)))
        )

    if args.class_weights == "none":
        class_weights = None
    elif args.class_weights == "auto":
        class_weights = common.compute_class_weights(
            train_loader, num_classes, ignore_index, mode=args.class_weight_mode,
            class_names=class_names, verbose=True,
        ).cuda()
    else:
        parsed = [float(value.strip()) for value in args.class_weights.split(",") if value.strip()]
        if len(parsed) != num_classes:
            raise ValueError("Expected {} class weights, got {}".format(num_classes, len(parsed)))
        class_weights = torch.tensor(parsed, dtype=torch.float32, device="cuda")

    loss_cfg = {"mode": args.loss_mode, "dice_weight": args.dice_weight, "focal_gamma": args.focal_gamma}
    logger.info("Loss config: %s", loss_cfg)
    if class_weights is not None:
        logger.info("Class weights (%s): %s", args.class_weight_mode, class_weights.detach().cpu().tolist())

    base_model = build_model(cfg.model, class_names=class_names).cuda()
    trainable_stats = common.configure_trainable_layers(base_model, args.unfreeze_depth)
    logger.info("Trainable params by group after unfreeze depth=%d: %s", args.unfreeze_depth, trainable_stats)
    mixers: List[nn.Module] = []
    if args.use_embedding_mixer:
        channels = getattr(base_model.decode_head, "text_channels", 512)
        mixers.append(common.EmbeddingMixer(channels))
    wrapper = common.FineTuneWrapper(base_model, mixers=mixers, mix_strategy=args.mix_strategy).cuda()
    common.log_parameter_counts(wrapper, logger)
    optimizer = common.build_optimizer(
        wrapper, base_lr=args.learning_rate, weight_decay=args.weight_decay,
        backbone_lr_mult=args.backbone_lr_mult, logger=logger,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = common.build_scheduler(optimizer, total_steps=total_steps, warmup_steps=args.warmup_steps)
    best_checkpoints: List[Tuple[float, Path]] = []
    metrics_log_path = run_dir / "metrics.jsonl"

    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        total_loss = 0.0
        loss_window = deque(maxlen=100)
        train_confusion = (
            torch.zeros((num_classes, num_classes), dtype=torch.float64, device="cuda")
            if args.log_online_train_metrics else None
        )
        progress = tqdm(train_loader, desc="Epoch {}".format(epoch))
        for data in progress:
            images = data["img"].data[0].cuda()
            targets = data["gt_semantic_seg"].data[0].long().squeeze(1).cuda()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = wrapper(images)
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            loss = common.compute_loss(logits, targets, loss_cfg, class_weights, ignore_index=ignore_index)
            if train_confusion is not None:
                predictions = logits.detach().argmax(dim=1)
                train_confusion = common._accumulate_confusion(
                    train_confusion, predictions, targets, num_classes=num_classes, ignore_index=ignore_index
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.item())
            loss_window.append(loss_value)
            total_loss += loss_value * images.size(0)
            progress.set_postfix(
                loss="{:.4f}".format(loss_value),
                avg100="{:.4f}".format(sum(loss_window) / len(loss_window)),
                lr="{:.2e}".format(optimizer.param_groups[0]["lr"]),
            )

        train_loss = total_loss / len(train_loader.dataset)
        online_train_metrics = None
        if train_confusion is not None:
            online_train_metrics = add_task_metrics(
                common._compute_metrics_from_confusion(train_confusion, class_names), class_names
            )
            log_metric_breakdown(logger, "Epoch {} online train".format(epoch), online_train_metrics, class_names)

        probe_metrics = None
        if probe_dataset is not None:
            probe_dir = run_dir / "train_viz" / "epoch_{:03d}".format(epoch)
            probe_metrics = evaluate_train_probe(
                wrapper, probe_dataset, class_names, ignore_index, probe_dir,
                batch_size=args.train_probe_batch_size,
                fixed_indices=fixed_probe_indices,
                top_k=args.train_probe_top_k,
                min_damage_ratio=args.train_probe_min_damage_ratio,
            )
            log_metric_breakdown(logger, "Epoch {} clean train probe".format(epoch), probe_metrics, class_names)

        viz_dir = run_dir / "val_viz" / "epoch_{:03d}".format(epoch)
        metrics, val_loss = evaluate_stitched_tiles(
            wrapper, val_dataset, args.val_tiles_manifest, class_names, loss_cfg,
            class_weights, ignore_index, viz_dir=viz_dir,
            max_visualizations=args.val_save_visualizations,
        )
        logger.info(
            "Epoch %d — train_loss: %.4f | stitched val mIoU: %.4f | "
            "mIoU_no_background: %.4f | DAMAGE_MACRO_MIOU: %.4f | mF1: %.4f | val_loss: %.4f",
            epoch, train_loss, metrics.get("mIoU", 0.0), metrics.get("mIoU_no_background", 0.0),
            metrics.get("DAMAGE_MACRO_MIOU", 0.0), metrics.get("mF1", 0.0), val_loss,
        )
        log_metric_breakdown(logger, "Epoch {} stitched val".format(epoch), metrics, class_names)
        score_name, score_value = selection_score(metrics, val_loss, args.select_best_by)
        best_checkpoints = common.save_checkpoint(
            model=wrapper, optimizer=optimizer, out_dir=run_dir, epoch=epoch, metrics=metrics,
            train_loss=train_loss, val_loss=val_loss, score_name=score_name,
            score_value=score_value, best_pool=best_checkpoints,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": metrics,
            "online_train_metrics": online_train_metrics,
            "clean_train_probe_metrics": probe_metrics,
            "selection_score_name": score_name,
            "selection_score_value": score_value,
        }
        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Training with stitched tiled validation finished")


if __name__ == "__main__":
    main()
