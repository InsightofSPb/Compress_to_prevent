"""
Single-GPU fine-tuning script for LPOSS/MaskCLIP models.
"""

import argparse
import datetime
import json
import sys
import os
import cv2
import random
import math
from pathlib import Path
from typing import List, Optional, Tuple
from collections import deque

import mmcv  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from mmseg.datasets import build_dataloader, build_dataset
from tqdm import tqdm
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.logger import get_logger
from models import build_model


# ===================== Metric groups =====================
STRUCTURAL_DAMAGE = {"CRACK", "SPALLING", "DELAMINATION", "MISSING_ELEMENT"}
SURFACE_STAIN = {"WATER_STAIN", "EFFLORESCENCE", "CORROSION"}
HUMAN_ACTIVITY = {"TEXT_OR_IMAGES", "REPAIRS"}


# ============================================================
#                       MODEL WRAPPERS
# ============================================================

class IdentityHead(nn.Module):
    def forward(self, feats: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        return feats


class EmbeddingMixer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 256):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, feats: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        mixed = self.proj_out(self.act(self.proj_in(feats)))
        return feats + mixed


class FineTuneWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        mixers: Optional[List[nn.Module]] = None,
        mix_strategy: str = "add",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.mixers = nn.ModuleList(mixers or [IdentityHead()])
        self.mix_strategy = mix_strategy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        backbone = getattr(self.base_model, "clip_backbone", None) or self.base_model

        try:
            logits, feats = backbone(x, return_feat=True)
        except TypeError:
            output = backbone(x)
            if isinstance(output, tuple):
                logits, feats = output[0], output[1] if len(output) > 1 else None
            else:
                logits, feats = output, None

        if feats is None:
            feats = logits

        mixed_feats = feats
        for head in self.mixers:
            mixed_feats = head(mixed_feats, logits)

        decode_head = getattr(self.base_model, "decode_head", None)
        if decode_head is None and hasattr(self.base_model, "clip_backbone"):
            decode_head = getattr(self.base_model.clip_backbone, "decode_head", None)

        if decode_head is None:
            raise AttributeError("No decode head found for fine-tuning")

        mixed_logits = decode_head.cls_seg(mixed_feats)

        if self.mix_strategy == "replace":
            combined = mixed_logits
        elif self.mix_strategy == "concat":
            combined = torch.stack([logits, mixed_logits], dim=0).mean(dim=0)
        else:
            combined = (logits + mixed_logits) / 2

        return combined, {"base_logits": logits, "mixed_logits": mixed_logits}


# ============================================================
#                       FREEZING
# ============================================================

def _set_requires_grad(module: nn.Module, is_trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = is_trainable


def _apply_unfreeze_depth(blocks: nn.ModuleList, depth: int) -> None:
    if depth == -1:
        _set_requires_grad(blocks, True)
        return

    _set_requires_grad(blocks, False)
    if depth <= 0:
        return

    for block in blocks[-depth:]:
        _set_requires_grad(block, True)


def _resolve_maskclip_components(model: nn.Module) -> Tuple[Optional[nn.Module], Optional[nn.ModuleList], Optional[nn.Module]]:
    clip_model = getattr(model, "clip_backbone", None)
    if clip_model is None and hasattr(model, "backbone"):
        clip_model = model

    if clip_model is None:
        return None, None, None

    visual_backbone = getattr(clip_model, "backbone", None)
    if visual_backbone is None and hasattr(clip_model, "visual"):
        visual_backbone = clip_model

    resblocks = None
    if (
        visual_backbone is not None
        and hasattr(visual_backbone, "visual")
        and hasattr(visual_backbone.visual, "transformer")
        and hasattr(visual_backbone.visual.transformer, "resblocks")
    ):
        resblocks = visual_backbone.visual.transformer.resblocks

    decode_head = getattr(clip_model, "decode_head", None)
    return clip_model, resblocks, decode_head


def configure_trainable_layers(model: nn.Module, depth: int) -> Dict[str, int]:
    trainable_stats = {"maskclip_backbone": 0, "head": 0, "other": 0}

    _set_requires_grad(model, False)

    _, resblocks, decode_head = _resolve_maskclip_components(model)
    if resblocks is not None:
        _apply_unfreeze_depth(resblocks, depth)

    if decode_head is not None:
        _set_requires_grad(decode_head, True)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "decode_head" in name:
            trainable_stats["head"] += param.numel()
        elif "backbone.visual" in name:
            trainable_stats["maskclip_backbone"] += param.numel()
        else:
            trainable_stats["other"] += param.numel()

    return trainable_stats


# ============================================================
#                       DATALOADERS
# ============================================================

def build_dataloaders(train_cfg: str, batch_size: int, workers: int, val_cfg: Optional[str] = None):
    train_cfg = mmcv.Config.fromfile(train_cfg)
    train_dataset = build_dataset(train_cfg.data.train)

    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=workers,
        dist=False,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )

    if val_cfg:
        val_cfg_obj = mmcv.Config.fromfile(val_cfg)
        val_dataset = build_dataset(val_cfg_obj.data.val, dict(test_mode=False))

        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=workers,
            dist=False,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True,
        )
    else:
        val_loader = None
    return train_loader, val_loader


# ============================================================
#                       LOSS + CLASS WEIGHTS
# ============================================================

def _sanitize_targets(target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    if target.min() < 0 or target.max() >= num_classes:
        target = target.clone()
        invalid_mask = (target < 0) | (target >= num_classes)
        target[invalid_mask] = ignore_index
    return target


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    valid_mask = (target != ignore_index).float()
    safe_target = target.clone()
    safe_target[target == ignore_index] = 0
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1)
    probs = probs * valid_mask
    one_hot = one_hot * valid_mask

    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    gamma: float,
    ignore_index: int,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index, reduction="none")
    valid_mask = (target != ignore_index)
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    return focal[valid_mask].mean() if valid_mask.any() else ce.new_tensor(0.0)


def compute_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: Dict,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

    num_classes = logits.shape[1]
    target = _sanitize_targets(target, num_classes, ignore_index)

    mode = loss_cfg.get("mode", "ce_weighted")
    dice_weight = float(loss_cfg.get("dice_weight", 1.0))

    if mode == "ce_weighted":
        return F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index)

    if mode == "ce_dice":
        ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index)
        dice = multiclass_dice_loss(logits, target, ignore_index=ignore_index)
        return ce + dice_weight * dice

    if mode == "focal_dice":
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        fl = focal_loss(logits, target, class_weights, gamma=gamma, ignore_index=ignore_index)
        dice = multiclass_dice_loss(logits, target, ignore_index=ignore_index)
        return fl + dice_weight * dice

    raise ValueError(f"Unsupported loss.mode: {mode}")


def compute_class_weights(loader, num_classes: int, ignore_index: int = 255, mode: str = "inverse") -> torch.Tensor:
    pixel_counts = torch.zeros(num_classes, dtype=torch.float64)

    for data in loader:
        gt = data["gt_semantic_seg"].data[0].squeeze(1)
        for c in range(num_classes):
            pixel_counts[c] += (gt == c).sum().item()

    if 0 <= ignore_index < num_classes:
        pixel_counts[ignore_index] = 0.0

    counts = pixel_counts + 1.0
    if mode == "median_freq":
        freq = counts / counts.sum()
        med = torch.median(freq)
        weights = med / freq
    elif mode == "effective_num":
        beta = 0.999
        effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=counts.dtype), counts)
        weights = (1.0 - beta) / effective_num
    else:
        weights = 1.0 / counts

    if num_classes > 0:
        weights[0] = 1.0
    weights = weights / weights.mean()
    return weights.float()


def validate_class_coverage(
    train_dataset, val_dataset, num_classes: int, logger
) -> None:
    """Validate that all stages see the same class layout, including background."""

    train_classes = list(getattr(train_dataset, "CLASSES", []))
    if len(train_classes) != num_classes:
        raise ValueError(
            f"Train dataset reports {len(train_classes)} classes,"
            f" but {num_classes} were inferred."
        )

    if val_dataset is not None:
        val_classes = list(getattr(val_dataset, "CLASSES", []))
        if train_classes != val_classes:
            raise ValueError(
                "Train/val class lists differ; all classes (including background) must match."
            )

        val_ignore = getattr(val_dataset, "ignore_index", 255)
        train_ignore = getattr(train_dataset, "ignore_index", 255)
        if val_ignore != train_ignore:
            logger.warning(
                "Train ignore_index=%s but val ignore_index=%s; ensure background/ignored pixels align.",
                train_ignore,
                val_ignore,
            )


def log_parameter_counts(model: nn.Module, logger) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model parameters: total=%d, trainable=%d, frozen=%d",
        total_params,
        trainable_params,
        total_params - trainable_params,
    )


def build_optimizer(
    wrapper: nn.Module,
    base_lr: float,
    weight_decay: float,
    backbone_lr_mult: float,
    logger,
):
    backbone_params, head_params, other_params = [], [], []

    for name, param in wrapper.named_parameters():
        if not param.requires_grad:
            continue
        if "base_model.backbone.visual" in name:
            backbone_params.append(param)
        elif "base_model.decode_head" in name or "mixers" in name:
            head_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"name": "maskclip_backbone", "params": backbone_params, "lr": base_lr * backbone_lr_mult})
    if head_params:
        param_groups.append({"name": "head", "params": head_params, "lr": base_lr})
    if other_params:
        param_groups.append({"name": "other", "params": other_params, "lr": base_lr})

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)

    for idx, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group["params"])
        logger.info("Optimizer group %d (%s): lr=%g params=%d", idx, group.get("name", "unnamed"), group["lr"], num_params)

    return optimizer


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    warmup_steps = max(0, warmup_steps)

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ============================================================
#                       EVALUATION
# ============================================================

def _accumulate_confusion(confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    with torch.no_grad():
        preds = preds.view(-1)
        targets = targets.view(-1)

        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

        if preds.numel() == 0:
            return confusion

        combined = targets * num_classes + preds
        hist = torch.bincount(combined, minlength=num_classes * num_classes)
        confusion = confusion + hist.view(num_classes, num_classes)
    return confusion


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0)
    return result


def _compute_group_metrics(confusion: np.ndarray, group_indices: List[int]) -> dict:
    tp = confusion[group_indices, :][:, group_indices].sum()
    fp = confusion[:, group_indices].sum() - tp
    fn = confusion[group_indices, :].sum() - tp
    iou = _safe_divide(tp, tp + fp + fn)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    acc = _safe_divide(tp, tp + fn)
    return {"iou": float(iou), "f1": float(f1), "accuracy": float(acc)}


def _compute_metrics_from_confusion(confusion: torch.Tensor, class_names: List[str]) -> dict:
    confusion_np = confusion.cpu().numpy()
    true_positives = np.diag(confusion_np)
    false_positives = confusion_np.sum(axis=0) - true_positives
    false_negatives = confusion_np.sum(axis=1) - true_positives

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

    miou = float(np.nanmean(class_iou))
    mf1 = float(np.nanmean(class_f1))
    macc = float(np.nanmean(class_acc))

    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    group_indices = {
        "STRUCTURAL_DAMAGE": [name_to_idx[c] for c in STRUCTURAL_DAMAGE if c in name_to_idx],
        "SURFACE_STAIN": [name_to_idx[c] for c in SURFACE_STAIN if c in name_to_idx],
        "HUMAN_ACTIVITY": [name_to_idx[c] for c in HUMAN_ACTIVITY if c in name_to_idx],
    }

    group_metrics = {}
    for group_name, indices in group_indices.items():
        if not indices:
            continue
        group_metrics[group_name] = _compute_group_metrics(confusion_np, indices)

    combined_indices = sorted({idx for indices in group_indices.values() for idx in indices})
    if combined_indices:
        group_metrics["COMBINED_GROUPS"] = _compute_group_metrics(confusion_np, combined_indices)

    return {
        "mIoU": miou,
        "mF1": mf1,
        "mAcc": macc,
        "class_metrics": class_metrics,
        "group_metrics": group_metrics,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    class_names: List[str],
    loss_cfg: Dict,
    class_weights: Optional[torch.Tensor],
    ignore_index: int,
) -> Tuple[dict, float]:
    model.eval()
    num_classes = len(class_names)
    device = next(model.parameters()).device
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    total_loss = 0.0

    results = []
    for data in loader:
        imgs = data["img"].data[0].cuda()
        targets = data["gt_semantic_seg"].data[0].long().squeeze(1).cuda()

        logits, _ = model(imgs)

        target_shape = targets.shape[-2:]
        logits = F.interpolate(logits, size=target_shape, mode="bilinear", align_corners=False)

        loss = compute_loss(logits, targets, loss_cfg, class_weights, ignore_index=ignore_index)
        total_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(dim=1)
        confusion = _accumulate_confusion(confusion, preds, targets, num_classes, ignore_index)
        results.extend(list(preds.cpu().numpy()))

    metrics = _compute_metrics_from_confusion(confusion, class_names)
    val_loss = total_loss / len(loader.dataset)

    if results:
        u, c = np.unique(results[0], return_counts=True)
        print("[VAL DEBUG] First sample class dist:", dict(zip(u.tolist(), c.tolist())))

    return metrics, val_loss


# ============================================================
#                       VISUALIZATION
# ============================================================

def overlay_mask(image, mask, palette, ignore_index=255):
    overlay = image.copy()
    color_mask = np.zeros_like(image)

    for idx, color in enumerate(palette):
        color_mask[mask == idx] = color

    valid = mask != ignore_index
    overlay[valid] = (0.6 * overlay[valid] + 0.4 * color_mask[valid]).astype(np.uint8)
    return overlay


def draw_legend(img, classes, palette):
    y = 20
    for i, name in enumerate(classes):
        color = palette[i]
        cv2.rectangle(img, (10, y - 10), (30, y + 10), color, -1)
        cv2.putText(img, name, (40, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22
    return img


@torch.no_grad()
def save_val_visualizations(epoch, model, val_loader, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dataset = val_loader.dataset
    palette = dataset.PALETTE
    classes = dataset.CLASSES

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:10]

    model.eval()

    for idx in indices:
        data = dataset[idx]
        img_info = dataset.img_infos[idx]
        img_path = img_info.get("filename") or img_info.get("img_path")

        if not os.path.isabs(img_path):
            img_path = os.path.join(dataset.img_dir, os.path.basename(img_path))

        img = mmcv.imread(img_path)
        gt = dataset.get_gt_seg_map_by_idx(idx)

        img_tensor = data["img"].data[0]
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)
        if img_tensor.dim() == 3 and img_tensor.size(0) == 1:
            img_tensor = img_tensor.expand(3, -1, -1)

        logits, _ = model(img_tensor.unsqueeze(0).cuda())
        target_size = img.shape[:2]
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        gt_resized = mmcv.imresize(gt.astype(np.uint8), target_size[::-1], interpolation="nearest")

        gt_overlay = overlay_mask(img, gt_resized, palette)
        pred_overlay = overlay_mask(img, pred, palette)

        canvas = np.concatenate([img, gt_overlay, pred_overlay], axis=1)
        canvas = draw_legend(canvas, classes, palette)

        out_path = os.path.join(out_dir, f"epoch{epoch:03d}_idx{idx:03d}.png")
        mmcv.imwrite(canvas, out_path)


# ============================================================
#                       CHECKPOINT
# ============================================================

def save_checkpoint(model, optimizer, out_dir, epoch, metrics, average_loss, best_pool):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "average_loss": average_loss,
    }

    ckpt_path = out_dir / f"epoch_{epoch:04d}_loss_{average_loss:.4f}.pth"
    torch.save(checkpoint, ckpt_path)

    best_pool.append((average_loss, ckpt_path))
    best_pool.sort(key=lambda x: x[0])
    while len(best_pool) > 3:
        _, old_path = best_pool.pop()
        if old_path.exists():
            old_path.unlink()
    return best_pool


# ============================================================
#                       ARGS & MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU fine-tuning entrypoint")
    parser.add_argument("config")
    parser.add_argument("--train-dataset-config", required=True)
    parser.add_argument("--val-dataset-config")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
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
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = PROJECT_ROOT / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=args.config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"finetune_{timestamp}_lr{args.learning_rate}_depth{args.unfreeze_depth}_bs{args.batch_size}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(cfg)
    logger.info("Starting fine-tuning")

    train_loader, val_loader = build_dataloaders(
        args.train_dataset_config,
        args.batch_size,
        args.num_workers,
        args.val_dataset_config,
    )

    class_names = train_loader.dataset.CLASSES
    num_classes = len(class_names)

    validate_class_coverage(train_loader.dataset, val_loader.dataset if val_loader else None, num_classes, logger)

    # Здесь уже можно увидеть, "особенный" ли класс 0:
    logger.info("Class names (index -> name): %s", {i: n for i, n in enumerate(class_names)})

    ignore_index = getattr(train_loader.dataset, "ignore_index", 255)

    if args.class_weights == "none":
        class_weights = None
    elif args.class_weights == "auto":
        class_weights = compute_class_weights(
            train_loader,
            num_classes,
            ignore_index,
            mode=args.class_weight_mode,
        ).cuda()
    else:
        parsed = [float(x.strip()) for x in args.class_weights.split(",") if x.strip()]
        if len(parsed) != num_classes:
            raise ValueError(f"Expected {num_classes} class weights, got {len(parsed)}")
        class_weights = torch.tensor(parsed, dtype=torch.float32, device="cuda")

    loss_cfg = {
        "mode": args.loss_mode,
        "dice_weight": args.dice_weight,
        "focal_gamma": args.focal_gamma,
    }
    logger.info("Loss config: %s", loss_cfg)
    if class_weights is not None:
        logger.info("Class weights (%s): %s", args.class_weight_mode, class_weights.detach().cpu().tolist())

    base_model = build_model(cfg.model, class_names=class_names).cuda()
    trainable_stats = configure_trainable_layers(base_model, args.unfreeze_depth)
    logger.info("Trainable params by group after unfreeze depth=%d: %s", args.unfreeze_depth, trainable_stats)

    mixers: List[nn.Module] = []
    if args.use_embedding_mixer:
        channels = getattr(base_model.decode_head, "text_channels", 512)
        mixers.append(EmbeddingMixer(channels))

    wrapper = FineTuneWrapper(
        base_model, mixers=mixers, mix_strategy=args.mix_strategy
    ).cuda()

    log_parameter_counts(wrapper, logger)

    optimizer = build_optimizer(
        wrapper,
        base_lr=args.learning_rate,
        weight_decay=args.weight_decay,
        backbone_lr_mult=args.backbone_lr_mult,
        logger=logger,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=args.warmup_steps)

    best_checkpoints: List[Tuple[float, Path]] = []
    metrics_log_path = run_dir / "metrics.jsonl"

    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        total_loss = 0.0
        loss_window = deque(maxlen=100)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for data in progress:
            imgs = data["img"].data[0].cuda()
            targets = data["gt_semantic_seg"].data[0].long().squeeze(1).cuda()

            optimizer.zero_grad()
            logits, _ = wrapper(imgs)
            loss = compute_loss(logits, targets, loss_cfg, class_weights, ignore_index=ignore_index)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            loss_window.append(loss_value)
            avg_recent = sum(loss_window) / len(loss_window)

            total_loss += loss_value * imgs.size(0)

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                avg100=f"{avg_recent:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        avg_loss = total_loss / len(train_loader.dataset)
        logger.info("Epoch %d avg loss: %.4f", epoch, avg_loss)
        logger.info("Epoch %d LR groups: %s", epoch, [group["lr"] for group in optimizer.param_groups])

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
        }

        metrics = {}
        if val_loader:
            metrics, val_loss = evaluate(
                wrapper, val_loader, class_names, loss_cfg, class_weights, ignore_index
            )
            epoch_record.update({
                "val_loss": val_loss,
                "metrics": metrics,
            })
            logger.info(
                "Validation — mIoU: %.4f | mF1: %.4f | mAcc: %.4f | loss: %.4f",
                metrics.get("mIoU", 0.0),
                metrics.get("mF1", 0.0),
                metrics.get("mAcc", 0.0),
                val_loss,
            )

            viz_dir = run_dir / "val_viz" / f"epoch_{epoch:03d}"
            save_val_visualizations(epoch, wrapper, val_loader, viz_dir)

        best_checkpoints = save_checkpoint(
            wrapper, optimizer, run_dir, epoch, metrics, avg_loss, best_checkpoints
        )

        with metrics_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

    logger.info("Training finished")


if __name__ == "__main__":
    main()
