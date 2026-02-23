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
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def compute_class_weights(
    loader,
    num_classes: int,
    ignore_index: int = 255,
    mode: str = "inverse",
    background_scale: float = 1.0,
    clamp_max: float = 20.0,
    class_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> torch.Tensor:
    pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    ignored_pixels = 0

    for data in loader:
        gt = data["gt_semantic_seg"].data[0].squeeze(1).to(torch.int64)  # (B,H,W)
        flat_all = gt.reshape(-1)

        if ignore_index is not None:
            ignored_pixels += (flat_all == ignore_index).sum().item()
            flat = flat_all[flat_all != ignore_index]
        else:
            flat = flat_all

        flat = flat[(flat >= 0) & (flat < num_classes)]
        if flat.numel() == 0:
            continue

        pixel_counts += torch.bincount(flat, minlength=num_classes).to(torch.float64)

    labeled_total = pixel_counts.sum().item()
    if labeled_total <= 0:
        raise RuntimeError("No labeled pixels found while computing class weights.")

    counts = pixel_counts + 1.0  # smoothing to avoid div-by-zero

    if mode == "median_freq":
        freq = counts / counts.sum().clamp_min(1.0)
        med = torch.median(freq)
        weights = med / freq
    elif mode == "effective_num":
        beta = 0.999
        beta_t = torch.tensor(beta, dtype=counts.dtype, device=counts.device)
        effective_num = 1.0 - torch.pow(beta_t, counts)
        weights = (1.0 - beta) / effective_num.clamp_min(1e-12)
    else:  # "inverse"
        weights = 1.0 / counts

    # Optional: down-weight BACKGROUND=0 if you want
    if num_classes > 0 and background_scale != 1.0:
        weights[0] *= float(background_scale)

    # Only cap extreme large weights; do NOT force a lower bound
    weights = torch.clamp(weights, max=float(clamp_max))
    weights = weights / weights.mean().clamp_min(1e-12)

    if verbose:
        total_seen = labeled_total + float(ignored_pixels)

        print("\n[CLASS WEIGHTS DEBUG]")
        print(f"  num_classes      : {num_classes}")
        print(f"  ignore_index     : {ignore_index}")
        print(f"  total_pixels_seen: {int(total_seen)}")
        print(f"  labeled_pixels   : {int(labeled_total)}")
        print(f"  ignored_pixels   : {int(ignored_pixels)}")

        freqs = (pixel_counts / max(1.0, labeled_total)).cpu().numpy()
        w = weights.cpu().numpy()

        order = np.argsort(-pixel_counts.cpu().numpy())
        topk = min(11, num_classes)
        print(f"  top-{topk} classes by pixel count:")
        for i in order[:topk]:
            name = class_names[i] if class_names and i < len(class_names) else str(i)
            print(
                f"    {i:2d} {name:16s} "
                f"count={int(pixel_counts[i].item()):12d} freq={freqs[i]:.6f} weight={w[i]:.6f}"
            )

        print("  weights:", [float(x) for x in w.tolist()])
        print()

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

        # Head: decode_head + mixers
        if ("decode_head" in name) or ("mixers" in name):
            head_params.append(param)

        # Visual backbone: MaskCLIP/CLIP visual transformer blocks
        elif ("clip_backbone.backbone.visual" in name) or ("backbone.visual" in name) or ("visual.transformer" in name):
            backbone_params.append(param)

        else:
            other_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({
            "name": "maskclip_backbone",
            "params": backbone_params,
            "lr": base_lr * backbone_lr_mult,
        })
    if head_params:
        param_groups.append({
            "name": "head",
            "params": head_params,
            "lr": base_lr,
        })
    if other_params:
        param_groups.append({
            "name": "other",
            "params": other_params,
            "lr": base_lr,
        })

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)

    # --- logs + sanity ---
    total_trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    grouped = 0

    for idx, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group["params"])
        grouped += num_params
        logger.info("Optimizer group %d (%s): lr=%g params=%d",
                    idx, group.get("name", "unnamed"), group["lr"], num_params)

    if grouped != total_trainable:
        logger.warning("Grouped params (%d) != trainable params (%d). Something is off.",
                       grouped, total_trainable)

    if not backbone_params:
        logger.warning("No backbone params matched patterns. Check name substrings.")
    if not head_params:
        logger.warning("No head params matched patterns. Check name substrings.")

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
#                       YEAR RESOLUTION HELPERS
# ============================================================

# Keep this regex near the helpers section
_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def _try_parse_year(value: Any) -> Optional[int]:
    """
    Conservative parser for explicit year-like values.

    IMPORTANT:
    - For strings, this function intentionally does NOT search arbitrary embedded
      years inside long paths, to avoid false positives like:
      .../february_march_2026_data/...
    - Use `_extract_year_from_filename_patterns(...)` for filename-based parsing.
    """
    if value is None:
        return None

    if isinstance(value, (int, np.integer)):
        year = int(value)
        return year if 1900 <= year <= 2100 else None

    if isinstance(value, float):
        if float(value).is_integer():
            year = int(value)
            return year if 1900 <= year <= 2100 else None
        return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None

        # exact "2025"
        if s.isdigit() and len(s) == 4:
            year = int(s)
            return year if 1900 <= year <= 2100 else None

        # exact date-like strings e.g. "2025-02-23", "2025/02/23"
        m = re.match(r"^\s*(19\d{2}|20\d{2})[-/\.]\d{1,2}[-/\.]\d{1,2}\s*$", s)
        if m:
            return int(m.group(1))

        return None

    return None


def _extract_year_recursive(obj: Any) -> Optional[int]:
    """
    Safe recursive search for an explicitly stored year in nested metadata.
    Does NOT parse arbitrary filenames/paths for embedded years.
    """
    if obj is None:
        return None

    # direct scalar year/date
    y = _try_parse_year(obj)
    if y is not None:
        return y

    if isinstance(obj, dict):
        # Prefer explicit year-like keys
        preferred_keys = (
            "year", "img_year", "capture_year", "shot_year", "date_year",
            "timestamp_year"
        )
        for k in preferred_keys:
            if k in obj:
                y = _extract_year_recursive(obj.get(k))
                if y is not None:
                    return y

        # Common nested structures
        nested_keys = ("img_info", "ann_info", "meta", "metadata", "extra", "info")
        for k in nested_keys:
            if k in obj:
                y = _extract_year_recursive(obj.get(k))
                if y is not None:
                    return y

        # Fallback over dict values (still safe because _try_parse_year is conservative)
        for v in obj.values():
            y = _extract_year_recursive(v)
            if y is not None:
                return y
        return None

    if isinstance(obj, (list, tuple)):
        for v in obj:
            y = _extract_year_recursive(v)
            if y is not None:
                return y
        return None

    return None

def _extract_year_from_filename_patterns(path_value: Any) -> Optional[int]:
    """
    Parse year from filename / basename ONLY (not full path), using dataset-specific
    patterns and safe fallbacks.

    Supported patterns (from your split/augmentation logic):
    - ..._2025(.ext)               -> suffix year
    - PXL_2025....                 -> year in PXL name
    - photo_...2025-..-.._...      -> year from date in photo_* pattern
    - IMG_*                        -> hardcoded 2025 (your rule)
    - generic 19xx/20xx in basename (safe fallback)
    """
    if path_value is None:
        return None

    s = str(path_value).strip()
    if not s:
        return None

    # Use basename only to avoid false year from directories like ".../2026_data/..."
    base = os.path.basename(s.replace("\\", "/"))
    if not base:
        return None

    stem = Path(base).stem  # filename without extension

    # 1) Explicit suffix year: "..._2025"
    m = re.search(r"_(19\d{2}|20\d{2})$", stem)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return y

    # 2) PXL pattern: "PXL_2025...."
    m = re.search(r"(?i)\bPXL[_-]?(19\d{2}|20\d{2})\b", stem)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return y

    # 3) photo_*YYYY-MM-DD* pattern
    #    Example: photo_...2025-01-31_...
    m = re.search(r"(?i)\bphoto[_-].*?(19\d{2}|20\d{2})[-/\.]\d{1,2}[-/\.]\d{1,2}\b", stem)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return y

    # 4) IMG_* pattern -> hardcoded 2025 (your rule)
    #    Works for augmented/tiled names if they still contain token "IMG_"
    if re.search(r"(?i)(?:^|[_-])IMG[_-]", stem) or stem.upper().startswith("IMG_") or "_IMG_" in stem.upper():
        return 2025

    # 5) Safe generic fallback: any year in basename/stem (NOT full path)
    m = _YEAR_RE.search(stem)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return y

    # Also check with extension included (rarely useful, but harmless)
    m = _YEAR_RE.search(base)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return y

    return None

def _path_aliases(path_value: Any) -> List[str]:
    if path_value is None:
        return []
    s = str(path_value)
    if not s:
        return []

    aliases = set()
    aliases.add(s)
    aliases.add(s.replace("\\", "/"))
    aliases.add(os.path.normpath(s))
    aliases.add(os.path.normpath(s).replace("\\", "/"))

    try:
        p = Path(s)
        aliases.add(p.name)
        aliases.add(os.path.basename(s))
        # strict=False avoids failure if file doesn't exist
        aliases.add(str(p.resolve(strict=False)))
        aliases.add(str(p.resolve(strict=False)).replace("\\", "/"))
    except Exception:
        aliases.add(os.path.basename(s))

    # Also keep basename of normalized path
    aliases.add(os.path.basename(os.path.normpath(s)))

    return [a for a in aliases if isinstance(a, str) and a]


def _register_path_year(lookup: Dict[str, int], path_value: Any, year: int, img_dir: Optional[str] = None) -> None:
    if path_value is None:
        return

    for alias in _path_aliases(path_value):
        lookup[alias] = int(year)

    # If relative path and dataset has img_dir, register joined aliases too
    try:
        s = str(path_value)
        if img_dir and s and not os.path.isabs(s):
            joined = os.path.join(img_dir, s)
            for alias in _path_aliases(joined):
                lookup[alias] = int(year)
    except Exception:
        pass


def _collect_year_lookup_from_dataset(dataset, lookup: Dict[str, int]) -> None:
    if dataset is None:
        return

    # Handle wrappers like RepeatDataset / ConcatDataset
    if hasattr(dataset, "dataset") and getattr(dataset, "dataset") is not None:
        _collect_year_lookup_from_dataset(getattr(dataset, "dataset"), lookup)

    if hasattr(dataset, "datasets") and getattr(dataset, "datasets") is not None:
        for ds in getattr(dataset, "datasets"):
            _collect_year_lookup_from_dataset(ds, lookup)

    img_infos = getattr(dataset, "img_infos", None)
    img_dir = getattr(dataset, "img_dir", None)
    if not img_infos:
        return

    for info in img_infos:
        if not isinstance(info, dict):
            continue

        # Collect common path candidates first
        path_candidates: List[Any] = []
        for k in ("filename", "img_path", "file_name", "ori_filename", "path"):
            if k in info and info[k] is not None:
                path_candidates.append(info[k])

        nested_img_info = info.get("img_info")
        if isinstance(nested_img_info, dict):
            for k in ("filename", "img_path", "file_name", "ori_filename", "path"):
                if k in nested_img_info and nested_img_info[k] is not None:
                    path_candidates.append(nested_img_info[k])

        # 1) Prefer explicit year in metadata
        year = _extract_year_recursive(info)

        # 2) If no explicit year, parse from filename patterns (basename only)
        if year is None:
            for candidate in path_candidates:
                year = _extract_year_from_filename_patterns(candidate)
                if year is not None:
                    break

        if year is None:
            continue

        for candidate in path_candidates:
            _register_path_year(lookup, candidate, year, img_dir=img_dir)


def build_year_lookup(dataset) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    _collect_year_lookup_from_dataset(dataset, lookup)
    return lookup


def _extract_batch_metas(data: dict) -> List[dict]:
    img_metas = data.get("img_metas")
    if img_metas is None:
        return []

    # mmcv DataContainer -> .data[0] is a list of meta dicts for the batch
    try:
        metas = img_metas.data[0]
    except Exception:
        metas = img_metas

    if isinstance(metas, dict):
        return [metas]
    if isinstance(metas, (list, tuple)):
        return [m for m in metas if isinstance(m, dict)]
    return []


def _resolve_year_from_meta(meta: dict, year_lookup: Optional[Dict[str, int]] = None) -> Optional[int]:
    # 1) Explicit year in meta (best source, if Collect/meta_keys includes it)
    if isinstance(meta, dict) and "year" in meta:
        y = _extract_year_recursive({"year": meta.get("year")})
        if y is not None:
            return y

    # 2) Build path candidates from common meta keys
    path_candidates: List[Any] = []
    for k in ("filename", "ori_filename", "img_path", "path"):
        if k in meta and meta[k] is not None:
            path_candidates.append(meta[k])

    nested = meta.get("img_info")
    if isinstance(nested, dict):
        for k in ("filename", "ori_filename", "img_path", "path"):
            if k in nested and nested[k] is not None:
                path_candidates.append(nested[k])

    # 3) Lookup by exact aliases (best fallback)
    if year_lookup:
        for p in path_candidates:
            for alias in _path_aliases(p):
                if alias in year_lookup:
                    return int(year_lookup[alias])

    # 4) Parse from filename patterns (basename only, safe)
    for p in path_candidates:
        y = _extract_year_from_filename_patterns(p)
        if y is not None:
            return y

    # 5) Last safe attempt: recursive explicit-year search in meta (conservative only)
    #    (won't parse arbitrary embedded years in paths)
    return _extract_year_recursive(meta)
@torch.no_grad()
def build_train_tail_batch_summary(
    preds: torch.Tensor,
    targets: torch.Tensor,
    batch_loss_value: float,
    batch_size: int,
    class_names: List[str],
    ignore_index: int,
    batch_years: Optional[List[Optional[int]]] = None,
) -> dict:
    """
    Build compact metric summary for one training batch.
    Stores only confusion matrices + small counters (no full masks).

    Robust to shape mismatch between preds and targets (e.g., logits/preds from lower-res head):
    preds will be upsampled with nearest interpolation to target spatial size.
    """
    num_classes = len(class_names)

    # Move only what's needed to CPU for compact aggregation
    preds_cpu = preds.detach().to("cpu")
    targets_cpu = targets.detach().to("cpu")
    targets_cpu = _sanitize_targets(targets_cpu, num_classes, ignore_index)

    # Ensure dtypes
    if preds_cpu.dtype != torch.int64:
        preds_cpu = preds_cpu.to(torch.int64)
    if targets_cpu.dtype != torch.int64:
        targets_cpu = targets_cpu.to(torch.int64)

    # --- FIX: align spatial sizes before confusion accumulation ---
    # Expected shapes: preds [B,H,W], targets [B,H,W]
    # But preds may be [B,h,w] from non-upsampled logits.argmax(...)
    if preds_cpu.ndim != 3 or targets_cpu.ndim != 3:
        raise ValueError(
            f"Expected preds/targets to be 3D [B,H,W], got preds={tuple(preds_cpu.shape)}, "
            f"targets={tuple(targets_cpu.shape)}"
        )

    if preds_cpu.shape[0] != targets_cpu.shape[0]:
        raise ValueError(
            f"Batch mismatch in train-tail metrics: preds batch={preds_cpu.shape[0]} "
            f"vs targets batch={targets_cpu.shape[0]}"
        )

    if preds_cpu.shape[-2:] != targets_cpu.shape[-2:]:
        preds_cpu = F.interpolate(
            preds_cpu.unsqueeze(1).float(),
            size=targets_cpu.shape[-2:],
            mode="nearest",
        ).squeeze(1).to(torch.int64)

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device="cpu")
    confusion = _accumulate_confusion(confusion, preds_cpu, targets_cpu, num_classes, ignore_index)

    year_confusions: Dict[Union[int, str], torch.Tensor] = {}
    year_image_counts: Dict[Union[int, str], int] = {}
    unresolved_year_samples = 0

    if batch_years is None:
        batch_years = [None] * int(batch_size)

    year_confusions, year_image_counts, unresolved_year_samples = _accumulate_per_year_confusions(
        year_confusions=year_confusions,
        year_image_counts=year_image_counts,
        preds=preds_cpu,
        targets=targets_cpu,
        batch_years=batch_years,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    return {
        "confusion": confusion,  # CPU tensor
        "loss_sum": float(batch_loss_value) * int(batch_size),
        "num_samples": int(batch_size),
        "year_confusions": year_confusions,      # dict[key] -> CPU confusion tensor
        "year_image_counts": year_image_counts,  # dict[key] -> int
        "num_samples_unresolved_year": int(unresolved_year_samples),
    }


def aggregate_train_tail_summaries(
    summaries,
    class_names: List[str],
) -> Tuple[dict, float]:
    """
    Aggregate summaries for the last N training batches into the same metric format
    as validation (plus metrics_by_year).
    """
    summaries = list(summaries)
    if len(summaries) == 0:
        empty_conf = torch.zeros((len(class_names), len(class_names)), dtype=torch.float64)
        metrics = _compute_metrics_from_confusion(empty_conf, class_names)
        metrics["metrics_by_year"] = {}
        metrics["num_samples_unresolved_year"] = 0
        metrics["window_num_batches"] = 0
        metrics["window_num_samples"] = 0
        return metrics, 0.0

    num_classes = len(class_names)
    total_confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device="cpu")
    total_loss_sum = 0.0
    total_samples = 0
    unresolved_year_samples = 0

    year_confusions: Dict[Union[int, str], torch.Tensor] = {}
    year_image_counts: Dict[Union[int, str], int] = {}

    for s in summaries:
        total_confusion += s["confusion"]
        total_loss_sum += float(s.get("loss_sum", 0.0))
        total_samples += int(s.get("num_samples", 0))
        unresolved_year_samples += int(s.get("num_samples_unresolved_year", 0))

        # aggregate per-year confusions
        for key, conf in s.get("year_confusions", {}).items():
            if key not in year_confusions:
                year_confusions[key] = conf.clone()
            else:
                year_confusions[key] += conf

        # aggregate per-year image counts
        for key, n in s.get("year_image_counts", {}).items():
            year_image_counts[key] = int(year_image_counts.get(key, 0)) + int(n)

    metrics = _compute_metrics_from_confusion(total_confusion, class_names)
    metrics["metrics_by_year"] = _compute_metrics_by_year(year_confusions, year_image_counts, class_names)
    metrics["num_samples_unresolved_year"] = int(unresolved_year_samples)
    metrics["window_num_batches"] = int(len(summaries))
    metrics["window_num_samples"] = int(total_samples)

    avg_loss = total_loss_sum / max(1, total_samples)
    return metrics, float(avg_loss)

def resolve_batch_years(
    data: dict,
    batch_size: int,
    year_lookup: Optional[Dict[str, int]] = None,
) -> List[Optional[int]]:
    metas = _extract_batch_metas(data)
    years: List[Optional[int]] = []

    for meta in metas:
        years.append(_resolve_year_from_meta(meta, year_lookup=year_lookup))

    # Safety: keep same length as batch
    if len(years) < batch_size:
        years.extend([None] * (batch_size - len(years)))
    elif len(years) > batch_size:
        years = years[:batch_size]

    if len(years) == 0 and batch_size > 0:
        years = [None] * batch_size

    return years


def _sort_year_metric_keys(keys: List[Union[int, str]]) -> List[Union[int, str]]:
    def key_fn(x):
        if isinstance(x, (int, np.integer)):
            return (0, int(x), str(x))
        sx = str(x)
        if sx.isdigit():
            return (0, int(sx), sx)
        if sx.upper() == "UNKNOWN":
            return (2, 9999, sx)
        return (1, 9999, sx)
    return sorted(keys, key=key_fn)


def _accumulate_per_year_confusions(
    year_confusions: Dict[Union[int, str], torch.Tensor],
    year_image_counts: Dict[Union[int, str], int],
    preds: torch.Tensor,
    targets: torch.Tensor,
    batch_years: List[Optional[int]],
    num_classes: int,
    ignore_index: int,
) -> Tuple[Dict[Union[int, str], torch.Tensor], Dict[Union[int, str], int], int]:
    unresolved = 0
    bsz = preds.shape[0]
    device = preds.device

    for i in range(bsz):
        y = batch_years[i] if i < len(batch_years) else None
        if y is None:
            key: Union[int, str] = "UNKNOWN"
            unresolved += 1
        else:
            key = int(y)

        if key not in year_confusions:
            year_confusions[key] = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
            year_image_counts[key] = 0

        year_confusions[key] = _accumulate_confusion(
            year_confusions[key], preds[i], targets[i], num_classes, ignore_index
        )
        year_image_counts[key] = int(year_image_counts.get(key, 0)) + 1

    return year_confusions, year_image_counts, unresolved


def _compute_metrics_by_year(
    year_confusions: Dict[Union[int, str], torch.Tensor],
    year_image_counts: Optional[Dict[Union[int, str], int]],
    class_names: List[str],
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}

    for key in _sort_year_metric_keys(list(year_confusions.keys())):
        conf = year_confusions[key]
        m = _compute_metrics_from_confusion(conf, class_names)
        m["num_images"] = int(year_image_counts.get(key, 0)) if year_image_counts is not None else None
        m["num_labeled_pixels"] = int(conf.sum().item())
        out[str(key)] = m

    return out


def log_metrics_overall_and_by_year(logger, split_name: str, metrics: dict, loss_value: Optional[float] = None) -> None:
    if loss_value is None:
        logger.info(
            "%s — mIoU: %.4f | mF1: %.4f | mAcc: %.4f",
            split_name,
            metrics.get("mIoU", 0.0),
            metrics.get("mF1", 0.0),
            metrics.get("mAcc", 0.0),
        )
    else:
        logger.info(
            "%s — mIoU: %.4f | mF1: %.4f | mAcc: %.4f | loss: %.4f",
            split_name,
            metrics.get("mIoU", 0.0),
            metrics.get("mF1", 0.0),
            metrics.get("mAcc", 0.0),
            float(loss_value),
        )

    by_year = metrics.get("metrics_by_year", {})
    unresolved = int(metrics.get("num_samples_unresolved_year", 0))

    if not by_year:
        logger.info("%s per-year metrics: not available (year could not be resolved).", split_name)
        return

    logger.info("%s per-year metrics (%d years, unresolved samples=%d):", split_name, len(by_year), unresolved)
    for year_key in _sort_year_metric_keys(list(by_year.keys())):
        yk = str(year_key)
        m = by_year[yk]
        logger.info(
            "  year=%s | mIoU=%.4f | mF1=%.4f | mAcc=%.4f | imgs=%s | labeled_px=%s",
            yk,
            m.get("mIoU", 0.0),
            m.get("mF1", 0.0),
            m.get("mAcc", 0.0),
            m.get("num_images", 0),
            m.get("num_labeled_pixels", 0),
        )


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
    year_lookup: Optional[Dict[str, int]] = None,
) -> Tuple[dict, float]:
    model.eval()
    num_classes = len(class_names)
    device = next(model.parameters()).device
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    total_loss = 0.0

    # Per-year accumulators
    year_confusions: Dict[Union[int, str], torch.Tensor] = {}
    year_image_counts: Dict[Union[int, str], int] = {}
    unresolved_year_samples = 0

    # Keep only one tiny debug snapshot instead of storing all predictions
    first_pred_hist: Optional[Dict[int, int]] = None

    for data in loader:
        imgs = data["img"].data[0].cuda(non_blocking=True)
        targets = data["gt_semantic_seg"].data[0].long().squeeze(1).cuda(non_blocking=True)

        logits, _ = model(imgs)

        target_shape = targets.shape[-2:]
        logits = F.interpolate(logits, size=target_shape, mode="bilinear", align_corners=False)

        loss = compute_loss(logits, targets, loss_cfg, class_weights, ignore_index=ignore_index)
        total_loss += float(loss.item()) * imgs.size(0)

        preds = logits.argmax(dim=1)
        confusion = _accumulate_confusion(confusion, preds, targets, num_classes, ignore_index)

        # per-year confusion accumulation
        batch_years = resolve_batch_years(data, batch_size=imgs.size(0), year_lookup=year_lookup)
        year_confusions, year_image_counts, n_unres = _accumulate_per_year_confusions(
            year_confusions=year_confusions,
            year_image_counts=year_image_counts,
            preds=preds,
            targets=targets,
            batch_years=batch_years,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        unresolved_year_samples += int(n_unres)

        # Debug only for the very first sample/batch
        if first_pred_hist is None and preds.numel() > 0:
            u, c = np.unique(preds[0].detach().cpu().numpy(), return_counts=True)
            first_pred_hist = dict(zip(u.tolist(), c.tolist()))

        # Explicit cleanup helps with fragmentation in long runs
        del imgs, targets, logits, preds, loss

    metrics = _compute_metrics_from_confusion(confusion, class_names)
    metrics["metrics_by_year"] = _compute_metrics_by_year(year_confusions, year_image_counts, class_names)
    metrics["num_samples_unresolved_year"] = int(unresolved_year_samples)

    val_loss = total_loss / len(loader.dataset)

    if first_pred_hist is not None:
        print("[VAL DEBUG] First sample class dist:", first_pred_hist)

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

def save_checkpoint(
    model,
    optimizer,
    out_dir: Path,
    epoch: int,
    metrics: dict,
    train_loss: float,
    val_loss: Optional[float],
    score_name: str,
    score_value: float,
    best_pool: List[Tuple[float, Path]],
):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,  # kept as validation metrics for backward compatibility
        "train_loss": float(train_loss),  # optimization train loss (not train_eval_loss)
        "val_loss": float(val_loss) if val_loss is not None else None,
        "selection_score_name": score_name,
        "selection_score_value": float(score_value),
    }

    ckpt_path = out_dir / f"epoch_{epoch:04d}_{score_name}_{score_value:.4f}.pth"
    torch.save(checkpoint, ckpt_path)

    # keep best K (smaller score = better, т.к. loss)
    best_pool.append((score_value, ckpt_path))
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

    # NEW: train metrics strategy to avoid extra full-pass load
    parser.add_argument(
        "--train-metrics-mode",
        choices=["tail", "full", "none"],
        default="tail",
        help=(
            "How to compute train metrics at epoch end: "
            "'tail' = aggregate only last N training batches (cheap, default), "
            "'full' = run full evaluate(train_loader) pass (expensive), "
            "'none' = disable train metrics."
        ),
    )
    parser.add_argument(
        "--train-metrics-tail-batches",
        type=int,
        default=20,
        help="Number of last training batches to aggregate when --train-metrics-mode tail.",
    )

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
    logger.info("Class names (index -> name): %s", {i: n for i, n in enumerate(class_names)})

    ignore_index = getattr(train_loader.dataset, "ignore_index", 255)

    # Build year lookups once (for per-year metrics)
    train_year_lookup = build_year_lookup(train_loader.dataset)
    val_year_lookup = build_year_lookup(val_loader.dataset) if val_loader else {}
    logger.info("Year lookup keys: train=%d, val=%d", len(train_year_lookup), len(val_year_lookup))

    if args.class_weights == "none":
        class_weights = None
    elif args.class_weights == "auto":
        class_weights = compute_class_weights(
            train_loader,
            num_classes,
            ignore_index,
            mode=args.class_weight_mode,
            class_names=class_names,
            verbose=True,
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

    wrapper = FineTuneWrapper(base_model, mixers=mixers, mix_strategy=args.mix_strategy).cuda()

    # Optional debug: show some trainable names once
    logger.info("=== Trainable parameter name samples ===")
    shown = 0
    for n, p in wrapper.named_parameters():
        if p.requires_grad:
            logger.info("trainable: %s | shape=%s", n, tuple(p.shape))
            shown += 1
            if shown >= 30:
                break
    logger.info("=== End samples ===")

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

        # NEW: store compact summaries only for the last N train batches
        use_train_tail_metrics = (args.train_metrics_mode == "tail")
        tail_n = max(1, int(args.train_metrics_tail_batches))
        train_tail_summaries = deque(maxlen=tail_n) if use_train_tail_metrics else None

        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for data in progress:
            imgs = data["img"].data[0].cuda(non_blocking=True)
            targets = data["gt_semantic_seg"].data[0].long().squeeze(1).cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = wrapper(imgs)
            loss = compute_loss(logits, targets, loss_cfg, class_weights, ignore_index=ignore_index)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = float(loss.item())
            loss_window.append(loss_value)
            avg_recent = sum(loss_window) / len(loss_window)

            batch_size = int(imgs.size(0))
            total_loss += loss_value * batch_size

            # NEW: collect train-tail metrics on the fly (no extra forward pass, no full train eval)
            if use_train_tail_metrics:
                with torch.no_grad():
                    metric_logits = logits.detach()
                    if metric_logits.shape[-2:] != targets.shape[-2:]:
                        metric_logits = F.interpolate(
                            metric_logits,
                            size=targets.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    preds = metric_logits.argmax(dim=1)

                    batch_years = resolve_batch_years(
                        data,
                        batch_size=batch_size,
                        year_lookup=train_year_lookup if len(train_year_lookup) > 0 else None,
                    )
                    batch_summary = build_train_tail_batch_summary(
                        preds=preds,
                        targets=targets,
                        batch_loss_value=loss_value,
                        batch_size=batch_size,
                        class_names=class_names,
                        ignore_index=ignore_index,
                        batch_years=batch_years,
                    )
                    train_tail_summaries.append(batch_summary)

                    del metric_logits

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                avg100=f"{avg_recent:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # Help GC / CUDA allocator a bit in long epochs
            del imgs, targets, logits, loss
            if use_train_tail_metrics:
                del preds

        train_loss = total_loss / len(train_loader.dataset)
        logger.info("Epoch %d train loss (optimization): %.4f", epoch, train_loss)
        logger.info("Epoch %d LR groups: %s", epoch, [group["lr"] for group in optimizer.param_groups])

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),  # optimization loss during training loop
        }

        metrics: dict = {}  # validation metrics kept here for checkpoint compatibility
        val_loss: Optional[float] = None

        # ---------------------------
        # Train metrics
        # ---------------------------
        if args.train_metrics_mode == "full":
            # Expensive: extra full pass over train (old behavior)
            train_eval_metrics, train_eval_loss = evaluate(
                wrapper,
                train_loader,
                class_names,
                loss_cfg,
                class_weights,
                ignore_index,
                year_lookup=train_year_lookup if len(train_year_lookup) > 0 else None,
            )
            epoch_record.update({
                "train_eval_loss": float(train_eval_loss),
                "train_metrics": train_eval_metrics,
            })
            log_metrics_overall_and_by_year(logger, "TrainEval(full)", train_eval_metrics, train_eval_loss)

        elif args.train_metrics_mode == "tail":
            # Cheap: aggregate only the last N batches already seen during training
            train_tail_metrics, train_tail_loss = aggregate_train_tail_summaries(
                train_tail_summaries,
                class_names=class_names,
            )
            epoch_record.update({
                "train_tail_eval_loss": float(train_tail_loss),
                "train_tail_metrics": train_tail_metrics,
                "train_tail_batches": int(train_tail_metrics.get("window_num_batches", 0)),
                "train_tail_samples": int(train_tail_metrics.get("window_num_samples", 0)),
            })
            log_metrics_overall_and_by_year(
                logger,
                f"TrainTail(last_{train_tail_metrics.get('window_num_batches', 0)}_batches)",
                train_tail_metrics,
                train_tail_loss,
            )
        else:
            logger.info("Train metrics are disabled (--train-metrics-mode none).")

        # Optional: reduce allocator pressure before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------------------------
        # Validation metrics
        # ---------------------------
        if val_loader:
            metrics, val_loss = evaluate(
                wrapper,
                val_loader,
                class_names,
                loss_cfg,
                class_weights,
                ignore_index,
                year_lookup=val_year_lookup if len(val_year_lookup) > 0 else None,
            )
            epoch_record.update({
                "val_loss": float(val_loss),
                "metrics": metrics,  # validation metrics (legacy field name preserved)
            })

            log_metrics_overall_and_by_year(logger, "Validation", metrics, val_loss)

            viz_dir = run_dir / "val_viz" / f"epoch_{epoch:03d}"
            save_val_visualizations(epoch, wrapper, val_loader, viz_dir)

        # ---- choose score for "best" by val_loss if available, else train_loss ----
        if val_loader and val_loss is not None:
            score_name = "val_loss"
            score_value = float(val_loss)
        else:
            score_name = "train_loss"
            score_value = float(train_loss)

        best_checkpoints = save_checkpoint(
            model=wrapper,
            optimizer=optimizer,
            out_dir=run_dir,
            epoch=epoch,
            metrics=metrics,
            train_loss=train_loss,
            val_loss=val_loss,
            score_name=score_name,
            score_value=score_value,
            best_pool=best_checkpoints,
        )

        # log record
        with metrics_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

    logger.info("Training finished")


if __name__ == "__main__":
    main()