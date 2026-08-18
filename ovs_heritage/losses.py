"""Strict raw-logit losses for canonical v2 two-head supervision."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .projection import OntologyProjection


@dataclass(frozen=True)
class CombinedLoss:
    total: torch.Tensor
    main: torch.Tensor
    ornament: torch.Tensor
    metadata: dict[str, float | None]


def _validate_raw_logits(logits: torch.Tensor, label: str, *, detect_softmax: bool = False) -> None:
    if not logits.is_floating_point():
        raise ValueError(f"{label} must be floating-point raw logits")
    if torch.isfinite(logits).logical_not().any():
        raise ValueError(f"{label} contain non-finite values")
    if detect_softmax and logits.numel() and logits.min() >= 0 and logits.max() <= 1:
        sums = logits.sum(dim=1)
        if torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
            raise ValueError(f"{label} appear to be normalized probabilities; raw logits are required")


def main_segmentation_loss(
    main_logits: torch.Tensor, y_main: torch.Tensor,
    projection: OntologyProjection | None = None,
) -> torch.Tensor:
    projection = projection or OntologyProjection.canonical_v2()
    if main_logits.ndim != 4 or main_logits.shape[1] != projection.main_channel_count:
        raise ValueError(f"main_logits must be [N,{projection.main_channel_count},H,W]")
    _validate_raw_logits(main_logits, "main_logits", detect_softmax=True)
    if y_main.ndim == 4 and y_main.shape[1] == 1:
        y_main = y_main[:, 0]
    if y_main.ndim != 3 or main_logits.shape[0] != y_main.shape[0] or main_logits.shape[2:] != y_main.shape[1:]:
        raise ValueError("main_logits and Y_main have incompatible shapes")
    channels = projection.semantic_main_to_channels(y_main)
    if torch.all(channels == projection.ignore_index):
        return main_logits.sum() * 0.0
    return F.cross_entropy(main_logits, channels, ignore_index=projection.ignore_index)


def ornament_region_loss(
    ornament_logits: torch.Tensor, y_ornament: torch.Tensor,
    *, pos_weight: float | None = None,
) -> torch.Tensor:
    if ornament_logits.ndim != 4 or ornament_logits.shape[1] != 1:
        raise ValueError("ornament_logits must be [N,1,H,W]")
    _validate_raw_logits(ornament_logits, "ornament_logits")
    if y_ornament.ndim == 3:
        y_ornament = y_ornament.unsqueeze(1)
    if y_ornament.shape != ornament_logits.shape:
        raise ValueError("ornament_logits and Y_ornament have incompatible shapes")
    OntologyProjection._validate_integer_target(y_ornament, "Y_ornament")
    found = set(torch.unique(y_ornament.detach()).cpu().tolist())
    invalid = sorted(found - {0, 1, 255})
    if invalid:
        raise ValueError(f"Y_ornament contains invalid values {invalid}; allowed values are 0, 1, 255")
    if pos_weight is not None and (not math.isfinite(pos_weight) or pos_weight <= 0):
        raise ValueError("pos_weight must be finite and positive")
    valid = y_ornament != 255
    if not valid.any():
        return ornament_logits.sum() * 0.0
    safe_target = torch.where(valid, y_ornament, torch.zeros_like(y_ornament)).to(ornament_logits.dtype)
    weight_tensor = None if pos_weight is None else ornament_logits.new_tensor([pos_weight])
    elementwise = F.binary_cross_entropy_with_logits(
        ornament_logits, safe_target, reduction="none", pos_weight=weight_tensor,
    )
    return elementwise[valid].mean()


def combined_two_head_loss(
    main_logits: torch.Tensor, ornament_logits: torch.Tensor,
    y_main: torch.Tensor, y_ornament: torch.Tensor,
    *, lambda_ornament: float = 1.0, pos_weight: float | None = None,
    projection: OntologyProjection | None = None,
) -> CombinedLoss:
    if not math.isfinite(lambda_ornament) or lambda_ornament < 0:
        raise ValueError("lambda_ornament must be finite and non-negative")
    main = main_segmentation_loss(main_logits, y_main, projection)
    ornament = ornament_region_loss(ornament_logits, y_ornament, pos_weight=pos_weight)
    total = main + lambda_ornament * ornament
    return CombinedLoss(total, main, ornament, {
        "lambda_ornament": float(lambda_ornament),
        "pos_weight": None if pos_weight is None else float(pos_weight),
    })


supervised_cross_entropy = main_segmentation_loss
