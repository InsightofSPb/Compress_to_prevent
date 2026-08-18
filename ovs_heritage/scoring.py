"""Raw cosine dense scorer; intentionally contains no softmax or vocabulary state."""
from __future__ import annotations
import math

import torch
import torch.nn.functional as F
from torch import nn


class RawCosineScorer(nn.Module):
    def __init__(self, scale: float = 100.0, eps: float = 1e-12):
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.scale = float(scale)
        self.eps = eps

    def forward(self, features: torch.Tensor, prototypes: torch.Tensor,
                *, scale: torch.Tensor | float | None = None,
                bias: torch.Tensor | float | None = None) -> torch.Tensor:
        if features.ndim not in (3, 4):
            raise ValueError("features must be [D,H,W] or [N,D,H,W]")
        if prototypes.ndim != 2:
            raise ValueError("prototypes must be [C,D]")
        if not features.is_floating_point() or not prototypes.is_floating_point():
            raise ValueError("features and prototypes must be floating-point tensors")
        if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():
            raise ValueError("features and prototypes must be finite")
        if torch.any(torch.linalg.vector_norm(prototypes, dim=1) <= self.eps):
            raise ValueError("prototypes must have finite non-zero norms")
        unbatched = features.ndim == 3
        if unbatched:
            features = features.unsqueeze(0)
        if features.shape[1] != prototypes.shape[1]:
            raise ValueError(f"embedding dimension mismatch: features={features.shape[1]}, prototypes={prototypes.shape[1]}")
        prototypes = prototypes.to(device=features.device, dtype=features.dtype)
        logits = torch.einsum("ndhw,cd->nchw", F.normalize(features, dim=1, eps=self.eps),
                              F.normalize(prototypes, dim=1, eps=self.eps))
        scale = self.scale if scale is None else scale
        scale = torch.as_tensor(scale, device=logits.device, dtype=logits.dtype)
        bias = torch.as_tensor(0.0 if bias is None else bias, device=logits.device, dtype=logits.dtype)
        for value, label in ((scale, "scale"), (bias, "bias")):
            if value.ndim > 1 or (value.ndim == 1 and value.numel() not in (1, prototypes.shape[0])):
                raise ValueError(f"{label} must be scalar or have one value per class")
            if not torch.isfinite(value).all():
                raise ValueError(f"{label} must be finite")
        if scale.ndim:
            scale = scale.view(1, -1, 1, 1)
        if bias.ndim:
            bias = bias.view(1, -1, 1, 1)
        logits = logits * scale + bias
        return logits[0] if unbatched else logits
