"""Minimal strict segmentation loss operating on raw logits."""
import torch
import torch.nn.functional as F

def supervised_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, *, ignore_index: int = 255) -> torch.Tensor:
    if logits.ndim != 4: raise ValueError("logits must be raw [N,C,H,W] scores")
    if targets.ndim == 4 and targets.shape[1] == 1: targets = targets[:, 0]
    if targets.ndim != 3: raise ValueError("targets must be [N,H,W] or [N,1,H,W]")
    if logits.shape[0] != targets.shape[0] or logits.shape[2:] != targets.shape[1:]:
        raise ValueError("logits and targets have incompatible batch/spatial shapes")
    found_values = torch.unique(targets.detach()).cpu().tolist()
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if targets.dtype not in integer_dtypes:
        raise ValueError(
            f"targets must have an integer dtype before cross_entropy, got {targets.dtype}; "
            f"found IDs {found_values}"
        )
    found = set(found_values)
    invalid = found - set(range(logits.shape[1])) - {ignore_index}
    if invalid: raise ValueError(f"unknown target IDs {sorted(invalid)} for {logits.shape[1]} channels; labels are not remapped to ignore")
    return F.cross_entropy(logits, targets.long(), ignore_index=ignore_index)
