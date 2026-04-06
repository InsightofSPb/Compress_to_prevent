from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import Weights


def mask_change_density(prev_mask: Sequence[int], curr_mask: Sequence[int]) -> float:
    if len(prev_mask) != len(curr_mask):
        raise ValueError("Mask sizes do not match")
    changed = sum(1 for p, c in zip(prev_mask, curr_mask) if p != c)
    return changed / max(len(prev_mask), 1)


def class_histogram_drift(prev_mask: Sequence[int], curr_mask: Sequence[int], n_classes: int = 6) -> float:
    prev_hist = [0] * n_classes
    curr_hist = [0] * n_classes
    for val in prev_mask:
        prev_hist[int(val) % n_classes] += 1
    for val in curr_mask:
        curr_hist[int(val) % n_classes] += 1
    denom_prev = max(sum(prev_hist), 1)
    denom_curr = max(sum(curr_hist), 1)
    prev_p = [v / denom_prev for v in prev_hist]
    curr_p = [v / denom_curr for v in curr_hist]
    return sum(abs(a - b) for a, b in zip(prev_p, curr_p)) / 2.0


def entropy_from_probs(prob_vectors: Sequence[Sequence[float]]) -> float:
    if not prob_vectors:
        return 0.0
    entropies = []
    for probs in prob_vectors:
        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * math.log2(p)
        entropies.append(h)
    return sum(entropies) / len(entropies)


def feature_cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return max(0.0, min(2.0, 1.0 - dot / (na * nb)))


def feature_l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def weighted_score(values: Dict[str, float], weights: Weights) -> float:
    return sum(values.get(name, 0.0) * w for name, w in weights.items())


def normalize_minmax(values: Iterable[float]) -> List[float]:
    vals = list(values)
    if not vals:
        return []
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        return [0.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]


def backend_agreement(changed_flags: Sequence[int]) -> float:
    if not changed_flags:
        return 0.0
    ones = sum(changed_flags)
    zeros = len(changed_flags) - ones
    return max(ones, zeros) / len(changed_flags)
