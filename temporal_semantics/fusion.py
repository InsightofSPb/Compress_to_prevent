from __future__ import annotations

from typing import Dict, List


def fuse_backend_scores(
    backend_scores: Dict[str, float],
    backend_agreement_score: float,
    weights: Dict[str, float],
    normalize_weights: bool = True,
) -> float:
    values = dict(backend_scores)
    values["backend_agreement_score"] = backend_agreement_score
    values.setdefault("backend_disagreement_score", 1.0 - backend_agreement_score)

    if normalize_weights:
        denom = sum(abs(weights.get(k, 0.0)) for k in values.keys())
        denom = denom if denom > 0 else 1.0
    else:
        denom = 1.0

    score = sum(values[k] * weights.get(k, 0.0) for k in values.keys())
    return score / denom


def attach_fused_scores(rows: List[Dict[str, object]], fused_by_tile: Dict[str, float]) -> List[Dict[str, object]]:
    for row in rows:
        key = f"{row['pair_id']}::{row['tile_id']}"
        row["semantic_score_fused"] = fused_by_tile.get(key, 0.0)
    return rows
