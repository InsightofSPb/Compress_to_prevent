from __future__ import annotations

from typing import Dict, List


def fuse_backend_scores(
    backend_scores: Dict[str, float],
    backend_agreement_score: float,
    weights: Dict[str, float],
) -> float:
    score = 0.0
    for name, value in backend_scores.items():
        score += weights.get(name, 0.0) * value
    score += weights.get("backend_agreement_score", 0.0) * backend_agreement_score
    return score


def attach_fused_scores(rows: List[Dict[str, object]], fused_by_tile: Dict[str, float]) -> List[Dict[str, object]]:
    for row in rows:
        key = f"{row['pair_id']}::{row['tile_id']}"
        row["semantic_score_fused"] = fused_by_tile.get(key, 0.0)
    return rows
