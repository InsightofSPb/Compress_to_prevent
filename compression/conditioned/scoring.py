from __future__ import annotations

from typing import Dict, List

from .model import ConditionedResidualModel
from .types import TileSample, TileScore


def score_tile(model: ConditionedResidualModel, sample: TileSample, method: str) -> TileScore:
    nll_bits, _ = model.nll_bits_with_components(sample.residual_bytes, sample.context_vector)
    num_bytes = max(len(sample.residual_bytes), 1)
    bpb = nll_bits / num_bytes
    return TileScore(
        pair_id=sample.pair_id,
        facade_id=sample.facade_id,
        split=sample.split,
        tile_id=sample.tile_id,
        tile_x=sample.tile_x,
        tile_y=sample.tile_y,
        score_type="model_bits",
        method=method,
        context_mode=sample.context_mode,
        conditioning_mechanism=model.conditioning_mechanism,
        model_bits=nll_bits,
        nll_bits=nll_bits,
        bits_per_byte=bpb,
    )


def summarize_pair_scores(tile_scores: List[TileScore]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[TileScore]] = {}
    for ts in tile_scores:
        grouped.setdefault(ts.pair_id, []).append(ts)

    out: List[Dict[str, object]] = []
    for pair_id, rows in grouped.items():
        total_bits = sum(r.model_bits for r in rows)
        out.append(
            {
                "pair_id": pair_id,
                "facade_id": rows[0].facade_id if rows else "",
                "split": rows[0].split if rows else "",
                "method": rows[0].method if rows else "semantic_conditioned_codec",
                "score_type": "model_bits",
                "context_mode": rows[0].context_mode if rows else "none",
                "conditioning_mechanism": rows[0].conditioning_mechanism if rows else "concat_context",
                "model_bits": total_bits,
                "nll_bits": total_bits,
                "bits_per_byte": sum(r.bits_per_byte for r in rows) / max(len(rows), 1),
                "n_tiles": len(rows),
            }
        )
    return out
