from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

from compression.io import write_csv_rows

from .dataset import build_conditioned_samples
from .model import ConditionedResidualModel
from .scoring import score_tile, summarize_pair_scores


def eval_semantic_conditioned_codec(
    residual_manifest_csv: Path,
    pairs_csv: Path,
    artifact_index_csv: Path,
    temporal_features_csv: Path | None,
    model_path: Path,
    split: str,
    tile_size: int,
    context_mode: str,
    context_dim: int,
    conditioning_mechanism: str,
    out_tile_csv: Path,
    out_pair_csv: Path,
    custom_sources: Sequence[str] | None = None,
) -> Dict[str, object]:
    model = ConditionedResidualModel.load(model_path)
    samples = build_conditioned_samples(
        residual_manifest_csv=residual_manifest_csv,
        pairs_csv=pairs_csv,
        artifact_index_csv=artifact_index_csv,
        temporal_features_csv=temporal_features_csv,
        tile_size=tile_size,
        context_mode=context_mode,
        context_dim=context_dim,
        custom_sources=custom_sources,
    )
    selected = [s for s in samples if s.split == split]

    tile_scores = [score_tile(model, s, method="semantic_conditioned_codec") for s in selected]
    pair_scores = summarize_pair_scores(tile_scores)

    write_csv_rows(
        out_tile_csv,
        [
            "pair_id",
            "facade_id",
            "split",
            "tile_id",
            "tile_x",
            "tile_y",
            "method",
            "score_type",
            "context_mode",
            "conditioning_mechanism",
            "model_bits",
            "nll_bits",
            "bits_per_byte",
        ],
        [ts.__dict__ for ts in tile_scores],
    )

    write_csv_rows(
        out_pair_csv,
        [
            "pair_id",
            "facade_id",
            "split",
            "method",
            "score_type",
            "context_mode",
            "conditioning_mechanism",
            "model_bits",
            "nll_bits",
            "bits_per_byte",
            "n_tiles",
        ],
        pair_scores,
    )

    mean_bpb = sum(ts.bits_per_byte for ts in tile_scores) / max(len(tile_scores), 1)
    return {
        "n_tiles": len(tile_scores),
        "n_pairs": len(pair_scores),
        "split": split,
        "context_mode": context_mode,
        "conditioning_mechanism": conditioning_mechanism,
        "mean_bits_per_byte": mean_bpb,
    }
