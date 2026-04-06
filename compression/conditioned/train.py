from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

from .dataset import build_conditioned_samples
from .model import ConditionedResidualModel


def train_semantic_conditioned_codec(
    residual_manifest_csv: Path,
    pairs_csv: Path,
    artifact_index_csv: Path,
    temporal_features_csv: Path | None,
    model_out: Path,
    tile_size: int,
    train_split: str,
    context_mode: str,
    context_dim: int,
    conditioning_mechanism: str,
    custom_sources: Sequence[str] | None = None,
    max_symbols_per_tile: int = 512,
    ridge_lambda: float = 1e-3,
) -> Dict[str, object]:
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
    train_samples = [s for s in samples if s.split == train_split]

    model = ConditionedResidualModel(
        conditioning_mechanism=conditioning_mechanism,
        ridge_lambda=ridge_lambda,
    )
    fit_summary = model.fit(train_samples, max_symbols_per_tile=max_symbols_per_tile)
    model.save(model_out)

    meta = {
        "method": "semantic_conditioned_codec",
        "score_type": "model_bits",
        "context_mode": context_mode,
        "conditioning_mechanism": conditioning_mechanism,
        "context_dim": context_dim,
        "tile_size": tile_size,
        "train_split": train_split,
        "train_tiles": len(train_samples),
        "fit_summary": fit_summary,
        "context_sources": list(custom_sources or []),
    }
    meta_path = model_out.with_suffix(model_out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta
