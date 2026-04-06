from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

ContextMode = Literal[
    "none",
    "lposs_only",
    "features_only",
    "temporal_semantic_only",
    "full",
    "custom",
]

ConditioningMechanism = Literal["concat_context", "film_context"]

CONTEXT_MODE_SOURCES: Dict[str, List[str]] = {
    "none": [],
    "lposs_only": ["lposs_mask_stats", "lposs_probs"],
    "features_only": ["dinov2_features", "clip_features", "siglip2_features"],
    "temporal_semantic_only": ["semantic_temporal_features", "semantic_fused_score"],
    "full": [
        "lposs_mask_stats",
        "lposs_probs",
        "dinov2_features",
        "clip_features",
        "siglip2_features",
        "semantic_temporal_features",
        "semantic_fused_score",
    ],
}


@dataclass(frozen=True)
class ConditioningConfig:
    context_mode: ContextMode = "full"
    context_sources: Sequence[str] = ()
    context_dim: int = 64
    conditioning_mechanism: ConditioningMechanism = "concat_context"
    max_symbols_per_tile: int = 512

    def resolved_sources(self) -> List[str]:
        if self.context_mode == "custom":
            return list(self.context_sources)
        return list(CONTEXT_MODE_SOURCES[self.context_mode])


@dataclass(frozen=True)
class TileSample:
    pair_id: str
    facade_id: str
    split: str
    tile_id: str
    tile_x: int
    tile_y: int
    x0: int
    y0: int
    x1: int
    y1: int
    residual_bytes: bytes
    context_vector: List[float]
    context_mode: str
    context_sources: List[str]
    context_backends: List[str]


@dataclass(frozen=True)
class TileScore:
    pair_id: str
    facade_id: str
    split: str
    tile_id: str
    tile_x: int
    tile_y: int
    score_type: str
    method: str
    context_mode: str
    conditioning_mechanism: str
    model_bits: float
    nll_bits: float
    bits_per_byte: float
