from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Tile:
    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    center_x: float
    center_y: float


@dataclass
class ArtifactRecord:
    sample_id: str
    backend: str
    image_path: str
    mask_path: str
    probs_path: str
    features_path: str
    overlay_path: str
    feature_grid_h: int
    feature_grid_w: int
    split: str
    status: str
    notes: str


Weights = Dict[str, float]


DEFAULT_LPOSS_WEIGHTS: Weights = {
    "mask_change_density": 0.5,
    "class_histogram_drift": 0.3,
    "prob_entropy_change": 0.2,
}

DEFAULT_FEATURE_WEIGHTS: Weights = {
    "feature_cosine_distance": 0.6,
    "feature_l2_distance": 0.4,
}
