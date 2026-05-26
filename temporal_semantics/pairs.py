from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from compression.io import read_csv_rows, write_csv_rows

from .fusion import attach_fused_scores, fuse_backend_scores
from .io import load_json, load_mask_from_pgm
from .scoring import (
    backend_agreement,
    class_histogram_drift,
    entropy_from_probs,
    feature_cosine_distance,
    feature_l2_distance,
    mask_change_density,
    normalize_minmax,
    weighted_score,
)
from .tiling import generate_tiles
from .types import DEFAULT_FEATURE_WEIGHTS, DEFAULT_LPOSS_WEIGHTS


def _index_artifacts(artifact_index_csv: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    rows = read_csv_rows(artifact_index_csv)
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        out[(row["sample_id"], row["backend"])] = row
    return out


def _sample_id_from_image(path: str) -> str:
    return Path(path).stem


def _feature_map(path: str) -> Dict[Tuple[int, int], List[float]]:
    if not path:
        return {}
    payload = load_json(Path(path))
    return {(int(item["x"]), int(item["y"])): [float(v) for v in item["vec"]] for item in payload.get("features", [])}


def _tile_mask(mask_payload: bytes, width: int, tile: Tuple[int, int, int, int]) -> List[int]:
    x0, y0, x1, y1 = tile
    vals = []
    for yy in range(y0, y1):
        start = yy * width + x0
        vals.extend(mask_payload[start : start + (x1 - x0)])
    return [int(v) for v in vals]


def _tile_probs(probs: List[List[float]], width: int, tile: Tuple[int, int, int, int]) -> List[List[float]]:
    x0, y0, x1, y1 = tile
    out = []
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            out.append(probs[yy * width + xx])
    return out


def build_temporal_semantic_features(
    pairs_csv: Path,
    artifact_index_csv: Path,
    out_csv: Path,
    tile_size: int,
    backends: Iterable[str],
    lposs_weights: Dict[str, float] | None = None,
    feature_weights: Dict[str, float] | None = None,
    fuse_weights: Dict[str, float] | None = None,
) -> List[Dict[str, object]]:
    lposs_weights = DEFAULT_LPOSS_WEIGHTS if lposs_weights is None else lposs_weights
    feature_weights = DEFAULT_FEATURE_WEIGHTS if feature_weights is None else feature_weights
    fuse_weights = {
        "semantic_score_lposs": 0.45,
        "semantic_score_dinov2": 0.3,
        "semantic_score_clip": 0.2,
        "backend_agreement_score": 0.05,
    } if fuse_weights is None else fuse_weights

    pairs = read_csv_rows(pairs_csv)
    artifact_map = _index_artifacts(artifact_index_csv)
    backend_list = list(backends)
    out_rows: List[Dict[str, object]] = []

    for pair in pairs:
        pair_id = pair["pair_id"]
        prev_id = _sample_id_from_image(pair["prev_image_path"])
        curr_id = _sample_id_from_image(pair["curr_image_path"])

        # Use LPOSS artifact dimensions as canonical tile frame.
        curr_lposs = artifact_map[(curr_id, "lposs")]
        w, h, curr_mask = load_mask_from_pgm(Path(curr_lposs["mask_path"]))
        _, _, prev_mask = load_mask_from_pgm(Path(artifact_map[(prev_id, "lposs")]["mask_path"]))
        curr_probs = load_json(Path(curr_lposs["probs_path"])).get("probs", [])
        prev_probs = load_json(Path(artifact_map[(prev_id, "lposs")]["probs_path"])).get("probs", [])

        tiles = generate_tiles(w, h, tile_size=tile_size)
        rows_pair_backend: Dict[str, List[Dict[str, object]]] = defaultdict(list)

        for backend in backend_list:
            prev_art = artifact_map[(prev_id, backend)]
            curr_art = artifact_map[(curr_id, backend)]

            prev_feat = _feature_map(prev_art.get("features_path", ""))
            curr_feat = _feature_map(curr_art.get("features_path", ""))

            raw_scores = []
            interim_rows = []
            for tile in tiles:
                tile_key = (tile.x0 // tile_size, tile.y0 // tile_size)
                row = {
                    "pair_id": pair_id,
                    "facade_id": pair.get("facade_id", ""),
                    "year_prev": pair.get("year_prev", ""),
                    "year_curr": pair.get("year_curr", ""),
                    "tile_id": tile.tile_id,
                    "x0": tile.x0,
                    "y0": tile.y0,
                    "x1": tile.x1,
                    "y1": tile.y1,
                    "center_x": tile.center_x,
                    "center_y": tile.center_y,
                    "backend": backend,
                    "mask_change_density": "",
                    "class_histogram_drift": "",
                    "prob_entropy_change": "",
                    "feature_cosine_distance": "",
                    "feature_l2_distance": "",
                    "backend_agreement_score": 0.0,
                    "semantic_score_backend": 0.0,
                    "semantic_score_fused": 0.0,
                }

                if backend == "lposs":
                    prev_tile = _tile_mask(prev_mask, w, (tile.x0, tile.y0, tile.x1, tile.y1))
                    curr_tile = _tile_mask(curr_mask, w, (tile.x0, tile.y0, tile.x1, tile.y1))
                    d_mask = mask_change_density(prev_tile, curr_tile)
                    d_hist = class_histogram_drift(prev_tile, curr_tile)
                    pe_prev = entropy_from_probs(_tile_probs(prev_probs, w, (tile.x0, tile.y0, tile.x1, tile.y1)))
                    pe_curr = entropy_from_probs(_tile_probs(curr_probs, w, (tile.x0, tile.y0, tile.x1, tile.y1)))
                    d_entropy = abs(pe_curr - pe_prev)
                    row.update(
                        {
                            "mask_change_density": d_mask,
                            "class_histogram_drift": d_hist,
                            "prob_entropy_change": d_entropy,
                        }
                    )
                    raw = weighted_score(
                        {
                            "mask_change_density": d_mask,
                            "class_histogram_drift": d_hist,
                            "prob_entropy_change": d_entropy,
                        },
                        lposs_weights,
                    )
                else:
                    v_prev = prev_feat.get(tile_key)
                    v_curr = curr_feat.get(tile_key)
                    if v_prev is None or v_curr is None:
                        d_cos, d_l2 = 1.0, 1.0
                    else:
                        d_cos = feature_cosine_distance(v_prev, v_curr)
                        d_l2 = feature_l2_distance(v_prev, v_curr)
                    row.update({"feature_cosine_distance": d_cos, "feature_l2_distance": d_l2})
                    raw = weighted_score({"feature_cosine_distance": d_cos, "feature_l2_distance": d_l2}, feature_weights)

                raw_scores.append(raw)
                interim_rows.append(row)

            for row, norm in zip(interim_rows, normalize_minmax(raw_scores)):
                row["semantic_score_backend"] = norm
                rows_pair_backend[backend].append(row)

        # cross-backend agreement + fused score per tile
        fused_by_tile: Dict[str, float] = {}
        for tile in tiles:
            tile_id = tile.tile_id
            backend_scores = {}
            changed_flags = []
            for backend in backend_list:
                backend_row = next(r for r in rows_pair_backend[backend] if r["tile_id"] == tile_id)
                s = float(backend_row["semantic_score_backend"])
                backend_scores[f"semantic_score_{backend}"] = s
                changed_flags.append(1 if s >= 0.5 else 0)
            agree = backend_agreement(changed_flags)
            fused = fuse_backend_scores(backend_scores, agree, fuse_weights)
            fused_by_tile[f"{pair_id}::{tile_id}"] = fused
            for backend in backend_list:
                backend_row = next(r for r in rows_pair_backend[backend] if r["tile_id"] == tile_id)
                backend_row["backend_agreement_score"] = agree

        pair_rows: List[Dict[str, object]] = []
        for backend in backend_list:
            pair_rows.extend(rows_pair_backend[backend])
        attach_fused_scores(pair_rows, fused_by_tile)
        out_rows.extend(pair_rows)

    fields = [
        "pair_id",
        "facade_id",
        "year_prev",
        "year_curr",
        "tile_id",
        "x0",
        "y0",
        "x1",
        "y1",
        "center_x",
        "center_y",
        "backend",
        "mask_change_density",
        "class_histogram_drift",
        "prob_entropy_change",
        "feature_cosine_distance",
        "feature_l2_distance",
        "backend_agreement_score",
        "semantic_score_backend",
        "semantic_score_fused",
    ]
    write_csv_rows(out_csv, fields, out_rows)
    return out_rows
