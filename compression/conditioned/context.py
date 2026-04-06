from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from compression.io import read_csv_rows
from temporal_semantics.io import load_json, load_mask_from_pgm

from .types import CONTEXT_MODE_SOURCES


def _tile_mask(mask_payload: bytes, width: int, x0: int, y0: int, x1: int, y1: int) -> List[int]:
    vals: List[int] = []
    for yy in range(y0, y1):
        start = yy * width + x0
        vals.extend(mask_payload[start : start + (x1 - x0)])
    return [int(v) for v in vals]


def _entropy(probs: Sequence[float]) -> float:
    return -sum(float(p) * math.log2(float(p)) for p in probs if float(p) > 0)


def _summarize_probs(probs_tile: List[List[float]]) -> List[float]:
    if not probs_tile:
        return [0.0, 0.0, 0.0]
    n_classes = len(probs_tile[0]) if probs_tile[0] else 0
    avg = [0.0] * n_classes
    ent = []
    for pv in probs_tile:
        for i, v in enumerate(pv):
            avg[i] += float(v)
        ent.append(_entropy(pv))
    n = max(len(probs_tile), 1)
    avg = [v / n for v in avg]
    return avg + [sum(ent) / max(len(ent), 1), max(max(pv) if pv else 0.0 for pv in probs_tile)]


def _summarize_mask(mask_tile: List[int], n_classes: int = 8) -> List[float]:
    hist = [0.0] * n_classes
    for val in mask_tile:
        hist[int(val) % n_classes] += 1.0
    n = max(len(mask_tile), 1)
    probs = [v / n for v in hist]
    return probs + [max(probs) if probs else 0.0]


def _load_feature_map(path: str) -> Dict[Tuple[int, int], List[float]]:
    if not path:
        return {}
    payload = load_json(Path(path))
    return {(int(it["x"]), int(it["y"])): [float(v) for v in it.get("vec", [])] for it in payload.get("features", [])}


def _project_vector(values: Sequence[float], dim: int) -> List[float]:
    vals = [float(v) for v in values]
    if dim <= 0:
        return []
    return vals[:dim] + [0.0] * max(0, dim - len(vals))


def resolve_context_sources(context_mode: str, custom_sources: Sequence[str] | None = None) -> List[str]:
    if context_mode == "custom":
        return list(custom_sources or [])
    if context_mode not in CONTEXT_MODE_SOURCES:
        raise ValueError(f"Unsupported context_mode: {context_mode}")
    return list(CONTEXT_MODE_SOURCES[context_mode])


def build_pair_context_index(
    pairs_csv: Path,
    artifact_index_csv: Path,
    temporal_features_csv: Path | None,
    context_mode: str,
    context_dim: int,
    tile_size: int,
    custom_sources: Sequence[str] | None = None,
) -> Dict[str, Dict[str, object]]:
    sources = resolve_context_sources(context_mode, custom_sources)
    if not sources:
        return {}

    pairs = {row["pair_id"]: row for row in read_csv_rows(pairs_csv)}
    artifacts = read_csv_rows(artifact_index_csv)
    by_sample_backend = {(row["sample_id"], row["backend"]): row for row in artifacts}
    temporal_rows = read_csv_rows(temporal_features_csv) if temporal_features_csv is not None and temporal_features_csv.exists() else []

    pair_tile_rows: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in temporal_rows:
        pair_tile_rows.setdefault((row["pair_id"], row["tile_id"]), []).append(row)

    out: Dict[str, Dict[str, object]] = {}
    for (pair_id, tile_id), rows in pair_tile_rows.items():
        pair = pairs.get(pair_id)
        if pair is None:
            continue
        curr_sample_id = Path(pair["curr_image_path"]).stem

        lposs_art = by_sample_backend.get((curr_sample_id, "lposs"), {})
        mask_w = 0
        mask_payload = b""
        probs_payload: List[List[float]] = []
        if lposs_art.get("mask_path"):
            mask_w, _, mask_payload = load_mask_from_pgm(Path(lposs_art["mask_path"]))
            probs_payload = list(load_json(Path(lposs_art["probs_path"])).get("probs", []))

        fmap_dino = _load_feature_map(by_sample_backend.get((curr_sample_id, "dinov2"), {}).get("features_path", ""))
        fmap_clip = _load_feature_map(by_sample_backend.get((curr_sample_id, "clip"), {}).get("features_path", ""))
        fmap_siglip = _load_feature_map(by_sample_backend.get((curr_sample_id, "siglip2"), {}).get("features_path", ""))

        x0, y0, x1, y1 = [int(v) for v in tile_id.split("_")]
        tile_x, tile_y = x0 // tile_size, y0 // tile_size

        raw: List[float] = []
        used_backends: List[str] = []
        rep = rows[0]
        if "lposs_mask_stats" in sources and mask_payload:
            raw.extend(_summarize_mask(_tile_mask(mask_payload, mask_w, x0, y0, x1, y1)))
            used_backends.append("lposs")
        if "lposs_probs" in sources and probs_payload:
            probs_tile = []
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    probs_tile.append([float(v) for v in probs_payload[yy * mask_w + xx]])
            raw.extend(_summarize_probs(probs_tile))
            used_backends.append("lposs")
        if "dinov2_features" in sources:
            raw.extend(fmap_dino.get((tile_x, tile_y), [0.0] * 6))
            used_backends.append("dinov2")
        if "clip_features" in sources:
            raw.extend(fmap_clip.get((tile_x, tile_y), [0.0] * 4))
            used_backends.append("clip")
        if "siglip2_features" in sources:
            raw.extend(fmap_siglip.get((tile_x, tile_y), [0.0] * 6))
            used_backends.append("siglip2")

        if "semantic_temporal_features" in sources:
            vals = []
            for r in rows:
                vals.append(
                    [
                        float(r.get("mask_change_density") or 0.0),
                        float(r.get("class_histogram_drift") or 0.0),
                        float(r.get("prob_entropy_change") or 0.0),
                        float(r.get("feature_cosine_distance") or 0.0),
                        float(r.get("feature_l2_distance") or 0.0),
                        float(r.get("semantic_score_backend") or 0.0),
                        float(r.get("backend_agreement_score") or 0.0),
                    ]
                )
            means = [sum(v[i] for v in vals) / max(len(vals), 1) for i in range(len(vals[0]))] if vals else [0.0] * 7
            raw.extend(means)
            used_backends.extend([str(r.get("backend", "unknown")) for r in rows])

        if "semantic_fused_score" in sources:
            raw.append(float(rep.get("semantic_score_fused") or 0.0))
            used_backends.append("fused")

        out[f"{pair_id}::{tile_id}"] = {
            "context_vector": _project_vector(raw, context_dim),
            "context_sources": sources,
            "context_backends": sorted(set(used_backends)),
        }

    return out
