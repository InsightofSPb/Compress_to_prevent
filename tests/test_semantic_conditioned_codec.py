from __future__ import annotations

import math
from pathlib import Path

from compression.conditioned.context import resolve_context_sources
from compression.conditioned.dataset import build_conditioned_samples
from compression.conditioned.eval import eval_semantic_conditioned_codec
from compression.conditioned.model import ConditionedResidualModel
from compression.conditioned.train import train_semantic_conditioned_codec
from compression.io import read_csv_rows, write_csv_rows


def _write_ppm(path: Path, width: int, height: int, value: int) -> None:
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes([value] * width * height * 3))


def _write_pgm(path: Path, width: int, height: int, values: bytes) -> None:
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + values)


def _make_fixture(tmp_path: Path) -> dict[str, Path]:
    img_prev = tmp_path / "p0.ppm"
    img_curr = tmp_path / "p1.ppm"
    residual = tmp_path / "r.ppm"
    _write_ppm(img_prev, 4, 4, 5)
    _write_ppm(img_curr, 4, 4, 7)
    _write_ppm(residual, 4, 4, 2)

    pairs_csv = tmp_path / "pairs.csv"
    write_csv_rows(
        pairs_csv,
        ["pair_id", "facade_id", "split", "prev_image_path", "curr_image_path"],
        [{"pair_id": "pairA", "facade_id": "facA", "split": "train", "prev_image_path": str(img_prev), "curr_image_path": str(img_curr)}],
    )

    residual_manifest = tmp_path / "residual_manifest.csv"
    write_csv_rows(
        residual_manifest,
        ["pair_id", "facade_id", "split", "residual_path"],
        [{"pair_id": "pairA", "facade_id": "facA", "split": "train", "residual_path": str(residual)}],
    )

    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    mask_path = artifacts_root / "p1_mask.pgm"
    probs_path = artifacts_root / "p1_probs.json"
    _write_pgm(mask_path, 4, 4, bytes([0, 1, 1, 0] * 4))
    probs = [[0.7, 0.2, 0.1, 0.0] for _ in range(16)]
    probs_path.write_text('{"probs": ' + str(probs).replace("'", '"') + '}', encoding="utf-8")

    for name, dim in [("dinov2", 6), ("clip", 4), ("siglip2", 6), ("lposs", 4)]:
        feats = {"features": [{"x": 0, "y": 0, "vec": [0.1] * dim}, {"x": 1, "y": 0, "vec": [0.2] * dim}, {"x": 0, "y": 1, "vec": [0.3] * dim}, {"x": 1, "y": 1, "vec": [0.4] * dim}]}
        (artifacts_root / f"p1_{name}_features.json").write_text(str(feats).replace("'", '"'), encoding="utf-8")

    artifact_index = tmp_path / "artifact_index.csv"
    write_csv_rows(
        artifact_index,
        ["sample_id", "backend", "image_path", "mask_path", "probs_path", "features_path", "overlay_path", "feature_grid_h", "feature_grid_w", "split", "status", "notes"],
        [
            {"sample_id": "p1", "backend": "lposs", "image_path": str(img_curr), "mask_path": str(mask_path), "probs_path": str(probs_path), "features_path": str(artifacts_root / "p1_lposs_features.json"), "overlay_path": "", "feature_grid_h": 2, "feature_grid_w": 2, "split": "train", "status": "ok", "notes": ""},
            {"sample_id": "p1", "backend": "dinov2", "image_path": str(img_curr), "mask_path": "", "probs_path": "", "features_path": str(artifacts_root / "p1_dinov2_features.json"), "overlay_path": "", "feature_grid_h": 2, "feature_grid_w": 2, "split": "train", "status": "ok", "notes": ""},
            {"sample_id": "p1", "backend": "clip", "image_path": str(img_curr), "mask_path": "", "probs_path": "", "features_path": str(artifacts_root / "p1_clip_features.json"), "overlay_path": "", "feature_grid_h": 2, "feature_grid_w": 2, "split": "train", "status": "ok", "notes": ""},
            {"sample_id": "p1", "backend": "siglip2", "image_path": str(img_curr), "mask_path": "", "probs_path": "", "features_path": str(artifacts_root / "p1_siglip2_features.json"), "overlay_path": "", "feature_grid_h": 2, "feature_grid_w": 2, "split": "train", "status": "ok", "notes": ""},
        ],
    )

    features_csv = tmp_path / "temporal_features.csv"
    rows = []
    for tx in range(2):
        for ty in range(2):
            rows.append(
                {
                    "pair_id": "pairA",
                    "facade_id": "facA",
                    "year_prev": "",
                    "year_curr": "",
                    "tile_id": f"{tx*2}_{ty*2}_{tx*2+2}_{ty*2+2}",
                    "x0": tx * 2,
                    "y0": ty * 2,
                    "x1": tx * 2 + 2,
                    "y1": ty * 2 + 2,
                    "center_x": 0,
                    "center_y": 0,
                    "backend": "lposs",
                    "mask_change_density": 0.1,
                    "class_histogram_drift": 0.2,
                    "prob_entropy_change": 0.3,
                    "feature_cosine_distance": 0.4,
                    "feature_l2_distance": 0.5,
                    "backend_agreement_score": 0.6,
                    "semantic_score_backend": 0.7,
                    "semantic_score_fused": 0.8,
                }
            )
    write_csv_rows(features_csv, list(rows[0].keys()), rows)

    return {
        "pairs_csv": pairs_csv,
        "residual_manifest": residual_manifest,
        "artifact_index": artifact_index,
        "features_csv": features_csv,
    }


def test_context_mode_dispatch() -> None:
    assert resolve_context_sources("none") == []
    assert "lposs_mask_stats" in resolve_context_sources("full")


def test_conditioned_dataset_and_tile_join(tmp_path: Path) -> None:
    fx = _make_fixture(tmp_path)
    samples = build_conditioned_samples(
        residual_manifest_csv=fx["residual_manifest"],
        pairs_csv=fx["pairs_csv"],
        artifact_index_csv=fx["artifact_index"],
        temporal_features_csv=fx["features_csv"],
        tile_size=2,
        context_mode="full",
        context_dim=16,
    )
    assert len(samples) == 4
    assert all(s.pair_id == "pairA" for s in samples)
    assert all(len(s.context_vector) == 16 for s in samples)


def test_conditioned_model_output_schema_and_finite(tmp_path: Path) -> None:
    fx = _make_fixture(tmp_path)
    samples = build_conditioned_samples(
        residual_manifest_csv=fx["residual_manifest"],
        pairs_csv=fx["pairs_csv"],
        artifact_index_csv=fx["artifact_index"],
        temporal_features_csv=fx["features_csv"],
        tile_size=2,
        context_mode="full",
        context_dim=8,
    )
    model = ConditionedResidualModel(conditioning_mechanism="film_context")
    model.fit(samples)
    total, parts = model.nll_bits_with_components(samples[0].residual_bytes, samples[0].context_vector)
    assert math.isfinite(total)
    assert len(parts) == len(samples[0].residual_bytes)


def test_train_eval_score_type_labeling(tmp_path: Path) -> None:
    fx = _make_fixture(tmp_path)
    model_path = tmp_path / "c1_model.json"
    train_semantic_conditioned_codec(
        residual_manifest_csv=fx["residual_manifest"],
        pairs_csv=fx["pairs_csv"],
        artifact_index_csv=fx["artifact_index"],
        temporal_features_csv=fx["features_csv"],
        model_out=model_path,
        tile_size=2,
        train_split="train",
        context_mode="full",
        context_dim=8,
        conditioning_mechanism="concat_context",
    )
    tile_csv = tmp_path / "tile.csv"
    pair_csv = tmp_path / "pair.csv"
    eval_semantic_conditioned_codec(
        residual_manifest_csv=fx["residual_manifest"],
        pairs_csv=fx["pairs_csv"],
        artifact_index_csv=fx["artifact_index"],
        temporal_features_csv=fx["features_csv"],
        model_path=model_path,
        split="train",
        tile_size=2,
        context_mode="full",
        context_dim=8,
        conditioning_mechanism="concat_context",
        out_tile_csv=tile_csv,
        out_pair_csv=pair_csv,
    )
    tile_rows = read_csv_rows(tile_csv)
    assert tile_rows[0]["score_type"] == "model_bits"
    assert "bits_per_byte" in tile_rows[0]
