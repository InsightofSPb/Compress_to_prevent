from pathlib import Path

from compression.io import read_csv_rows, write_csv_rows
from temporal_semantics.artifacts import export_semantic_artifacts, list_available_backends
from temporal_semantics.backends import default_registry
from temporal_semantics.evaluation import evaluate_temporal_semantic_features
from temporal_semantics.pairs import build_temporal_semantic_features
from temporal_semantics.scoring import (
    class_histogram_drift,
    feature_cosine_distance,
    mask_change_density,
)
from temporal_semantics.tiling import generate_tiles


def _write_ppm(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    payload = bytes(list(rgb) * (width * height))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)


def test_backend_registry_and_capabilities() -> None:
    reg = default_registry()
    caps = reg.capabilities("lposs")
    assert caps.provides_masks is True
    assert "siglip2" in reg.names()
    listed = list_available_backends()
    assert any(row["backend"] == "dinov2" for row in listed)


def test_artifact_index_schema(tmp_path: Path) -> None:
    img = tmp_path / "a.ppm"
    _write_ppm(img, 4, 4, (10, 20, 30))
    manifest = tmp_path / "manifest.csv"
    write_csv_rows(manifest, ["sample_id", "image_path", "split"], [{"sample_id": "s1", "image_path": str(img), "split": "train"}])
    export_semantic_artifacts(manifest, ["lposs", "dinov2", "clip", "siglip2"], tmp_path / "out", tile_size=2)

    rows = read_csv_rows(tmp_path / "out" / "artifact_index.csv")
    assert set(rows[0].keys()) == {
        "sample_id",
        "backend",
        "image_path",
        "mask_path",
        "probs_path",
        "features_path",
        "overlay_path",
        "feature_grid_h",
        "feature_grid_w",
        "image_width",
        "image_height",
        "split",
        "status",
        "capabilities_json",
        "notes",
    }


def test_tile_coordinate_determinism() -> None:
    t1 = generate_tiles(8, 8, tile_size=4)
    t2 = generate_tiles(8, 8, tile_size=4)
    assert [t.tile_id for t in t1] == [t.tile_id for t in t2]


def test_mask_change_density_toy() -> None:
    prev = [0, 0, 1, 1]
    curr = [0, 1, 1, 0]
    assert mask_change_density(prev, curr) == 0.5


def test_class_histogram_drift_toy() -> None:
    prev = [0, 0, 0, 1]
    curr = [0, 1, 1, 1]
    assert class_histogram_drift(prev, curr, n_classes=2) > 0.0


def test_feature_cosine_distance_identical() -> None:
    vec = [0.1, 0.2, 0.3]
    assert abs(feature_cosine_distance(vec, vec)) < 1e-9


def test_fused_semantic_score_and_agreement_schema(tmp_path: Path) -> None:
    img0 = tmp_path / "f_2020.ppm"
    img1 = tmp_path / "f_2021.ppm"
    _write_ppm(img0, 4, 4, (10, 20, 30))
    _write_ppm(img1, 4, 4, (20, 10, 30))

    manifest = tmp_path / "manifest.csv"
    write_csv_rows(
        manifest,
        ["sample_id", "image_path", "split"],
        [
            {"sample_id": "f_2020", "image_path": str(img0), "split": "train"},
            {"sample_id": "f_2021", "image_path": str(img1), "split": "train"},
        ],
    )
    out_dir = tmp_path / "artifacts"
    export_semantic_artifacts(manifest, ["lposs", "dinov2", "clip", "siglip2"], out_dir, tile_size=2)

    pairs = tmp_path / "pairs.csv"
    write_csv_rows(
        pairs,
        ["pair_id", "facade_id", "year_prev", "year_curr", "prev_image_path", "curr_image_path", "split"],
        [
            {
                "pair_id": "f_2020_2021",
                "facade_id": "f",
                "year_prev": 2020,
                "year_curr": 2021,
                "prev_image_path": str(img0),
                "curr_image_path": str(img1),
                "split": "train",
            }
        ],
    )

    features_csv = tmp_path / "features.csv"
    rows = build_temporal_semantic_features(
        pairs_csv=pairs,
        artifact_index_csv=out_dir / "artifact_index.csv",
        out_csv=features_csv,
        tile_size=2,
        backends=["lposs", "dinov2", "clip", "siglip2"],
    )
    assert len(rows) > 0
    assert "semantic_score_fused" in rows[0]
    assert "backend_agreement_score" in rows[0]
    assert "backend_disagreement_score" in rows[0]

    summary_csv = tmp_path / "summary.csv"
    topk_csv = tmp_path / "topk.csv"
    evaluate_temporal_semantic_features(features_csv, summary_csv, topk_csv, top_k=2)
    summary_rows = read_csv_rows(summary_csv)
    assert "mean_backend_agreement_score" in summary_rows[0]
