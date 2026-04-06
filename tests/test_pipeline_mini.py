from pathlib import Path

from compression.codecs import benchmark_residual_codecs
from compression.io import read_csv_rows, write_csv_rows
from compression.lm.workflow import eval_entropy_model, train_entropy_model
from compression.metrics import evaluate_change_metrics
from compression.pairs import build_facade_pairs, read_observations, write_split_pair_csvs
from compression.residuals import build_residual_dataset
from compression.tiles import eval_change_tiles


def _write_ppm(path: Path, width: int, height: int, value: int) -> None:
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes([value] * width * height * 3))


def test_mini_manifest_end_to_end_consistency(tmp_path: Path) -> None:
    i0, i1 = tmp_path / "f_2020.ppm", tmp_path / "f_2021.ppm"
    _write_ppm(i0, 4, 4, 0)
    _write_ppm(i1, 4, 4, 5)

    manifest = tmp_path / "manifest.csv"
    write_csv_rows(
        manifest,
        ["facade_id", "year", "image_path", "aligned_image_path", "split"],
        [
            {"facade_id": "f", "year": 2020, "image_path": str(i0), "aligned_image_path": str(i0), "split": "train"},
            {"facade_id": "f", "year": 2021, "image_path": str(i1), "aligned_image_path": str(i0), "split": "train"},
        ],
    )

    pairs = build_facade_pairs(read_observations(manifest), pair_mode="all_to_latest")
    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    write_split_pair_csvs(pairs, pair_dir)
    build_residual_dataset(pair_dir / "pairs_all.csv", tmp_path / "res")

    residual_manifest = tmp_path / "res" / "residual_manifest.csv"
    bench_csv = tmp_path / "bench.csv"
    bench_rows = benchmark_residual_codecs(residual_manifest, bench_csv, methods=["lzma", "fnlic"], level=1)
    assert len(bench_rows) >= 1
    assert all(row["score_type"] == "achieved_bits" for row in read_csv_rows(bench_csv))

    model = tmp_path / "model.json"
    train_entropy_model(residual_manifest, model, split="train", model_mode="bigram")
    eval_csv = tmp_path / "eval.csv"
    tile_lm_csv = tmp_path / "lm_tiles.csv"
    eval_entropy_model(residual_manifest, model, split="train", out_csv=eval_csv, tile_size=2, tile_out_csv=tile_lm_csv)
    assert all(row["score_type"] == "model_bits" for row in read_csv_rows(eval_csv))

    change_tiles_csv = tmp_path / "change_tiles.csv"
    heat_dir = tmp_path / "heat"
    eval_change_tiles(residual_manifest, change_tiles_csv, heat_dir, tile_size=2)

    labels = tmp_path / "labels.csv"
    tile_rows = read_csv_rows(change_tiles_csv)
    write_csv_rows(
        labels,
        ["pair_id", "tile_x", "tile_y", "label"],
        [
            {"pair_id": tile_rows[0]["pair_id"], "tile_x": tile_rows[0]["tile_x"], "tile_y": tile_rows[0]["tile_y"], "label": 1}
        ],
    )
    metrics_csv = tmp_path / "metrics.csv"
    evaluate_change_metrics(change_tiles_csv, labels, metrics_csv)
    assert read_csv_rows(metrics_csv)[0]["score_type"] == "change_score"
