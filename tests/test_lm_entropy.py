from pathlib import Path

from compression.io import read_csv_rows, write_csv_rows
from compression.lm.workflow import eval_entropy_model, train_entropy_model


def _write_ppm(path: Path, width: int, height: int, value: int) -> None:
    payload = bytes([value] * (width * height * 3))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)


def test_stronger_baseline_output_schema(tmp_path: Path) -> None:
    img_a = tmp_path / "a.ppm"
    img_b = tmp_path / "b.ppm"
    _write_ppm(img_a, 4, 4, 1)
    _write_ppm(img_b, 4, 4, 2)

    manifest = tmp_path / "residual_manifest.csv"
    write_csv_rows(
        manifest,
        ["pair_id", "split", "residual_path"],
        [
            {"pair_id": "p_train", "split": "train", "residual_path": str(img_a)},
            {"pair_id": "p_val", "split": "val", "residual_path": str(img_b)},
        ],
    )

    model_path = tmp_path / "model.json"
    train_entropy_model(manifest, model_path, split="train", model_mode="bigram", alpha=0.5)
    sample_csv = tmp_path / "eval.csv"
    tile_csv = tmp_path / "eval_tiles.csv"
    eval_entropy_model(manifest, model_path, split="val", out_csv=sample_csv, tile_size=2, tile_out_csv=tile_csv)

    sample_rows = read_csv_rows(sample_csv)
    tile_rows = read_csv_rows(tile_csv)
    assert sample_rows[0]["score_type"] == "model_bits"
    assert set(sample_rows[0].keys()) == {
        "pair_id",
        "split",
        "score_type",
        "bit_length",
        "num_symbols",
        "bits_per_symbol",
    }
    assert set(tile_rows[0].keys()) == {
        "pair_id",
        "split",
        "score_type",
        "tile_x",
        "tile_y",
        "bit_length",
        "bits_per_symbol",
        "tile_size",
    }
