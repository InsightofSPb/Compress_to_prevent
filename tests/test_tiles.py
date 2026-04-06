from pathlib import Path

from compression.io import read_csv_rows, write_csv_rows
from compression.tiles import eval_change_tiles, tile_scores


def test_tile_indexing_determinism() -> None:
    width, height = 8, 8
    residual = (width, height, bytes([0] * (width * height * 3)))
    first = tile_scores(residual, tile_size=4)
    second = tile_scores(residual, tile_size=4)
    assert first == second
    assert [item[:2] for item in first] == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_tile_heatmap_output_schema(tmp_path: Path) -> None:
    ppm = tmp_path / "r.ppm"
    ppm.write_bytes(b"P6\n4 4\n255\n" + bytes([0] * 4 * 4 * 3))
    manifest = tmp_path / "manifest.csv"
    write_csv_rows(manifest, ["pair_id", "split", "residual_path"], [{"pair_id": "p", "split": "val", "residual_path": str(ppm)}])

    out_csv = tmp_path / "tiles.csv"
    heat_dir = tmp_path / "heat"
    eval_change_tiles(manifest, out_csv, heat_dir, tile_size=2)
    rows = read_csv_rows(out_csv)
    assert set(rows[0].keys()) == {"pair_id", "split", "score_type", "tile_x", "tile_y", "tile_score", "tile_size", "heatmap_pgm"}
    assert rows[0]["score_type"] == "change_score"
    assert Path(rows[0]["heatmap_pgm"]).exists()
