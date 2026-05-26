import pytest
pytest.importorskip("numpy")

from pathlib import Path

from compression.baselines import compute_baseline_tile_scores
from compression.io import read_csv_rows, write_csv_rows
from compression.metrics import evaluate_change_metrics
from compression.tiles import eval_change_tiles


def _write_ppm(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    pix = bytes(list(rgb) * width * height)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pix)


def test_synthetic_smoke_suite(tmp_path: Path) -> None:
    prev = tmp_path / "prev.ppm"
    curr = tmp_path / "curr.ppm"
    residual = tmp_path / "residual.ppm"

    _write_ppm(prev, 8, 8, (10, 10, 10))
    _write_ppm(curr, 8, 8, (10, 10, 10))

    raw = bytearray(curr.read_bytes())
    header_end = raw.find(b"255\n") + 4
    payload = raw[header_end:]
    for y in range(0, 4):
        for x in range(0, 4):
            i = (y * 8 + x) * 3
            payload[i:i+3] = bytes([200, 200, 200])
    curr.write_bytes(raw[:header_end] + payload)
    curr_arr = curr.read_bytes()[header_end:]
    prev_arr = prev.read_bytes()[header_end:]
    res = bytes(((int(c)-int(p)) % 256) for p, c in zip(prev_arr, curr_arr))
    residual.write_bytes(f"P6\n8 8\n255\n".encode("ascii") + res)

    manifest = tmp_path / "residual_manifest.csv"
    write_csv_rows(
        manifest,
        ["pair_id", "facade_id", "split", "residual_path", "prev_aligned_path", "curr_image_path", "height", "width"],
        [{"pair_id": "p1", "facade_id": "f1", "split": "val", "residual_path": str(residual), "prev_aligned_path": str(prev), "curr_image_path": str(curr), "height": 8, "width": 8}],
    )

    proposed_csv = tmp_path / "proposed.csv"
    eval_change_tiles(manifest, proposed_csv, tmp_path / "heatmaps", tile_size=4)
    baseline_csv = tmp_path / "baseline.csv"
    compute_baseline_tile_scores(manifest, baseline_csv, methods=["absdiff_l1"], tile_size=4, skip_deep_baselines=True)

    labels = tmp_path / "labels.csv"
    write_csv_rows(labels, ["pair_id", "tile_x", "tile_y", "label"], [
        {"pair_id": "p1", "tile_x": 0, "tile_y": 0, "label": 1},
        {"pair_id": "p1", "tile_x": 1, "tile_y": 1, "label": 0},
    ])
    metrics_csv = tmp_path / "metrics.csv"
    evaluate_change_metrics(baseline_csv, labels, metrics_csv)

    assert proposed_csv.exists() and baseline_csv.exists() and metrics_csv.exists()
    metrics_rows = read_csv_rows(metrics_csv)
    assert len(metrics_rows) >= 1
    baseline_rows = read_csv_rows(baseline_csv)
    changed = [r for r in baseline_rows if r["tile_x"] == "0" and r["tile_y"] == "0"][0]
    unchanged = [r for r in baseline_rows if r["tile_x"] == "1" and r["tile_y"] == "1"][0]
    assert float(changed["tile_score"]) > float(unchanged["tile_score"])
