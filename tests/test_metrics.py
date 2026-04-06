from pathlib import Path

from compression.io import read_csv_rows, write_csv_rows
from compression.metrics import evaluate_change_metrics


def test_change_metric_output_schema(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    labels_csv = tmp_path / "labels.csv"
    out_csv = tmp_path / "metrics.csv"

    write_csv_rows(
        scores_csv,
        ["pair_id", "tile_x", "tile_y", "tile_score"],
        [
            {"pair_id": "p", "tile_x": 0, "tile_y": 0, "tile_score": 0.9},
            {"pair_id": "p", "tile_x": 1, "tile_y": 0, "tile_score": 0.1},
        ],
    )
    write_csv_rows(
        labels_csv,
        ["pair_id", "tile_x", "tile_y", "label"],
        [
            {"pair_id": "p", "tile_x": 0, "tile_y": 0, "label": 1},
            {"pair_id": "p", "tile_x": 1, "tile_y": 0, "label": 0},
        ],
    )

    evaluate_change_metrics(scores_csv, labels_csv, out_csv)
    rows = read_csv_rows(out_csv)
    assert len(rows) == 1
    assert set(rows[0].keys()) == {"n_tiles", "roc_auc", "average_precision"}
