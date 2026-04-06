from pathlib import Path

from compression.io import write_csv_rows
from compression.pairs import build_facade_pairs, read_observations


def test_pair_csv_generation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"facade_id": "facadeA", "year": "2020", "image_path": "/tmp/facadeA_2020.png", "split": "train"},
        {"facade_id": "facadeA", "year": "2021", "image_path": "/tmp/facadeA_2021.png", "split": "train"},
        {"facade_id": "facadeA", "year": "2022", "image_path": "/tmp/facadeA_2022.png", "split": "train"},
    ]
    write_csv_rows(manifest, ["facade_id", "year", "image_path", "split"], rows)

    observations = read_observations(manifest)
    pairs = build_facade_pairs(observations, pair_mode="consecutive")

    assert len(pairs) == 2
    assert pairs[0]["pair_id"] == "facadeA_2020_2021"
    assert pairs[1]["pair_id"] == "facadeA_2021_2022"
