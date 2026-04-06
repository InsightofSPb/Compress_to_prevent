from pathlib import Path

from compression.io import read_csv_rows, write_csv_rows
from compression.residuals import build_residual_dataset


def _write_ppm(path: Path, width: int, height: int, value: int) -> None:
    payload = bytes([value] * (width * height * 3))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)


def test_residual_manifest_consistency(tmp_path: Path) -> None:
    prev_path = tmp_path / "prev.ppm"
    curr_path = tmp_path / "curr.ppm"
    _write_ppm(prev_path, 4, 4, 0)
    _write_ppm(curr_path, 4, 4, 10)

    pairs_csv = tmp_path / "pairs.csv"
    write_csv_rows(
        pairs_csv,
        ["pair_id", "facade_id", "prev_image_path", "curr_image_path", "prev_aligned_path", "split"],
        [
            {
                "pair_id": "p1",
                "facade_id": "f1",
                "prev_image_path": str(prev_path),
                "curr_image_path": str(curr_path),
                "prev_aligned_path": str(prev_path),
                "split": "val",
            }
        ],
    )

    out_root = tmp_path / "residuals"
    build_residual_dataset(pairs_csv, out_root)
    manifest = read_csv_rows(out_root / "residual_manifest.csv")
    assert len(manifest) == 1
    assert manifest[0]["pair_id"] == "p1"
    assert Path(manifest[0]["residual_path"]).exists()
    assert manifest[0]["height"] == "4"
    assert manifest[0]["width"] == "4"
