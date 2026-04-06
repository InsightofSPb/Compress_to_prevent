from pathlib import Path

from compression.codecs import benchmark_residual_codecs
from compression.io import write_csv_rows


def _write_ppm(path: Path, width: int, height: int, value: int) -> None:
    payload = bytes([value] * (width * height * 3))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)


def test_codec_benchmark_csv_schema(tmp_path: Path) -> None:
    residual_path = tmp_path / "r.ppm"
    _write_ppm(residual_path, 4, 4, 0)

    manifest = tmp_path / "residual_manifest.csv"
    write_csv_rows(
        manifest,
        ["pair_id", "split", "residual_path"],
        [{"pair_id": "p1", "split": "test", "residual_path": str(residual_path)}],
    )

    out_csv = tmp_path / "bench.csv"
    rows = benchmark_residual_codecs(manifest, out_csv, codecs=["lzma"], level=1)

    assert len(rows) == 1
    assert set(rows[0].keys()) == {"pair_id", "split", "codec", "level", "payload_bytes", "achieved_bits", "model_bits"}
