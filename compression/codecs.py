from __future__ import annotations

import lzma
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .io import read_csv_rows, write_csv_rows

try:
    import zstandard as zstd  # type: ignore
except Exception:  # pragma: no cover - dependency optional in CI
    zstd = None


def compress_with_zstd(payload: bytes, level: int) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard package is required for zstd benchmarking")
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(payload)


def compress_with_lzma(payload: bytes, level: int) -> bytes:
    preset = max(0, min(level, 9))
    return lzma.compress(payload, preset=preset)


def compress_payload(payload: bytes, codec: str, level: int) -> Optional[bytes]:
    if codec == "zstd":
        return compress_with_zstd(payload, level)
    if codec == "lzma":
        return compress_with_lzma(payload, level)
    if codec in {"webp", "fnlic"}:
        return None
    raise ValueError(f"Unsupported codec: {codec}")


def benchmark_residual_codecs(
    residual_manifest_csv: Path,
    output_csv: Path,
    codecs: Iterable[str],
    level: int = 3,
) -> List[Dict[str, object]]:
    rows = read_csv_rows(residual_manifest_csv)
    out_rows: List[Dict[str, object]] = []

    for row in rows:
        payload = Path(row["residual_path"]).read_bytes()
        for codec_name in codecs:
            compressed = compress_payload(payload, codec_name, level)
            achieved_bits = "" if compressed is None else len(compressed) * 8
            out_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "split": row.get("split", "train"),
                    "codec": codec_name,
                    "level": level,
                    "payload_bytes": len(payload),
                    "achieved_bits": achieved_bits,
                    "model_bits": "",
                }
            )

    fields = ["pair_id", "split", "codec", "level", "payload_bytes", "achieved_bits", "model_bits"]
    write_csv_rows(output_csv, fields, out_rows)
    return out_rows
