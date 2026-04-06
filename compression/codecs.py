from __future__ import annotations

import io
import lzma
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .io import load_rgb_image, read_csv_rows, write_csv_rows

try:
    import zstandard as zstd  # type: ignore
except Exception:  # pragma: no cover
    zstd = None


def compress_with_zstd(payload: bytes, level: int) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard package is required for zstd benchmarking")
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(payload)


def compress_with_lzma(payload: bytes, level: int) -> bytes:
    preset = max(0, min(level, 9))
    return lzma.compress(payload, preset=preset)


def compress_with_webp(width: int, height: int, rgb_payload: bytes, level: int) -> bytes:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow with WebP support is required for webp benchmarking") from exc

    image = Image.frombytes("RGB", (width, height), rgb_payload)
    sink = io.BytesIO()
    image.save(sink, format="WEBP", lossless=True, method=max(0, min(level, 6)), quality=100)
    return sink.getvalue()


def _fnlic_lite_predictive_transform(width: int, height: int, rgb_payload: bytes) -> bytes:
    out = bytearray(len(rgb_payload))
    for y in range(height):
        for x in range(width):
            for c in range(3):
                idx = (y * width + x) * 3 + c
                left = out[idx - 3] if x > 0 else 0
                up = out[idx - width * 3] if y > 0 else 0
                pred = (left + up) // 2
                out[idx] = (rgb_payload[idx] - pred) % 256
    return bytes(out)


def compress_with_fnlic_lite(width: int, height: int, rgb_payload: bytes, level: int) -> bytes:
    transformed = _fnlic_lite_predictive_transform(width, height, rgb_payload)
    if zstd is not None:
        return compress_with_zstd(transformed, level)
    return compress_with_lzma(transformed, level)


def _benchmark_method(method: str, level: int, width: int, height: int, payload: bytes) -> Tuple[Optional[int], str, str]:
    if method == "zstd":
        compressed = compress_with_zstd(payload, level)
        return len(compressed) * 8, "ok", ""
    if method == "lzma":
        compressed = compress_with_lzma(payload, level)
        return len(compressed) * 8, "ok", ""
    if method == "webp":
        compressed = compress_with_webp(width, height, payload, level)
        return len(compressed) * 8, "ok", "image_domain_lossless_webp"
    if method == "fnlic":
        compressed = compress_with_fnlic_lite(width, height, payload, level)
        note = "fnlic_lite_predictive+zstd" if zstd is not None else "fnlic_lite_predictive+lzma_fallback"
        return len(compressed) * 8, "ok", note
    raise ValueError(f"Unsupported method: {method}")


def benchmark_residual_codecs(
    residual_manifest_csv: Path,
    output_csv: Path,
    methods: Iterable[str],
    level: int = 3,
    strict: bool = False,
) -> List[Dict[str, object]]:
    rows = read_csv_rows(residual_manifest_csv)
    out_rows: List[Dict[str, object]] = []

    for row in rows:
        width, height, payload = load_rgb_image(Path(row["residual_path"]))
        payload_bits = len(payload) * 8
        for method in methods:
            status = "ok"
            notes = ""
            achieved_bits: Optional[int] = None
            try:
                achieved_bits, status, notes = _benchmark_method(method, level, width, height, payload)
            except Exception as exc:
                if strict:
                    raise
                status = "unsupported"
                notes = str(exc)

            out_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "split": row.get("split", "train"),
                    "method": method,
                    "method_level": level,
                    "score_type": "achieved_bits",
                    "bit_length": "" if achieved_bits is None else achieved_bits,
                    "payload_bits": payload_bits,
                    "status": status,
                    "notes": notes,
                }
            )

    fields = [
        "pair_id",
        "split",
        "method",
        "method_level",
        "score_type",
        "bit_length",
        "payload_bits",
        "status",
        "notes",
    ]
    write_csv_rows(output_csv, fields, out_rows)
    return out_rows
