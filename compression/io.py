from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

RGBImage = Tuple[int, int, bytes]  # width, height, RGB interleaved bytes


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_ppm(path: Path) -> RGBImage:
    raw = path.read_bytes()
    if not raw.startswith(b"P6"):
        raise ValueError(f"Unsupported PPM header in {path}")
    parts = raw.split(b"\n", 3)
    if len(parts) < 4:
        raise ValueError(f"Malformed PPM file: {path}")
    _, dims, maxv, payload = parts
    width_s, height_s = dims.decode("ascii").split()
    width, height = int(width_s), int(height_s)
    if maxv.strip() != b"255":
        raise ValueError(f"Only 8-bit PPM supported, got maxv={maxv!r}")
    expected = width * height * 3
    if len(payload) != expected:
        raise ValueError(f"PPM payload size mismatch: expected={expected}, got={len(payload)}")
    return width, height, payload


def _save_ppm(path: Path, image: RGBImage) -> None:
    width, height, payload = image
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)


def load_rgb_image(path: Path) -> RGBImage:
    if path.suffix.lower() in {".ppm"}:
        return _load_ppm(path)

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for non-PPM image I/O") from exc

    image = Image.open(path).convert("RGB")
    width, height = image.size
    return width, height, image.tobytes()


def save_rgb_image(path: Path, image: RGBImage) -> None:
    if path.suffix.lower() in {".ppm"}:
        _save_ppm(path, image)
        return

    width, height, payload = image
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for non-PPM image I/O") from exc
    Image.frombytes("RGB", (width, height), payload).save(path)
