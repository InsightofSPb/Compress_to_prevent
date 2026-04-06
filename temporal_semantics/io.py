from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from compression.io import load_rgb_image, read_csv_rows, save_rgb_image, write_csv_rows


def load_manifest(path: Path) -> List[Dict[str, str]]:
    return read_csv_rows(path)


def write_index(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    fields = [
        "sample_id",
        "backend",
        "image_path",
        "mask_path",
        "probs_path",
        "features_path",
        "overlay_path",
        "feature_grid_h",
        "feature_grid_w",
        "split",
        "status",
        "notes",
    ]
    write_csv_rows(path, fields, rows)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_mask_as_pgm(path: Path, width: int, height: int, values: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(values))


def load_mask_from_pgm(path: Path) -> tuple[int, int, bytes]:
    raw = path.read_bytes()
    parts = raw.split(b"\n", 3)
    _, dims, _, payload = parts
    w_s, h_s = dims.decode("ascii").split()
    return int(w_s), int(h_s), payload


def make_overlay(mask_values: Sequence[int], width: int, height: int) -> tuple[int, int, bytes]:
    palette = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    out = bytearray(width * height * 3)
    for i, cls in enumerate(mask_values):
        color = palette[cls % len(palette)]
        base = i * 3
        out[base : base + 3] = bytes(color)
    return width, height, bytes(out)


def load_image(path: Path) -> tuple[int, int, bytes]:
    return load_rgb_image(path)


def save_overlay(path: Path, width: int, height: int, payload: bytes) -> None:
    save_rgb_image(path, (width, height, payload))
