from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from compression.io import save_rgb_image


def render_heatmap_pgm(path: Path, width: int, height: int, values: List[float]) -> None:
    mx = max(values) if values else 1.0
    if mx <= 0:
        mx = 1.0
    payload = bytes(int((v / mx) * 255) for v in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + payload)


def render_tile_overlay(path: Path, image: tuple[int, int, bytes], tile_scores: Dict[str, float], tile_size: int) -> None:
    w, h, rgb = image
    out = bytearray(rgb)
    mx = max(tile_scores.values()) if tile_scores else 1.0
    if mx <= 0:
        mx = 1.0
    for key, score in tile_scores.items():
        x0, y0 = [int(v) for v in key.split("_")[:2]]
        intensity = int((score / mx) * 255)
        for yy in range(y0, min(y0 + tile_size, h)):
            for xx in range(x0, min(x0 + tile_size, w)):
                idx = (yy * w + xx) * 3
                out[idx] = max(out[idx], intensity)
    save_rgb_image(path, (w, h, bytes(out)))
