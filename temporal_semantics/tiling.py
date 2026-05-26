from __future__ import annotations

from typing import List

from .types import Tile


def generate_tiles(width: int, height: int, tile_size: int, stride: int | None = None) -> List[Tile]:
    step = tile_size if stride is None else stride
    tiles: List[Tile] = []
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tiles.append(
                Tile(
                    tile_id=f"{x0}_{y0}_{x1}_{y1}",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    center_x=(x0 + x1) / 2.0,
                    center_y=(y0 + y1) / 2.0,
                )
            )
    return tiles
