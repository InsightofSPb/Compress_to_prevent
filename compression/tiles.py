from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .io import load_rgb_image, read_csv_rows, write_csv_rows


def residual_byte_to_signed(value: int) -> int:
    return ((value + 128) % 256) - 128


def tile_scores(residual_rgb: tuple[int, int, bytes], tile_size: int) -> List[Tuple[int, int, float]]:
    width, height, payload = residual_rgb
    out: List[Tuple[int, int, float]] = []

    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            acc = 0.0
            count = 0
            for yy in range(y0, min(y0 + tile_size, height)):
                for xx in range(x0, min(x0 + tile_size, width)):
                    base = (yy * width + xx) * 3
                    for channel in range(3):
                        acc += abs(residual_byte_to_signed(payload[base + channel]))
                        count += 1
            out.append((x0 // tile_size, y0 // tile_size, acc / max(count, 1)))
    return out


def _save_heatmap_pgm(path: Path, heatmap: List[List[float]]) -> None:
    h = len(heatmap)
    w = len(heatmap[0]) if h > 0 else 0
    mx = max((v for row in heatmap for v in row), default=1.0)
    if mx <= 0:
        mx = 1.0
    payload = bytearray()
    for row in heatmap:
        for value in row:
            payload.append(int((value / mx) * 255))
    path.write_bytes(f"P5\n{w} {h}\n255\n".encode("ascii") + bytes(payload))


def eval_change_tiles(
    residual_manifest_csv: Path,
    out_scores_csv: Path,
    heatmap_dir: Path,
    tile_size: int,
) -> List[Dict[str, object]]:
    rows = read_csv_rows(residual_manifest_csv)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, object]] = []
    for row in rows:
        residual = load_rgb_image(Path(row["residual_path"]))
        scores = tile_scores(residual, tile_size=tile_size)

        max_x = max(tile_x for tile_x, _, _ in scores)
        max_y = max(tile_y for _, tile_y, _ in scores)
        heatmap = [[0.0 for _ in range(max_x + 1)] for _ in range(max_y + 1)]
        for tile_x, tile_y, score in scores:
            heatmap[tile_y][tile_x] = score

        pgm_path = heatmap_dir / f"{row['pair_id']}.pgm"
        _save_heatmap_pgm(pgm_path, heatmap)

        for tile_x, tile_y, score in scores:
            out_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "split": row.get("split", "train"),
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "score_type": "change_score",
                    "tile_score": score,
                    "tile_size": tile_size,
                    "heatmap_pgm": str(pgm_path),
                }
            )

    fields = ["pair_id", "split", "score_type", "tile_x", "tile_y", "tile_score", "tile_size", "heatmap_pgm"]
    write_csv_rows(out_scores_csv, fields, out_rows)
    return out_rows
