from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from compression.io import load_rgb_image, read_csv_rows

from .visualize import render_heatmap_pgm, render_tile_overlay


def _tile_grid_dims(width: int, height: int, tile_size: int) -> tuple[int, int]:
    return (width + tile_size - 1) // tile_size, (height + tile_size - 1) // tile_size


def render_temporal_semantic_previews(
    features_csv: Path,
    pairs_csv: Path,
    out_dir: Path,
    tile_size: int,
    include_fused: bool = False,
) -> List[Path]:
    rows = read_csv_rows(features_csv)
    pairs = {row["pair_id"]: row for row in read_csv_rows(pairs_csv)}
    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair_id"], row["backend"])].append(row)

    rendered: List[Path] = []
    for (pair_id, backend), group in grouped.items():
        pair = pairs[pair_id]
        w, h, rgb = load_rgb_image(Path(pair["curr_image_path"]))
        grid_w, grid_h = _tile_grid_dims(w, h, tile_size)

        values = [0.0] * (grid_w * grid_h)
        tile_scores: Dict[str, float] = {}
        for row in group:
            tx = int(row["x0"]) // tile_size
            ty = int(row["y0"]) // tile_size
            score = float(row["semantic_score_backend"])
            values[ty * grid_w + tx] = score
            tile_scores[f"{row['x0']}_{row['y0']}"] = score

        backend_dir = out_dir / backend
        heat_path = backend_dir / f"{pair_id}_heatmap.pgm"
        overlay_path = backend_dir / f"{pair_id}_overlay.ppm"
        render_heatmap_pgm(heat_path, grid_w, grid_h, values)
        render_tile_overlay(overlay_path, (w, h, rgb), tile_scores, tile_size=tile_size)
        rendered.extend([heat_path, overlay_path])

    if include_fused:
        by_pair: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_pair[row["pair_id"]].append(row)
        for pair_id, group in by_pair.items():
            pair = pairs[pair_id]
            w, h, rgb = load_rgb_image(Path(pair["curr_image_path"]))
            grid_w, grid_h = _tile_grid_dims(w, h, tile_size)
            fused_values = [0.0] * (grid_w * grid_h)
            fused_scores: Dict[str, float] = {}
            seen = set()
            for row in group:
                tile_key = (row["x0"], row["y0"])
                if tile_key in seen:
                    continue
                seen.add(tile_key)
                tx = int(row["x0"]) // tile_size
                ty = int(row["y0"]) // tile_size
                score = float(row["semantic_score_fused"])
                fused_values[ty * grid_w + tx] = score
                fused_scores[f"{row['x0']}_{row['y0']}"] = score
            fused_dir = out_dir / "fused"
            heat_path = fused_dir / f"{pair_id}_heatmap_fused.pgm"
            overlay_path = fused_dir / f"{pair_id}_overlay_fused.ppm"
            render_heatmap_pgm(heat_path, grid_w, grid_h, fused_values)
            render_tile_overlay(overlay_path, (w, h, rgb), fused_scores, tile_size=tile_size)
            rendered.extend([heat_path, overlay_path])

    return rendered
