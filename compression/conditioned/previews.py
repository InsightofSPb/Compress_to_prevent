from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from compression.io import load_rgb_image, read_csv_rows
from temporal_semantics.visualize import render_heatmap_pgm, render_tile_overlay


def render_semantic_conditioned_codec_previews(
    pairs_csv: Path,
    conditioned_tile_csv: Path,
    unconditioned_tile_csv: Path,
    semantic_features_csv: Path | None,
    out_dir: Path,
    tile_size: int,
) -> List[Path]:
    pairs = {row["pair_id"]: row for row in read_csv_rows(pairs_csv)}
    cond = read_csv_rows(conditioned_tile_csv)
    unc = read_csv_rows(unconditioned_tile_csv)
    sem_rows = read_csv_rows(semantic_features_csv) if semantic_features_csv is not None and semantic_features_csv.exists() else []

    by_pair_cond: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_pair_unc: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_pair_sem: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in cond:
        by_pair_cond[row["pair_id"]].append(row)
    for row in unc:
        by_pair_unc[row["pair_id"]].append(row)
    for row in sem_rows:
        by_pair_sem[row["pair_id"]].append(row)

    rendered: List[Path] = []
    for pair_id, pair in pairs.items():
        if pair_id not in by_pair_cond or pair_id not in by_pair_unc:
            continue
        w, h, rgb = load_rgb_image(Path(pair["curr_image_path"]))
        grid_w = (w + tile_size - 1) // tile_size
        grid_h = (h + tile_size - 1) // tile_size

        def to_grid(rows: List[Dict[str, str]], key: str) -> List[float]:
            vals = [0.0] * (grid_w * grid_h)
            for r in rows:
                tx, ty = int(r["tile_x"]), int(r["tile_y"])
                vals[ty * grid_w + tx] = float(r[key])
            return vals

        v_cond = to_grid(by_pair_cond[pair_id], "bits_per_byte")
        v_unc = to_grid(by_pair_unc[pair_id], "bits_per_byte")
        v_diff = [a - b for a, b in zip(v_cond, v_unc)]
        sem_vals = [0.0] * (grid_w * grid_h)
        for row in by_pair_sem.get(pair_id, []):
            tx = int(row["x0"]) // tile_size
            ty = int(row["y0"]) // tile_size
            sem_vals[ty * grid_w + tx] = float(row.get("semantic_score_fused") or 0.0)

        pair_dir = out_dir / pair_id
        heat_unc = pair_dir / "unconditioned_heatmap.pgm"
        heat_cond = pair_dir / "conditioned_heatmap.pgm"
        heat_diff = pair_dir / "conditioned_minus_unconditioned.pgm"
        heat_sem = pair_dir / "semantic_fused_heatmap.pgm"
        overlay = pair_dir / "conditioned_overlay.ppm"

        render_heatmap_pgm(heat_unc, grid_w, grid_h, v_unc)
        render_heatmap_pgm(heat_cond, grid_w, grid_h, v_cond)
        render_heatmap_pgm(heat_diff, grid_w, grid_h, [abs(v) for v in v_diff])
        render_heatmap_pgm(heat_sem, grid_w, grid_h, sem_vals)
        tile_scores = {f"{int(r['tile_x']) * tile_size}_{int(r['tile_y']) * tile_size}": float(r["bits_per_byte"]) for r in by_pair_cond[pair_id]}
        render_tile_overlay(overlay, (w, h, rgb), tile_scores=tile_scores, tile_size=tile_size)

        rendered.extend([heat_unc, heat_cond, heat_diff, heat_sem, overlay])
    return rendered
