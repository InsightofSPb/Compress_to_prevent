from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from compression.io import load_rgb_image, read_csv_rows
from temporal_semantics.tiling import generate_tiles

from .context import build_pair_context_index
from .types import TileSample


def _tile_bytes(image: tuple[int, int, bytes], x0: int, y0: int, x1: int, y1: int) -> bytes:
    w, _, payload = image
    out = bytearray()
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            base = (yy * w + xx) * 3
            out.extend(payload[base : base + 3])
    return bytes(out)


def build_conditioned_samples(
    residual_manifest_csv: Path,
    pairs_csv: Path,
    artifact_index_csv: Path,
    temporal_features_csv: Path | None,
    tile_size: int,
    context_mode: str,
    context_dim: int,
    custom_sources: Sequence[str] | None = None,
) -> List[TileSample]:
    residual_rows = read_csv_rows(residual_manifest_csv)
    pair_rows = {row["pair_id"]: row for row in read_csv_rows(pairs_csv)}

    context_index = build_pair_context_index(
        pairs_csv=pairs_csv,
        artifact_index_csv=artifact_index_csv,
        temporal_features_csv=temporal_features_csv,
        context_mode=context_mode,
        context_dim=context_dim,
        tile_size=tile_size,
        custom_sources=custom_sources,
    )

    samples: List[TileSample] = []
    for row in residual_rows:
        pair_id = row["pair_id"]
        pair = pair_rows.get(pair_id, {})
        facade_id = row.get("facade_id") or pair.get("facade_id", "")
        split = row.get("split", "train")
        image = load_rgb_image(Path(row["residual_path"]))
        w, h, _ = image

        for tile in generate_tiles(w, h, tile_size=tile_size):
            context_key = f"{pair_id}::{tile.tile_id}"
            c_payload = context_index.get(context_key, {})
            samples.append(
                TileSample(
                    pair_id=pair_id,
                    facade_id=facade_id,
                    split=split,
                    tile_id=tile.tile_id,
                    tile_x=tile.x0 // tile_size,
                    tile_y=tile.y0 // tile_size,
                    x0=tile.x0,
                    y0=tile.y0,
                    x1=tile.x1,
                    y1=tile.y1,
                    residual_bytes=_tile_bytes(image, tile.x0, tile.y0, tile.x1, tile.y1),
                    context_vector=list(c_payload.get("context_vector", [0.0] * context_dim)),
                    context_mode=context_mode,
                    context_sources=list(c_payload.get("context_sources", [])),
                    context_backends=list(c_payload.get("context_backends", [])),
                )
            )
    return samples
