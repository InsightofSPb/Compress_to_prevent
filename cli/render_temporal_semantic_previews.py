#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal_semantics.previews import render_temporal_semantic_previews


def main() -> None:
    parser = argparse.ArgumentParser(description="Render temporal semantic heatmaps and overlays")
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=32)
    args = parser.parse_args()

    outputs = render_temporal_semantic_previews(
        features_csv=args.features_csv,
        pairs_csv=args.pairs_csv,
        out_dir=args.out_dir,
        tile_size=args.tile_size,
    )
    print(f"Rendered {len(outputs)} preview files")


if __name__ == "__main__":
    main()
