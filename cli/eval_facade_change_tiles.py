#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.tiles import eval_change_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tile-wise facade change heatmaps")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--out-scores-csv", type=Path, required=True)
    parser.add_argument("--heatmap-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=32)
    args = parser.parse_args()

    rows = eval_change_tiles(args.residual_manifest, args.out_scores_csv, args.heatmap_dir, tile_size=args.tile_size)
    print(f"Wrote {len(rows)} tile rows")


if __name__ == "__main__":
    main()
