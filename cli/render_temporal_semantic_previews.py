#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal_semantics.artifacts import list_available_backends
from temporal_semantics.previews import render_temporal_semantic_previews


def main() -> None:
    parser = argparse.ArgumentParser(description="Render temporal semantic heatmaps and overlays")
    parser.add_argument("--features-csv", type=Path)
    parser.add_argument("--pairs-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--include-fused", action="store_true")
    parser.add_argument("--list-backends", action="store_true")
    args = parser.parse_args()

    if args.list_backends:
        print(json.dumps(list_available_backends(), indent=2))
        return

    if args.features_csv is None or args.pairs_csv is None or args.out_dir is None:
        raise SystemExit("--features-csv, --pairs-csv and --out-dir are required unless --list-backends is used")

    outputs = render_temporal_semantic_previews(
        features_csv=args.features_csv,
        pairs_csv=args.pairs_csv,
        out_dir=args.out_dir,
        tile_size=args.tile_size,
        include_fused=args.include_fused,
    )
    print(f"Rendered {len(outputs)} preview files")


if __name__ == "__main__":
    main()
