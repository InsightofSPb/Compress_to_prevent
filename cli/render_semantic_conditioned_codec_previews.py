#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.conditioned.previews import render_semantic_conditioned_codec_previews


def main() -> None:
    parser = argparse.ArgumentParser(description="Render conditioned vs unconditioned C1 previews")
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--conditioned-tile-csv", type=Path, required=True)
    parser.add_argument("--unconditioned-tile-csv", type=Path, required=True)
    parser.add_argument("--semantic-features-csv", type=Path, default=None)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rendered = render_semantic_conditioned_codec_previews(
        pairs_csv=args.pairs_csv,
        conditioned_tile_csv=args.conditioned_tile_csv,
        unconditioned_tile_csv=args.unconditioned_tile_csv,
        semantic_features_csv=args.semantic_features_csv,
        out_dir=args.out_dir,
        tile_size=args.tile_size,
    )
    print(f"Rendered {len(rendered)} preview artifacts")


if __name__ == "__main__":
    main()
