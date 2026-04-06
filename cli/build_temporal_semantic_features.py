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
from temporal_semantics.pairs import build_temporal_semantic_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tile-level temporal semantic features for aligned pairs")
    parser.add_argument("--pairs-csv", type=Path)
    parser.add_argument("--artifact-index-csv", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--backends", type=str, default="lposs,dinov2,clip,siglip2")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--list-backends", action="store_true")
    args = parser.parse_args()

    if args.list_backends:
        print(json.dumps(list_available_backends(), indent=2))
        return

    if args.pairs_csv is None or args.artifact_index_csv is None or args.out_csv is None:
        raise SystemExit("--pairs-csv, --artifact-index-csv and --out-csv are required unless --list-backends is used")

    rows = build_temporal_semantic_features(
        pairs_csv=args.pairs_csv,
        artifact_index_csv=args.artifact_index_csv,
        out_csv=args.out_csv,
        tile_size=args.tile_size,
        backends=[v.strip() for v in args.backends.split(",") if v.strip()],
    )
    print(f"Built {len(rows)} temporal semantic tile rows")


if __name__ == "__main__":
    main()
