#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal_semantics.artifacts import export_semantic_artifacts, list_available_backends


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-image temporal semantic artifacts")
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--backends", type=str, default="lposs,dinov2,clip,siglip2")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-backends", action="store_true")
    args = parser.parse_args()

    if args.list_backends:
        print(json.dumps(list_available_backends(), indent=2))
        return

    if args.manifest_csv is None or args.out_dir is None:
        raise SystemExit("--manifest-csv and --out-dir are required unless --list-backends is used")

    rows = export_semantic_artifacts(
        manifest_csv=args.manifest_csv,
        backends=[v.strip() for v in args.backends.split(",") if v.strip()],
        out_dir=args.out_dir,
        tile_size=args.tile_size,
        force=args.force,
    )
    print(f"Exported/indexed {len(rows)} artifact rows")


if __name__ == "__main__":
    main()
