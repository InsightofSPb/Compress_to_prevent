#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.io import ensure_dir
from compression.pairs import build_facade_pairs, read_observations, write_split_pair_csvs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal facade pair CSVs")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pair-mode", choices=["consecutive", "all_to_latest"], default="consecutive")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    observations = read_observations(args.manifest_csv)
    pairs = build_facade_pairs(observations, pair_mode=args.pair_mode)
    write_split_pair_csvs(pairs, args.out_dir)
    print(f"Built {len(pairs)} pairs in {args.out_dir}")


if __name__ == "__main__":
    main()
