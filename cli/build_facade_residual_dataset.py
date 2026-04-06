#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.residuals import build_residual_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build facade residual dataset")
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    rows = build_residual_dataset(args.pairs_csv, args.out_root)
    print(f"Built {len(rows)} residual samples under {args.out_root}")


if __name__ == "__main__":
    main()
