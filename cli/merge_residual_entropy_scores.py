#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.lm.workflow import merge_entropy_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge residual entropy component scores")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    n_rows = merge_entropy_scores(args.inputs, args.out_csv)
    print(f"Merged into {n_rows} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
