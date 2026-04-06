#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.metrics import evaluate_change_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tile-wise facade change metrics")
    parser.add_argument("--tile-scores-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    summary = evaluate_change_metrics(args.tile_scores_csv, args.labels_csv, args.out_csv)
    print(f"Computed metrics: {summary}")


if __name__ == "__main__":
    main()
