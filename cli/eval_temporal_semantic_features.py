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
from temporal_semantics.evaluation import evaluate_temporal_semantic_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate temporal semantic tile features")
    parser.add_argument("--features-csv", type=Path)
    parser.add_argument("--out-summary-csv", type=Path)
    parser.add_argument("--out-topk-csv", type=Path)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--list-backends", action="store_true")
    args = parser.parse_args()

    if args.list_backends:
        print(json.dumps(list_available_backends(), indent=2))
        return

    if args.features_csv is None or args.out_summary_csv is None or args.out_topk_csv is None:
        raise SystemExit("--features-csv, --out-summary-csv and --out-topk-csv are required unless --list-backends is used")

    summary = evaluate_temporal_semantic_features(
        features_csv=args.features_csv,
        out_summary_csv=args.out_summary_csv,
        topk_csv=args.out_topk_csv,
        labels_csv=args.labels_csv,
        top_k=args.top_k,
    )
    print(f"Evaluation summary: {summary}")


if __name__ == "__main__":
    main()
