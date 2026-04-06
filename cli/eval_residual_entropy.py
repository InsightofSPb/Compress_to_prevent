#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.lm.workflow import eval_entropy_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate residual entropy baseline")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-out-csv", type=Path, default=None)
    args = parser.parse_args()

    summary = eval_entropy_model(
        residual_manifest=args.residual_manifest,
        model_path=args.model_path,
        split=args.split,
        out_csv=args.out_csv,
        tile_size=args.tile_size,
        tile_out_csv=args.tile_out_csv,
    )
    print(f"Evaluation summary: {summary}")


if __name__ == "__main__":
    main()
