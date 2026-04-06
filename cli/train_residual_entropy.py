#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.lm.workflow import train_entropy_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual entropy baseline")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--model-mode", type=str, choices=["unigram", "bigram"], default="bigram")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    summary = train_entropy_model(
        residual_manifest=args.residual_manifest,
        model_out=args.model_out,
        split=args.train_split,
        model_mode=args.model_mode,
        alpha=args.alpha,
    )
    print(f"Training summary: {summary}")


if __name__ == "__main__":
    main()
