#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.io import read_csv_rows
from compression.lm import ByteEntropyModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual byte entropy baseline")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    rows = read_csv_rows(args.residual_manifest)
    train_rows = [row for row in rows if row.get("split", "train") == args.train_split]
    payloads = [Path(row["residual_path"]).read_bytes() for row in train_rows]

    model = ByteEntropyModel(alpha=args.alpha)
    model.fit(payloads)
    model.save(args.model_out)
    print(f"Trained model on {len(train_rows)} samples and saved to {args.model_out}")


if __name__ == "__main__":
    main()
