#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.io import read_csv_rows, write_csv_rows
from compression.lm import ByteEntropyModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate residual byte entropy baseline")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    model = ByteEntropyModel.load(args.model_path)
    rows = read_csv_rows(args.residual_manifest)
    selected = [row for row in rows if row.get("split", "train") == args.split]

    out_rows = []
    for row in selected:
        payload = Path(row["residual_path"]).read_bytes()
        nll_bits = model.nll_bits(payload)
        out_rows.append(
            {
                "pair_id": row["pair_id"],
                "split": row.get("split", "train"),
                "nll_bits": nll_bits,
                "num_bytes": len(payload),
                "bits_per_byte": nll_bits / max(len(payload), 1),
            }
        )

    write_csv_rows(args.out_csv, ["pair_id", "split", "nll_bits", "num_bytes", "bits_per_byte"], out_rows)
    print(f"Evaluated {len(out_rows)} samples on split={args.split}")


if __name__ == "__main__":
    main()
