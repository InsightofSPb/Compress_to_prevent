#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.io import read_csv_rows, write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge residual entropy component scores")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    merged = defaultdict(lambda: {"nll_bits": 0.0, "num_bytes": 0.0, "split": ""})
    for path in args.inputs:
        for row in read_csv_rows(path):
            key = row["pair_id"]
            merged[key]["nll_bits"] += float(row["nll_bits"])
            merged[key]["num_bytes"] += float(row["num_bytes"])
            merged[key]["split"] = row.get("split", merged[key]["split"])

    out_rows = []
    for pair_id, payload in merged.items():
        bits = payload["nll_bits"]
        nbytes = payload["num_bytes"]
        out_rows.append(
            {
                "pair_id": pair_id,
                "split": payload["split"],
                "nll_bits": bits,
                "num_bytes": nbytes,
                "bits_per_byte": bits / max(nbytes, 1.0),
            }
        )

    write_csv_rows(args.out_csv, ["pair_id", "split", "nll_bits", "num_bytes", "bits_per_byte"], out_rows)
    print(f"Merged {len(args.inputs)} files into {args.out_csv}")


if __name__ == "__main__":
    main()
