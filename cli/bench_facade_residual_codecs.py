#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.codecs import benchmark_residual_codecs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark residual codecs")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--codecs", type=str, default="zstd,lzma")
    parser.add_argument("--level", type=int, default=3)
    args = parser.parse_args()

    codecs = [token.strip() for token in args.codecs.split(",") if token.strip()]
    rows = benchmark_residual_codecs(args.residual_manifest, args.out_csv, codecs=codecs, level=args.level)
    print(f"Wrote {len(rows)} codec records to {args.out_csv}")


if __name__ == "__main__":
    main()
