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
    parser.add_argument("--methods", type=str, default="zstd,lzma,webp,fnlic")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--strict", action="store_true", help="Fail if a method is unavailable")
    args = parser.parse_args()

    methods = [token.strip() for token in args.methods.split(",") if token.strip()]
    rows = benchmark_residual_codecs(
        args.residual_manifest,
        args.out_csv,
        methods=methods,
        level=args.level,
        strict=args.strict,
    )
    print(f"Wrote {len(rows)} codec records to {args.out_csv}")


if __name__ == "__main__":
    main()
