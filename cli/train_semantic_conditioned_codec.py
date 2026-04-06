#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.conditioned.train import train_semantic_conditioned_codec


def main() -> None:
    parser = argparse.ArgumentParser(description="Train C1 semantic-conditioned residual codec")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--artifact-index-csv", type=Path, required=True)
    parser.add_argument("--temporal-features-csv", type=Path, default=None)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--context-mode", type=str, choices=["none", "lposs_only", "features_only", "temporal_semantic_only", "full", "custom"], default="full")
    parser.add_argument("--custom-context-sources", type=str, default="")
    parser.add_argument("--conditioning-mechanism", type=str, choices=["concat_context", "film_context"], default="concat_context")
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--max-symbols-per-tile", type=int, default=512)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    args = parser.parse_args()

    summary = train_semantic_conditioned_codec(
        residual_manifest_csv=args.residual_manifest,
        pairs_csv=args.pairs_csv,
        artifact_index_csv=args.artifact_index_csv,
        temporal_features_csv=args.temporal_features_csv,
        model_out=args.model_out,
        tile_size=args.tile_size,
        train_split=args.train_split,
        context_mode=args.context_mode,
        context_dim=args.context_dim,
        conditioning_mechanism=args.conditioning_mechanism,
        custom_sources=[s.strip() for s in args.custom_context_sources.split(",") if s.strip()],
        max_symbols_per_tile=args.max_symbols_per_tile,
        ridge_lambda=args.ridge_lambda,
    )
    print(f"Training summary: {summary}")


if __name__ == "__main__":
    main()
