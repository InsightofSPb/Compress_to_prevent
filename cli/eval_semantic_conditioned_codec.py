#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.conditioned.eval import eval_semantic_conditioned_codec


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate C1 semantic-conditioned residual codec")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--artifact-index-csv", type=Path, required=True)
    parser.add_argument("--temporal-features-csv", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--context-mode", type=str, choices=["none", "lposs_only", "features_only", "temporal_semantic_only", "full", "custom"], default="full")
    parser.add_argument("--custom-context-sources", type=str, default="")
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--conditioning-mechanism", type=str, choices=["concat_context", "film_context"], default="concat_context")
    parser.add_argument("--out-tile-csv", type=Path, required=True)
    parser.add_argument("--out-pair-csv", type=Path, required=True)
    args = parser.parse_args()

    summary = eval_semantic_conditioned_codec(
        residual_manifest_csv=args.residual_manifest,
        pairs_csv=args.pairs_csv,
        artifact_index_csv=args.artifact_index_csv,
        temporal_features_csv=args.temporal_features_csv,
        model_path=args.model_path,
        split=args.split,
        tile_size=args.tile_size,
        context_mode=args.context_mode,
        context_dim=args.context_dim,
        conditioning_mechanism=args.conditioning_mechanism,
        out_tile_csv=args.out_tile_csv,
        out_pair_csv=args.out_pair_csv,
        custom_sources=[s.strip() for s in args.custom_context_sources.split(",") if s.strip()],
    )
    print(f"Evaluation summary: {summary}")


if __name__ == "__main__":
    main()
