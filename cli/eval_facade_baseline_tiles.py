#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.baselines import compute_baseline_tile_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate facade baseline change tiles")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="absdiff_l1,absdiff_l2,grayscale_absdiff,ssim_change,dinov2_patch_cosine,lpips_change")
    parser.add_argument("--out-scores-csv", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--dinov2-model-name", type=str, default="dinov2_vitb14")
    parser.add_argument("--dinov2-cache-dir", type=Path)
    parser.add_argument("--dinov2-weights-path", type=Path)
    parser.add_argument("--dinov2-repo-dir", type=Path)
    parser.add_argument("--lpips-net", type=str, default="alex")
    parser.add_argument("--temporal-features-csv", type=Path)
    parser.add_argument("--artifact-index-csv", type=Path)
    parser.add_argument("--skip-deep-baselines", action="store_true")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = compute_baseline_tile_scores(
        residual_manifest_csv=args.residual_manifest,
        out_scores_csv=args.out_scores_csv,
        methods=methods,
        tile_size=args.tile_size,
        device=args.device,
        feature_cache_dir=args.feature_cache_dir,
        dinov2_model_name=args.dinov2_model_name,
        dinov2_cache_dir=args.dinov2_cache_dir,
        dinov2_weights_path=args.dinov2_weights_path,
        lpips_net=args.lpips_net,
        dinov2_repo_dir=args.dinov2_repo_dir,
        temporal_features_csv=args.temporal_features_csv,
        artifact_index_csv=args.artifact_index_csv,
        skip_deep_baselines=args.skip_deep_baselines,
    )
    print(f"Wrote {len(rows)} baseline tile rows to {args.out_scores_csv}")


if __name__ == "__main__":
    main()
