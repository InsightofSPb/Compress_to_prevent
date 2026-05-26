#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.baselines import compute_baseline_tile_scores
from compression.io import write_csv_rows
from compression.metrics import evaluate_change_metrics
from compression.tiles import eval_change_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full facade evaluation suite")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--baseline-methods", type=str, default="absdiff_l1,absdiff_l2,grayscale_absdiff,ssim_change,dinov2_patch_cosine,lpips_change")
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
    parser.add_argument("--include-proposed", dest="include_proposed", action="store_true", default=True)
    parser.add_argument("--no-include-proposed", dest="include_proposed", action="store_false")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.out_dir / "heatmaps"
    proposed_csv = args.out_dir / "proposed_tiles.csv"
    baseline_csv = args.out_dir / "baseline_tiles.csv"
    all_csv = args.out_dir / "all_tile_scores.csv"
    summary_csv = args.out_dir / "summary_metrics.csv"

    all_rows = []
    if args.include_proposed:
        proposed_rows = eval_change_tiles(args.residual_manifest, proposed_csv, heatmap_dir, tile_size=args.tile_size)
        for r in proposed_rows:
            r["method"] = "proposed_residual"
            if r.get("score_type") == "change_score":
                r["score_type"] = "residual_change_score"
        all_rows.extend(proposed_rows)

    methods = [m.strip() for m in args.baseline_methods.split(",") if m.strip()]
    baseline_rows = compute_baseline_tile_scores(
        residual_manifest_csv=args.residual_manifest,
        out_scores_csv=baseline_csv,
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
    all_rows.extend(baseline_rows)

    fields = ["pair_id", "facade_id", "split", "method", "score_type", "tile_x", "tile_y", "tile_score", "tile_size", "heatmap_pgm"]
    normalized = []
    for r in all_rows:
        rr = {k: r.get(k, "") for k in fields}
        normalized.append(rr)
    write_csv_rows(all_csv, fields, normalized)
    evaluate_change_metrics(all_csv, args.labels_csv, summary_csv)
    print(f"Wrote suite outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
