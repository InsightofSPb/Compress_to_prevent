#!/usr/bin/env python3
"""Compute tile-level temporal change baseline scores on aligned RGB pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression.baselines import compute_baseline_tile_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute valid-region tile-level temporal change baselines.")
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="absdiff_l1,ssim_change",
                        help="Comma-separated: absdiff_l1,absdiff_l2,grayscale_absdiff,ssim_change,lpips_change,dinov2_patch_cosine")
    parser.add_argument("--splits", type=str, default="",
                        help="Optional comma-separated split restriction, e.g. val,test. Empty evaluates all manifest rows.")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--min-valid-ratio", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--dinov2-model-name", default="dinov2_vitb14")
    parser.add_argument("--dinov2-cache-dir", type=Path, default=None)
    parser.add_argument("--dinov2-weights-path", type=Path, default=None)
    parser.add_argument("--dinov2-repo-dir", type=Path, default=None)
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--deep-batch-size", type=int, default=128,
                        help="Batch size for tile-level deep baselines such as LPIPS.")
    parser.add_argument("--skip-deep-baselines", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")
    if args.deep_batch_size <= 0:
        raise ValueError("deep-batch-size must be positive")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    splits = [item.strip() for item in args.splits.split(",") if item.strip()] or None
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = compute_baseline_tile_scores(
        residual_manifest_csv=args.residual_manifest,
        out_scores_csv=args.out_csv,
        methods=methods,
        tile_size=args.tile_size,
        min_valid_ratio=args.min_valid_ratio,
        device=args.device,
        feature_cache_dir=args.feature_cache_dir,
        dinov2_model_name=args.dinov2_model_name,
        dinov2_cache_dir=args.dinov2_cache_dir,
        dinov2_weights_path=args.dinov2_weights_path,
        dinov2_repo_dir=args.dinov2_repo_dir,
        lpips_net=args.lpips_net,
        skip_deep_baselines=args.skip_deep_baselines,
        include_splits=splits,
        deep_batch_size=args.deep_batch_size,
        show_progress=not args.no_progress,
    )
    method_counts = {}
    split_counts = {}
    for row in rows:
        method = str(row["method"])
        split = str(row["split"])
        method_counts[method] = method_counts.get(method, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1
    report = {
        "residual_manifest": str(args.residual_manifest),
        "out_csv": str(args.out_csv),
        "methods_requested": methods,
        "splits_requested": splits,
        "tile_size": args.tile_size,
        "min_valid_ratio": args.min_valid_ratio,
        "device": args.device,
        "deep_batch_size": args.deep_batch_size,
        "n_score_rows": len(rows),
        "score_rows_by_method": method_counts,
        "score_rows_by_split_all_methods": split_counts,
        "valid_region_policy": "invalid aligned pixels excluded; low-coverage tiles dropped",
        "lpips_edge_tile_policy": "right/bottom partial tiles are edge-padded to fixed tile size before LPIPS inference",
    }
    report_path = args.out_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Built tile baseline score rows: {}".format(len(rows)))
    print("Rows by method: {}".format(method_counts))
    print("Rows by split across methods: {}".format(split_counts))
    print("Scores: {}".format(args.out_csv))
    print("Report: {}".format(report_path))


if __name__ == "__main__":
    main()
