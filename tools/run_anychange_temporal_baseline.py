#!/usr/bin/env python3
"""Run AnyChange (SAM-based zero-shot change detection) on temporal RGB pairs.

The script adapts the official ``torchange`` AnyChange implementation to the
same tile-score CSV protocol used by the facade temporal heatmap experiments.
It can therefore be evaluated with ``tools/evaluate_temporal_tile_scores.py``
and included in existing baseline/robustness/bootstrap tables.

Protocol
--------
* Each pair is read from a pair manifest with ``prev_aligned_path``,
  ``curr_image_path`` and ``valid_mask_path`` columns.
* Invalid alignment pixels are neutralised before inference by copying the
  aligned previous RGB values into the current RGB frame.
* AnyChange mask proposals are rasterised into a continuous confidence map by
  taking the maximum proposal ``change_confidence`` at each pixel.
* Each output tile score is the mean continuous confidence over valid pixels.
* For ranking metrics, ``--change-confidence-threshold 180`` retains almost all
  proposals and avoids pre-thresholding the heatmap before AUPRC/AUROC. The
  official/demo value of 145 can be evaluated separately as a thresholded
  detector variant.

AnyChange is an external dependency from the official ``torchange`` package;
its source code and SAM weights are not vendored in this repository.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


FIELDS = [
    "pair_id", "facade_id", "split", "method", "score_type", "tile_x", "tile_y",
    "tile_score", "tile_size", "valid_pixel_count", "valid_ratio",
    "n_change_proposals", "pair_mean_valid_score", "pair_max_valid_score",
]
PAIR_FIELDS = [
    "pair_id", "facade_id", "split", "n_change_proposals", "n_retained_tiles",
    "pair_mean_valid_score", "pair_max_valid_score", "valid_ratio", "elapsed_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AnyChange and export tile-level temporal change scores.")
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--splits", default="val,test",
                        help="Comma-separated manifest splits to process; default: val,test.")
    parser.add_argument("--model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_h")
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--method-name", default="anychange_sam_vith_continuous")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--min-valid-ratio", type=float, default=0.50)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--change-confidence-threshold", type=float, default=180.0,
                        help="Use 180 for continuous ranking; AnyChange demo uses 145 for binary-like detections.")
    parser.add_argument("--area-thresh", type=float, default=0.8)
    parser.add_argument("--match-hist", action="store_true",
                        help="Enable AnyChange histogram matching as a separate optional protocol.")
    parser.add_argument("--no-bitemporal-match", action="store_true")
    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--heatmap-dir", type=Path, default=None)
    parser.add_argument("--save-every-pairs", type=int, default=1,
                        help="Write partial output CSV every N completed pairs; 0 writes only at the end.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Debug-only pair limit; do not use for final reported metrics.")
    parser.add_argument("--invalid-label-threshold", type=int, default=0,
                        help="Pixels with valid-mask value greater than this value are treated as valid.")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{str(k): (v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Cannot read RGB image: {}".format(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_valid_mask(path: Optional[Path], shape: Tuple[int, int], threshold: int) -> np.ndarray:
    if path is None:
        return np.ones(shape, dtype=bool)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError("Cannot read valid mask: {}".format(path))
    if mask.shape != shape:
        raise ValueError("Valid-mask size mismatch: expected {}, got {} for {}".format(shape, mask.shape, path))
    return mask > threshold


def parse_splits(value: str) -> Set[str]:
    splits = {item.strip() for item in value.split(",") if item.strip()}
    if not splits:
        raise ValueError("At least one split must be requested")
    return splits


def tile_grid(height: int, width: int, tile_size: int) -> Iterable[Tuple[int, int, slice, slice]]:
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            yield x0 // tile_size, y0 // tile_size, slice(y0, min(y0 + tile_size, height)), slice(x0, min(x0 + tile_size, width))


def rasterise_change_map(change_masks, height: int, width: int, rle_to_mask) -> Tuple[np.ndarray, int]:
    """Rasterise proposal confidence with max-overlapping-mask aggregation."""
    confidence_map = np.full((height, width), -1.0, dtype=np.float32)
    rles = change_masks["rles"] if "rles" in change_masks else []
    confidences = change_masks["change_confidence"] if "change_confidence" in change_masks else []
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.detach().cpu().numpy()
    for rle, confidence in zip(rles, confidences):
        mask = rle_to_mask(rle).astype(bool)
        confidence_map[mask] = np.maximum(confidence_map[mask], float(confidence))
    return confidence_map, len(rles)


def normalise_for_display(score_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = score_map[valid]
    normalised = np.zeros_like(score_map, dtype=np.float32)
    if values.size == 0:
        return normalised
    low, high = np.percentile(values, [5, 95])
    if float(high) <= float(low):
        return normalised
    normalised = np.clip((score_map - float(low)) / float(high - low), 0.0, 1.0)
    normalised[~valid] = 0.0
    return normalised


def save_visualisation(out_dir: Path, pair_id: str, current_rgb: np.ndarray, score_map: np.ndarray, valid: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    display = normalise_for_display(score_map, valid)
    heatmap_bgr = cv2.applyColorMap((display * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    current_bgr = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(current_bgr, 0.55, heatmap_bgr, 0.45, 0.0)
    cv2.imwrite(str(out_dir / (pair_id + "_anychange_heatmap.png")), heatmap_bgr)
    cv2.imwrite(str(out_dir / (pair_id + "_anychange_overlay.png")), overlay)


def main() -> None:
    args = parse_args()
    if not args.sam_checkpoint.is_file():
        raise FileNotFoundError("SAM checkpoint not found: {}".format(args.sam_checkpoint))
    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")
    if not 0.0 <= args.min_valid_ratio <= 1.0:
        raise ValueError("min-valid-ratio must be in [0, 1]")
    if not 0.0 <= args.change_confidence_threshold <= 180.0:
        raise ValueError("change-confidence-threshold must be in [0, 180]")
    if args.save_every_pairs < 0:
        raise ValueError("save-every-pairs must be non-negative")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("max-pairs must be positive")

    try:
        from torchange.models.segment_any_change import AnyChange
        from torchange.models.segment_any_change.segment_anything.utils.amg import rle_to_mask
    except Exception as exc:
        raise RuntimeError(
            "AnyChange is unavailable. Install the official implementation with: "
            "pip install -U --no-deps --force-reinstall git+https://github.com/Z-Zheng/pytorch-change-models"
        ) from exc

    requested_splits = parse_splits(args.splits)
    manifest_rows = [row for row in read_csv(args.pairs_manifest) if row.get("split", "") in requested_splits]
    if args.max_pairs is not None:
        manifest_rows = manifest_rows[:args.max_pairs]
        print("WARNING: --max-pairs active; this is a debug run and must not be reported as final evaluation.", flush=True)
    if not manifest_rows:
        raise ValueError("No pair rows selected for requested splits: {}".format(sorted(requested_splits)))

    print("Loading AnyChange {} from {}...".format(args.model_type, args.sam_checkpoint), flush=True)
    model = AnyChange(args.model_type, sam_checkpoint=str(args.sam_checkpoint))
    model.make_mask_generator(
        points_per_side=args.points_per_side,
        stability_score_thresh=args.stability_score_thresh,
    )
    model.set_hyperparameters(
        change_confidence_threshold=args.change_confidence_threshold,
        use_normalized_feature=True,
        area_thresh=args.area_thresh,
        match_hist=args.match_hist,
        bitemporal_match=not args.no_bitemporal_match,
    )

    heatmap_dir = args.heatmap_dir or (args.out_csv.parent / (args.out_csv.stem + "_visualisations"))
    score_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []
    iterator = tqdm(manifest_rows, desc="AnyChange temporal pairs", unit="pair", disable=args.no_progress)

    for index, row in enumerate(iterator, start=1):
        started = time.time()
        pair_id = row["pair_id"]
        facade_id = row.get("facade_id", "")
        split = row.get("split", "")
        previous = load_rgb(Path(row["prev_aligned_path"]))
        current = load_rgb(Path(row["curr_image_path"]))
        if previous.shape != current.shape:
            raise ValueError("RGB shape mismatch for pair {}: prev={}, curr={}".format(pair_id, previous.shape, current.shape))
        height, width = current.shape[:2]
        valid_path = Path(row["valid_mask_path"]) if row.get("valid_mask_path", "") else None
        valid = load_valid_mask(valid_path, (height, width), args.invalid_label_threshold)
        valid_ratio = float(valid.mean())
        current_neutral = current.copy()
        current_neutral[~valid] = previous[~valid]

        with torch.inference_mode():
            change_masks, _, _ = model.forward(previous, current_neutral)
        confidence_map, n_proposals = rasterise_change_map(change_masks, height, width, rle_to_mask)
        if int(valid.sum()):
            pair_mean = float(confidence_map[valid].mean())
            pair_max = float(confidence_map[valid].max())
        else:
            pair_mean, pair_max = -1.0, -1.0

        retained_tiles = 0
        for tile_x, tile_y, ys, xs in tile_grid(height, width, args.tile_size):
            valid_tile = valid[ys, xs]
            valid_count = int(valid_tile.sum())
            tile_valid_ratio = float(valid_count / max(valid_tile.size, 1))
            if valid_count == 0 or tile_valid_ratio < args.min_valid_ratio:
                continue
            tile_score = float(confidence_map[ys, xs][valid_tile].mean())
            score_rows.append({
                "pair_id": pair_id,
                "facade_id": facade_id,
                "split": split,
                "method": args.method_name,
                "score_type": "anychange_continuous_mask_confidence",
                "tile_x": tile_x,
                "tile_y": tile_y,
                "tile_score": tile_score,
                "tile_size": args.tile_size,
                "valid_pixel_count": valid_count,
                "valid_ratio": tile_valid_ratio,
                "n_change_proposals": n_proposals,
                "pair_mean_valid_score": pair_mean,
                "pair_max_valid_score": pair_max,
            })
            retained_tiles += 1

        if args.save_heatmaps:
            save_visualisation(heatmap_dir, pair_id, current, confidence_map, valid)
        pair_rows.append({
            "pair_id": pair_id,
            "facade_id": facade_id,
            "split": split,
            "n_change_proposals": n_proposals,
            "n_retained_tiles": retained_tiles,
            "pair_mean_valid_score": pair_mean,
            "pair_max_valid_score": pair_max,
            "valid_ratio": valid_ratio,
            "elapsed_seconds": time.time() - started,
        })
        model.clear_cached_embedding()
        if args.save_every_pairs and index % args.save_every_pairs == 0:
            write_csv(args.out_csv, FIELDS, score_rows)
            write_csv(args.out_csv.with_suffix(".pairs.csv"), PAIR_FIELDS, pair_rows)
        iterator.set_postfix(split=split, proposals=n_proposals, tiles=len(score_rows))

    write_csv(args.out_csv, FIELDS, score_rows)
    pair_summary_path = args.out_csv.with_suffix(".pairs.csv")
    write_csv(pair_summary_path, PAIR_FIELDS, pair_rows)
    report = {
        "pairs_manifest": str(args.pairs_manifest),
        "out_csv": str(args.out_csv),
        "splits": sorted(requested_splits),
        "model_type": args.model_type,
        "sam_checkpoint": str(args.sam_checkpoint),
        "method_name": args.method_name,
        "tile_size": args.tile_size,
        "min_valid_ratio": args.min_valid_ratio,
        "points_per_side": args.points_per_side,
        "stability_score_thresh": args.stability_score_thresh,
        "change_confidence_threshold": args.change_confidence_threshold,
        "area_thresh": args.area_thresh,
        "match_hist": args.match_hist,
        "bitemporal_match": not args.no_bitemporal_match,
        "n_pairs": len(pair_rows),
        "n_tile_scores": len(score_rows),
        "tiles_by_split": {split: sum(1 for item in score_rows if item["split"] == split) for split in sorted(requested_splits)},
        "pair_summary_csv": str(pair_summary_path),
        "validity_policy": "Invalid aligned pixels are neutralised before AnyChange inference and excluded from tile averaging.",
        "continuous_score_policy": "Per-pixel map is max AnyChange change_confidence across overlapping proposals; uncovered pixels receive -1.0.",
        "ranking_note": "For AUPRC/AUROC, change-confidence-threshold=180 is used to retain proposal confidence ranking before metric computation.",
        "debug_max_pairs": args.max_pairs,
    }
    report_path = args.out_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Completed AnyChange temporal baseline")
    print("Pairs:", len(pair_rows))
    print("Tile scores:", len(score_rows))
    print("Scores:", args.out_csv)
    print("Pair summary:", pair_summary_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
