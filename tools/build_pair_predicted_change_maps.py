#!/usr/bin/env python3
"""Build aligned semantic-change maps from stitched predicted masks.

This companion to ``build_pair_change_reference_maps.py`` constructs the same
semantic change definitions, but uses full-resolution predictions exported by
``export_stitched_segmentation_predictions.py`` rather than manual masks. The
result is intended for whole-pipeline qualitative figures and optional
predicted-semantic-signal analysis. It must not be confused with a ground-truth
reference used to validate the temporal heatmap.

Output PNG encoding is compatible with the manual-reference builder:
``0=no semantic change``, ``1=semantic change``, ``255=invalid/ignored``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_pair_change_reference_maps import (  # noqa: E402
    DEFAULT_CONTENT_LABELS,
    DEFAULT_DAMAGE_LABELS,
    DEFAULT_REPAIR_LABELS,
    index_homographies,
    parse_labels,
    read_mask,
    retained_type_change,
    write_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned predicted semantic-change maps for RGB temporal pairs.")
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, action="append", required=True,
                        help="Manifest produced by export_stitched_segmentation_predictions.py; repeat for val/test.")
    parser.add_argument("--ref-spx-out", type=Path, required=True,
                        help="Directory containing ref_spx_batch_out/pairs with direct homographies.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", default="val,test",
                        help="Comma-separated pair splits to include; default: val,test.")
    parser.add_argument("--damage-labels", default=",".join(str(v) for v in DEFAULT_DAMAGE_LABELS))
    parser.add_argument("--repair-labels", default=",".join(str(v) for v in DEFAULT_REPAIR_LABELS))
    parser.add_argument("--content-labels", default=",".join(str(v) for v in DEFAULT_CONTENT_LABELS))
    parser.add_argument("--valid-threshold", type=int, default=0)
    parser.add_argument("--invalid-label", type=int, default=255)
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


def load_prediction_index(paths: Sequence[Path]) -> Tuple[Dict[str, Path], Dict[str, str]]:
    index: Dict[str, Path] = {}
    models: Dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("Prediction manifest not found: {}".format(path))
        for row in read_csv(path):
            stem = row.get("image_stem", "") or Path(row["source_image"]).stem
            prediction_path = Path(row["prediction_path"])
            if stem in index and index[stem] != prediction_path:
                raise ValueError("Duplicate prediction entries for image stem {}".format(stem))
            if not prediction_path.is_file():
                raise FileNotFoundError("Predicted mask not found: {}".format(prediction_path))
            index[stem] = prediction_path
            models[stem] = row.get("model_label", "")
    return index, models


def main() -> None:
    args = parse_args()
    requested_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    if not requested_splits:
        raise ValueError("At least one split must be requested")
    damage_labels = parse_labels(args.damage_labels, "damage-labels")
    repair_labels = parse_labels(args.repair_labels, "repair-labels", allow_empty=True)
    content_labels = parse_labels(args.content_labels, "content-labels", allow_empty=True)
    inspection_labels = tuple(sorted(set(damage_labels + repair_labels + content_labels)))
    damage_or_repair_labels = tuple(sorted(set(damage_labels + repair_labels)))
    intervention_or_content_labels = tuple(sorted(set(repair_labels + content_labels)))

    pairs = [row for row in read_csv(args.pairs_manifest) if row.get("split", "") in requested_splits]
    predictions, models = load_prediction_index(args.prediction_manifest)
    homographies = index_homographies(args.ref_spx_out)
    output_rows: List[Dict[str, object]] = []
    missing_predictions: List[str] = []

    for row in pairs:
        pair_id = row["pair_id"]
        facade_id = row["facade_id"]
        year_prev, year_curr = int(row["year_prev"]), int(row["year_curr"])
        prev_stem = Path(row["prev_image_path"]).stem
        curr_stem = Path(row["curr_image_path"]).stem
        if prev_stem not in predictions or curr_stem not in predictions:
            missing_predictions.append(pair_id)
            continue
        key = (facade_id, year_prev, year_curr)
        if key not in homographies:
            raise KeyError("No direct homography for pair: {}".format(pair_id))
        H, _, quality = homographies[key]
        previous = read_mask(predictions[prev_stem])
        current = read_mask(predictions[curr_stem])
        valid = read_mask(Path(row["valid_mask_path"])) > args.valid_threshold
        height, width = current.shape[:2]
        previous_aligned = cv2.warpPerspective(
            previous, H, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=args.invalid_label,
        )
        if valid.shape != current.shape:
            raise ValueError("Valid/current predicted mask shape mismatch for {}".format(pair_id))
        valid = valid & (previous_aligned != args.invalid_label) & (current != args.invalid_label)
        if not bool(valid.any()):
            raise ValueError("No valid predicted semantic pixels for pair: {}".format(pair_id))

        changes = {
            "any_semantic_change": previous_aligned != current,
            "damage_presence_change": np.isin(previous_aligned, damage_labels) != np.isin(current, damage_labels),
            "damage_type_change": retained_type_change(previous_aligned, current, damage_labels),
            "damage_or_repair_change": retained_type_change(previous_aligned, current, damage_or_repair_labels),
            "intervention_or_content_change": retained_type_change(previous_aligned, current, intervention_or_content_labels),
            "inspection_relevant_change": retained_type_change(previous_aligned, current, inspection_labels),
        }
        pair_dir = args.out_dir / row.get("split", "unsplit") / facade_id / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        prev_aligned_path = pair_dir / "prev_predicted_mask_aligned.png"
        curr_path = pair_dir / "curr_predicted_mask.png"
        cv2.imwrite(str(prev_aligned_path), previous_aligned)
        cv2.imwrite(str(curr_path), current)
        change_paths: Dict[str, str] = {}
        for name, change in changes.items():
            path = pair_dir / (name + ".png")
            write_reference(path, change, valid, args.invalid_label)
            change_paths[name + "_path"] = str(path.resolve())
        n_valid = int(valid.sum())
        output_rows.append({
            "pair_id": pair_id,
            "facade_id": facade_id,
            "year_prev": year_prev,
            "year_curr": year_curr,
            "split": row.get("split", ""),
            "mask_source": "predicted",
            "prediction_model_label": models.get(curr_stem, ""),
            "prev_predicted_mask_path": str(predictions[prev_stem].resolve()),
            "curr_predicted_mask_path": str(predictions[curr_stem].resolve()),
            "prev_predicted_mask_aligned_path": str(prev_aligned_path.resolve()),
            "curr_predicted_mask_in_current_coordinates_path": str(curr_path.resolve()),
            "valid_mask_path": row["valid_mask_path"],
            "homography_quality": quality,
            "valid_pixel_count": n_valid,
            "valid_ratio": "{:.8f}".format(n_valid / max(height * width, 1)),
            **change_paths,
            **{name + "_ratio": "{:.8f}".format(float(change[valid].mean())) for name, change in changes.items()},
        })

    if missing_predictions:
        raise FileNotFoundError(
            "Predicted masks are missing for {} selected temporal pairs, first entries: {}".format(
                len(missing_predictions), missing_predictions[:10]
            )
        )
    if not output_rows:
        raise ValueError("No predicted semantic temporal pairs were generated")
    manifest_path = args.out_dir / "pair_predicted_change_maps.csv"
    fields = list(output_rows[0].keys())
    write_csv(manifest_path, fields, output_rows)
    report = {
        "source_pairs_manifest": str(args.pairs_manifest),
        "prediction_manifests": [str(path) for path in args.prediction_manifest],
        "ref_spx_out": str(args.ref_spx_out),
        "out_dir": str(args.out_dir),
        "mask_source": "stitched_finetuned_semantic_predictions",
        "requested_splits": sorted(requested_splits),
        "n_pairs": len(output_rows),
        "pairs_by_split": {split: sum(1 for row in output_rows if row["split"] == split) for split in sorted(requested_splits)},
        "reference_warning": "These maps are model-derived semantic change visualisations, not ground-truth targets for quantitative validation.",
        "label_groups": {
            "damage": list(damage_labels),
            "repairs": list(repair_labels),
            "visual_content_or_signage": list(content_labels),
            "inspection_relevant": list(inspection_labels),
            "damage_or_repair": list(damage_or_repair_labels),
        },
        "manifest": str(manifest_path),
    }
    (args.out_dir / "predicted_change_map_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Built predicted semantic change maps:", len(output_rows))
    print("Pairs by split:", report["pairs_by_split"])
    print("Manifest:", manifest_path)
    print("WARNING: predicted maps are visual/whole-pipeline signals, not GT references for validation metrics.")


if __name__ == "__main__":
    main()
