#!/usr/bin/env python3
"""Build aligned manual-reference change maps for RGB temporal pairs.

This tool is for quantitative evaluation of temporal heatmaps. For each pair
from the aligned RGB compression manifest it loads the direct homography saved
under ref_spx_batch_out/pairs, warps the manual mask of the previous year into
the current image coordinates with nearest-neighbour interpolation, and writes
binary reference maps restricted to the saved valid alignment mask.

Generated references:
  * any_semantic_change: previous aligned semantic label != current label.
  * damage_presence_change: damage/non-damage status changed.
  * damage_type_change: damage status or damage category changed; all non-damage
    classes are collapsed to zero before comparison.

Reference PNG encoding is 0=no change, 1=change, 255=invalid/ignored pixel.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_DAMAGE_LABELS = (1, 2, 3, 4, 5, 6, 7)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned GT change references for temporal RGB pairs.")
    parser.add_argument("--pairs-manifest", type=Path, required=True,
                        help="Aligned pairs_all.csv with prev/curr paths, valid_mask_path and split.")
    parser.add_argument("--masks-dir", type=Path, required=True,
                        help="Directory containing manual semantic masks with stems matching RGB images.")
    parser.add_argument("--ref-spx-out", type=Path, required=True,
                        help="Directory containing ref_spx_batch_out/pairs with JSON homographies.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--damage-labels", type=str, default=",".join(str(v) for v in DEFAULT_DAMAGE_LABELS),
                        help="Comma-separated semantic label ids treated as damage.")
    parser.add_argument("--valid-threshold", type=int, default=0)
    parser.add_argument("--invalid-label", type=int, default=255)
    parser.add_argument("--max-rgb-warp-mae", type=float, default=None,
                        help="Optional error if direct H disagrees with saved src_warp beyond this valid-region MAE.")
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


def parse_damage_labels(value: str) -> Tuple[int, ...]:
    labels = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not labels:
        raise ValueError("damage-labels cannot be empty")
    return labels


def index_homographies(ref_spx_out: Path) -> Dict[Tuple[str, int, int], Tuple[np.ndarray, Path, str]]:
    pair_root = ref_spx_out / "pairs"
    if not pair_root.is_dir():
        raise FileNotFoundError("Missing ref_spx pair root: {}".format(pair_root))
    result: Dict[Tuple[str, int, int], Tuple[np.ndarray, Path, str]] = {}
    for json_path in pair_root.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        facade_id, year_a, year_b, H = data.get("facade_id"), data.get("year_a"), data.get("year_b"), data.get("H")
        if facade_id is None or year_a is None or year_b is None or H is None:
            continue
        matrix = np.asarray(H, dtype=np.float64)
        if matrix.shape != (3, 3):
            continue
        key = (str(facade_id), int(year_a), int(year_b))
        quality = str(data.get("status_quality") or data.get("quality") or "")
        if key in result:
            raise ValueError("Duplicate homography for pair {} in {} and {}".format(key, result[key][1], json_path))
        result[key] = (matrix, json_path, quality)
    if not result:
        raise ValueError("No JSON records containing direct H were found under: {}".format(pair_root))
    return result


def find_mask(masks_dir: Path, image_path: Path) -> Path:
    direct = masks_dir / (image_path.stem + ".png")
    if direct.is_file():
        return direct
    matches = [path for path in masks_dir.glob(image_path.stem + ".*") if path.suffix.lower() in IMAGE_EXTS]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("Exactly one manual mask required for image stem {}: {}".format(image_path.stem, matches))


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError("Could not read mask: {}".format(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Could not read RGB image: {}".format(path))
    return image


def write_reference(path: Path, values: np.ndarray, valid: np.ndarray, invalid_label: int) -> None:
    output = np.full(values.shape, invalid_label, dtype=np.uint8)
    output[valid] = values[valid].astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), output):
        raise OSError("Failed to save reference map: {}".format(path))


def compute_rgb_warp_mae(prev_image: Path, saved_warp: Path, H: np.ndarray,
                         out_size: Tuple[int, int], valid: np.ndarray) -> float:
    prev = read_rgb(prev_image)
    saved = read_rgb(saved_warp)
    width, height = out_size
    warped = cv2.warpPerspective(prev, H, (width, height), flags=cv2.INTER_LINEAR)
    if saved.shape != warped.shape:
        raise ValueError("Saved warp/recreated warp shape mismatch: {} vs {}".format(saved.shape, warped.shape))
    if not bool(valid.any()):
        return float("nan")
    return float(np.abs(saved.astype(np.float32) - warped.astype(np.float32))[valid].mean())


def main() -> None:
    args = parse_args()
    damage_labels = parse_damage_labels(args.damage_labels)
    if not 0 <= args.invalid_label <= 255 or not 0 <= args.valid_threshold <= 255:
        raise ValueError("invalid-label and valid-threshold must be in [0, 255]")
    pairs = read_csv(args.pairs_manifest)
    H_index = index_homographies(args.ref_spx_out)
    if not pairs:
        raise ValueError("Empty aligned pair manifest: {}".format(args.pairs_manifest))

    rows: List[Dict[str, object]] = []
    missing_H: List[str] = []
    mae_values: List[float] = []
    for row in pairs:
        pair_id = row["pair_id"]
        facade_id = row["facade_id"]
        year_prev, year_curr = int(row["year_prev"]), int(row["year_curr"])
        key = (facade_id, year_prev, year_curr)
        if key not in H_index:
            missing_H.append(pair_id)
            continue
        H, h_json_path, quality = H_index[key]
        prev_image = Path(row["prev_image_path"])
        curr_image = Path(row["curr_image_path"])
        saved_warp = Path(row["prev_aligned_path"])
        valid_path = Path(row["valid_mask_path"])
        prev_mask_path = find_mask(args.masks_dir, prev_image)
        curr_mask_path = find_mask(args.masks_dir, curr_image)
        prev_mask = read_mask(prev_mask_path)
        curr_mask = read_mask(curr_mask_path)
        valid_mask = read_mask(valid_path) > args.valid_threshold
        height, width = curr_mask.shape[:2]
        if valid_mask.shape != curr_mask.shape:
            raise ValueError("Valid/current mask shape mismatch for {}".format(pair_id))

        prev_aligned = cv2.warpPerspective(
            prev_mask, H, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=args.invalid_label,
        )
        valid = valid_mask & (prev_aligned != args.invalid_label) & (curr_mask != args.invalid_label)
        if not bool(valid.any()):
            raise ValueError("No valid GT pixels after mask alignment for pair: {}".format(pair_id))

        any_semantic_change = prev_aligned != curr_mask
        prev_damage = np.isin(prev_aligned, damage_labels)
        curr_damage = np.isin(curr_mask, damage_labels)
        damage_presence_change = prev_damage != curr_damage
        prev_damage_type = np.where(prev_damage, prev_aligned, 0)
        curr_damage_type = np.where(curr_damage, curr_mask, 0)
        damage_type_change = prev_damage_type != curr_damage_type

        pair_dir = args.out_dir / str(row.get("split", "unsplit") or "unsplit") / facade_id / pair_id
        prev_aligned_path = pair_dir / "prev_manual_mask_aligned.png"
        semantic_path = pair_dir / "any_semantic_change.png"
        presence_path = pair_dir / "damage_presence_change.png"
        damage_type_path = pair_dir / "damage_type_change.png"
        pair_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(prev_aligned_path), prev_aligned)
        write_reference(semantic_path, any_semantic_change, valid, args.invalid_label)
        write_reference(presence_path, damage_presence_change, valid, args.invalid_label)
        write_reference(damage_type_path, damage_type_change, valid, args.invalid_label)

        rgb_warp_mae = compute_rgb_warp_mae(prev_image, saved_warp, H, (width, height), valid)
        mae_values.append(rgb_warp_mae)
        if args.max_rgb_warp_mae is not None and rgb_warp_mae > args.max_rgb_warp_mae:
            raise ValueError("RGB H validation failed for {}: MAE {:.6f} > {:.6f}".format(
                pair_id, rgb_warp_mae, args.max_rgb_warp_mae
            ))
        n_valid = int(valid.sum())
        rows.append({
            "pair_id": pair_id,
            "facade_id": facade_id,
            "year_prev": year_prev,
            "year_curr": year_curr,
            "split": row.get("split", ""),
            "prev_manual_mask_path": str(prev_mask_path),
            "curr_manual_mask_path": str(curr_mask_path),
            "prev_manual_mask_aligned_path": str(prev_aligned_path),
            "valid_mask_path": str(valid_path),
            "any_semantic_change_path": str(semantic_path),
            "damage_presence_change_path": str(presence_path),
            "damage_type_change_path": str(damage_type_path),
            "valid_pixel_count": n_valid,
            "valid_ratio": "{:.8f}".format(n_valid / max(height * width, 1)),
            "any_semantic_change_ratio": "{:.8f}".format(float(any_semantic_change[valid].mean())),
            "damage_presence_change_ratio": "{:.8f}".format(float(damage_presence_change[valid].mean())),
            "damage_type_change_ratio": "{:.8f}".format(float(damage_type_change[valid].mean())),
            "homography_json_path": str(h_json_path),
            "alignment_quality": quality,
            "rgb_warp_recreation_mae": "{:.8f}".format(rgb_warp_mae),
        })

    fields = [
        "pair_id", "facade_id", "year_prev", "year_curr", "split",
        "prev_manual_mask_path", "curr_manual_mask_path", "prev_manual_mask_aligned_path",
        "valid_mask_path", "any_semantic_change_path", "damage_presence_change_path",
        "damage_type_change_path", "valid_pixel_count", "valid_ratio",
        "any_semantic_change_ratio", "damage_presence_change_ratio", "damage_type_change_ratio",
        "homography_json_path", "alignment_quality", "rgb_warp_recreation_mae",
    ]
    write_csv(args.out_dir / "pair_change_references.csv", fields, rows)
    by_split = {}
    for split in ("train", "val", "test"):
        split_rows = [item for item in rows if item.get("split") == split]
        write_csv(args.out_dir / "pair_change_references_{}.csv".format(split), fields, split_rows)
        by_split[split] = len(split_rows)
    report = {
        "pairs_manifest": str(args.pairs_manifest),
        "ref_spx_out": str(args.ref_spx_out),
        "masks_dir": str(args.masks_dir),
        "damage_labels": list(damage_labels),
        "n_input_pairs": len(pairs),
        "n_reference_pairs": len(rows),
        "missing_direct_homography_pairs": missing_H,
        "pairs_by_split": by_split,
        "mean_rgb_warp_recreation_mae": float(np.nanmean(mae_values)) if mae_values else None,
        "max_rgb_warp_recreation_mae": float(np.nanmax(mae_values)) if mae_values else None,
        "reference_encoding": {"no_change": 0, "change": 1, "invalid_or_ignored": args.invalid_label},
        "primary_reference_candidate": "damage_type_change",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pair_change_reference_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Built aligned GT reference pairs: {} / {}".format(len(rows), len(pairs)))
    print("Pairs by split: {}".format(by_split))
    print("Pairs missing direct H: {}".format(len(missing_H)))
    if mae_values:
        print("RGB warp recreation MAE mean/max: {:.6f}/{:.6f}".format(
            float(np.nanmean(mae_values)), float(np.nanmax(mae_values))
        ))
    print("Manifest: {}".format(args.out_dir / "pair_change_references.csv"))


if __name__ == "__main__":
    main()
