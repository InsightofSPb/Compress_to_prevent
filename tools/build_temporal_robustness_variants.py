#!/usr/bin/env python3
"""Build controlled RGB temporal pair variants for robustness and ablation experiments.

Variants:

* ``resize_only``: replaces the geometrically warped previous observation by
  the raw previous RGB image resized to the current frame. The original valid
  support mask is retained so evaluation uses exactly the same supported image
  region as the aligned experiment. This variant is intended for separate
  retraining and measures the contribution of geometric registration.
* ``brightness_contrast``: applies a deterministic photometric perturbation to
  the current RGB image only on requested splits (test by default). Geometry,
  references and valid support remain unchanged. This variant is intended for
  robustness evaluation with the clean-trained compression model.
* ``soft_shadow``: applies a deterministic smooth cast-shadow-like attenuation
  to the current RGB image only on requested splits. It is also evaluated with
  the clean-trained compression model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build controlled temporal RGB robustness/ablation pair manifests.")
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("resize_only", "brightness_contrast", "soft_shadow"), required=True)
    parser.add_argument("--perturb-splits", default="test",
                        help="Comma-separated splits transformed for photometric variants. Ignored by resize_only.")
    parser.add_argument("--contrast-alpha", type=float, default=1.10)
    parser.add_argument("--brightness-beta", type=float, default=12.0)
    parser.add_argument("--shadow-strength", type=float, default=0.35,
                        help="Maximum multiplicative darkening for soft_shadow; must be in (0,1).")
    parser.add_argument("--shadow-softness", type=float, default=0.16,
                        help="Shadow band width relative to image diagonal; must be positive.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{str(key): (value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Could not read image: {}".format(path))
    return image


def save_bgr(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("Could not write image: {}".format(path))


def apply_brightness_contrast(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    transformed = image.astype(np.float32) * alpha + beta
    return np.clip(transformed, 0.0, 255.0).astype(np.uint8)


def apply_soft_shadow(image: np.ndarray, pair_id: str, strength: float, softness: float) -> np.ndarray:
    """Apply a deterministic smooth diagonal shadow stripe to one RGB image."""
    height, width = image.shape[:2]
    digest = hashlib.sha1(pair_id.encode("utf-8")).digest()
    angle = math.radians(20.0 + (digest[0] / 255.0) * 50.0)
    centre = 0.34 + (digest[1] / 255.0) * 0.32
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = xx / max(width - 1, 1)
    yn = yy / max(height - 1, 1)
    projected = xn * math.cos(angle) + yn * math.sin(angle)
    norm = abs(math.cos(angle)) + abs(math.sin(angle))
    projected = projected / max(norm, 1e-8)
    shadow = np.exp(-0.5 * ((projected - centre) / softness) ** 2)
    factor = 1.0 - strength * shadow
    return np.clip(image.astype(np.float32) * factor[..., None], 0.0, 255.0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if args.contrast_alpha <= 0:
        raise ValueError("contrast-alpha must be positive")
    if not 0.0 < args.shadow_strength < 1.0:
        raise ValueError("shadow-strength must be in (0, 1)")
    if args.shadow_softness <= 0:
        raise ValueError("shadow-softness must be positive")
    perturb_splits = {part.strip() for part in args.perturb_splits.split(",") if part.strip()}
    if not perturb_splits.issubset(SPLITS):
        raise ValueError("perturb-splits must contain only train,val,test")

    rows = read_csv(args.pairs_manifest)
    required = {"pair_id", "facade_id", "prev_image_path", "curr_image_path", "prev_aligned_path", "valid_mask_path", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Pair manifest must contain columns: {}".format(sorted(required)))
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError("Output directory is not empty: {}. Pass --overwrite.".format(args.out_dir))
        shutil.rmtree(args.out_dir)
    assets_dir = args.out_dir / "assets"
    output_rows: List[Dict[str, object]] = []
    modified_counts = {split: 0 for split in SPLITS}

    for source in rows:
        row: Dict[str, object] = dict(source)
        pair_id = source["pair_id"]
        split = source["split"]
        row["variant"] = args.variant
        row["variant_active"] = 0
        row["variant_protocol"] = ""

        if args.variant == "resize_only":
            current = read_bgr(Path(source["curr_image_path"]))
            previous_raw = read_bgr(Path(source["prev_image_path"]))
            resized = cv2.resize(previous_raw, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_LINEAR)
            out_path = assets_dir / split / pair_id / "prev_resize_only.png"
            save_bgr(out_path, resized)
            row["prev_aligned_path"] = str(out_path.resolve())
            row["alignment_source"] = "resize_only_no_geometric_registration_same_valid_support"
            row["variant_active"] = 1
            row["variant_protocol"] = "raw previous resized to current frame; original valid support retained"
            modified_counts[split] += 1
        elif split in perturb_splits:
            current = read_bgr(Path(source["curr_image_path"]))
            if args.variant == "brightness_contrast":
                modified = apply_brightness_contrast(current, args.contrast_alpha, args.brightness_beta)
                filename = "curr_brightness_contrast.png"
                protocol = "current RGB only; alpha={:.4f}; beta={:.4f}".format(args.contrast_alpha, args.brightness_beta)
            else:
                modified = apply_soft_shadow(current, pair_id, args.shadow_strength, args.shadow_softness)
                filename = "curr_soft_shadow.png"
                protocol = "current RGB only; deterministic smooth band; max_darkening={:.4f}; softness={:.4f}".format(
                    args.shadow_strength, args.shadow_softness
                )
            out_path = assets_dir / split / pair_id / filename
            save_bgr(out_path, modified)
            row["curr_image_path"] = str(out_path.resolve())
            row["variant_active"] = 1
            row["variant_protocol"] = protocol
            modified_counts[split] += 1
        else:
            row["variant_protocol"] = "clean row retained for threshold selection/training compatibility"
        output_rows.append(row)

    original_fields = list(rows[0].keys())
    fields = original_fields + [field for field in ["variant", "variant_active", "variant_protocol"] if field not in original_fields]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "pairs_all.csv", fields, output_rows)
    for split in SPLITS:
        write_csv(args.out_dir / "pairs_{}.csv".format(split), fields, [row for row in output_rows if row.get("split") == split])

    report = {
        "source_pairs_manifest": str(args.pairs_manifest),
        "variant": args.variant,
        "out_dir": str(args.out_dir),
        "n_pairs": len(output_rows),
        "pairs_by_split": {split: sum(1 for row in output_rows if row.get("split") == split) for split in SPLITS},
        "modified_pairs_by_split": modified_counts,
        "valid_support_policy": "Original alignment-derived valid_mask_path is retained for all variants to keep evaluated support comparable to the aligned pipeline.",
        "reference_policy": "Original aligned GT change references remain unchanged and define target change in current-image coordinates.",
        "training_policy": (
            "Retrain RGB/MSDZip on this variant for resize_only alignment ablation."
            if args.variant == "resize_only" else
            "Use the clean-trained RGB/MSDZip checkpoint; val remains clean and perturbation is applied only to requested report splits."
        ),
        "parameters": {
            "perturb_splits": sorted(perturb_splits),
            "contrast_alpha": args.contrast_alpha,
            "brightness_beta": args.brightness_beta,
            "shadow_strength": args.shadow_strength,
            "shadow_softness": args.shadow_softness,
        },
    }
    (args.out_dir / "variant_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Built temporal pair variant:", args.variant)
    print("Pairs by split:", report["pairs_by_split"])
    print("Modified pairs by split:", modified_counts)
    print("Manifest:", args.out_dir / "pairs_all.csv")
    print("Report:", args.out_dir / "variant_report.json")


if __name__ == "__main__":
    main()
