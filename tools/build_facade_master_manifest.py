#!/usr/bin/env python3
"""Build linked manifests for full segmentation data and temporal RGB data.

The local dataset has two related subsets:
  * all RGB/mask samples under raw_images/ and masks/ for segmentation;
  * a temporal subset listed in facades_images_with_years/temporal_manifest.csv
    for RGB residual/compression experiments.

Some existing temporal manifests use the historical column name ``mask_path``
for an RGB path. This script normalizes it to ``image_path`` while preserving
masks only from the true masks directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTS_DEFAULT = ".png,.jpg,.jpeg,.tif,.tiff,.webp"
SUFFIX_YEAR_RE = re.compile(r"^(?P<facade>.+)_(?P<year>\d{4})$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full segmentation and temporal RGB manifests.")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--raw-images-dir", type=Path, default=None)
    p.add_argument("--masks-dir", type=Path, default=None)
    p.add_argument("--temporal-manifest", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--image-exts", default=IMAGE_EXTS_DEFAULT)
    p.add_argument("--allow-missing-masks", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [{str(k): (v or "") for k, v in row.items()} for row in reader]
    return rows, fields


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def parse_exts(value: str) -> List[str]:
    return [ext.strip().lower() for ext in value.split(",") if ext.strip()]


def collect_files(folder: Path, exts: Sequence[str]) -> List[Path]:
    allowed = set(exts)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in allowed)


def index_by_name(paths: Sequence[Path]) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        result[path.name].append(path)
    return result


def index_by_stem(paths: Sequence[Path]) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        result[path.stem].append(path)
    return result


def find_unique(candidates: Sequence[Path], label: str) -> Optional[Path]:
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise ValueError("Ambiguous {} candidates: {}".format(label, [str(p) for p in candidates]))


def resolve_temporal_image(raw_value: str, raw_by_name: Dict[str, List[Path]], raw_by_stem: Dict[str, List[Path]]) -> Path:
    raw_path = Path(raw_value)
    if raw_path.exists():
        return raw_path.resolve()
    by_name = find_unique(raw_by_name.get(raw_path.name, []), "temporal image name")
    if by_name is not None:
        return by_name.resolve()
    by_stem = find_unique(raw_by_stem.get(raw_path.stem, []), "temporal image stem")
    if by_stem is not None:
        return by_stem.resolve()
    raise FileNotFoundError("Temporal RGB image cannot be matched to raw_images: {}".format(raw_value))


def resolve_mask(image: Path, masks_by_name: Dict[str, List[Path]], masks_by_stem: Dict[str, List[Path]]) -> Optional[Path]:
    by_name = find_unique(masks_by_name.get(image.name, []), "mask name")
    if by_name is not None:
        return by_name.resolve()
    by_stem = find_unique(masks_by_stem.get(image.stem, []), "mask stem")
    return by_stem.resolve() if by_stem is not None else None


def infer_non_temporal_id(image: Path) -> Tuple[str, str, str]:
    match = SUFFIX_YEAR_RE.match(image.stem)
    if match:
        return match.group("facade"), match.group("year"), "filename_suffix_year"
    return image.stem, "", "singleton_image_stem"


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    raw_dir = args.raw_images_dir or root / "raw_images"
    masks_dir = args.masks_dir or root / "masks"
    temporal_path = args.temporal_manifest or root / "facades_images_with_years" / "temporal_manifest.csv"
    out_dir = args.out_dir or root / "manifests" / "prepared"
    exts = parse_exts(args.image_exts)

    for required in (raw_dir, masks_dir):
        if not required.is_dir():
            raise FileNotFoundError("Directory does not exist: {}".format(required))
    if not temporal_path.is_file():
        raise FileNotFoundError("Temporal manifest does not exist: {}".format(temporal_path))

    raw_images = collect_files(raw_dir, exts)
    masks = collect_files(masks_dir, exts)
    raw_by_name, raw_by_stem = index_by_name(raw_images), index_by_stem(raw_images)
    masks_by_name, masks_by_stem = index_by_name(masks), index_by_stem(masks)
    temporal_rows, temporal_fields = read_csv(temporal_path)
    temporal_image_column = "image_path" if "image_path" in temporal_fields else "mask_path"
    if temporal_image_column not in temporal_fields:
        raise ValueError("Temporal manifest must have image_path or legacy mask_path with RGB paths.")

    temporal_lookup: Dict[str, Dict[str, str]] = {}
    normalized_temporal: List[Dict[str, str]] = []
    for index, row in enumerate(temporal_rows):
        image = resolve_temporal_image(row[temporal_image_column], raw_by_name, raw_by_stem)
        key = str(image)
        if key in temporal_lookup:
            raise ValueError("Duplicate temporal RGB image in manifest: {}".format(key))
        facade_id = row.get("facade_id", "").strip()
        year = row.get("year", "").strip()
        if not facade_id or not year:
            raise ValueError("Temporal row lacks facade_id/year: {}".format(row))
        normalized = {
            "temporal_row": str(index),
            "facade_id": facade_id,
            "year": year,
            "image_path": key,
            "mask_path": str(resolve_mask(image, masks_by_name, masks_by_stem) or ""),
            "is_temporal": "1",
            "facade_source": "temporal_manifest",
            "parse_rule": row.get("parse_rule", ""),
        }
        temporal_lookup[key] = normalized
        normalized_temporal.append(normalized)

    all_rows: List[Dict[str, str]] = []
    missing_masks: List[str] = []
    for sample_id, image in enumerate(raw_images):
        key = str(image.resolve())
        mask = resolve_mask(image, masks_by_name, masks_by_stem)
        if mask is None:
            missing_masks.append(key)
            if not args.allow_missing_masks:
                continue
        temporal = temporal_lookup.get(key)
        if temporal is not None:
            facade_id, year, source = temporal["facade_id"], temporal["year"], temporal["facade_source"]
            is_temporal = "1"
        else:
            facade_id, year, source = infer_non_temporal_id(image)
            is_temporal = "0"
        all_rows.append({
            "sample_id": str(sample_id),
            "facade_id": facade_id,
            "year": year,
            "image_path": key,
            "mask_path": str(mask) if mask is not None else "",
            "is_temporal": is_temporal,
            "facade_source": source,
        })

    if missing_masks and not args.allow_missing_masks:
        raise FileNotFoundError(
            "Missing masks for {} RGB images. First examples: {}. Pass --allow-missing-masks only for inspection.".format(
                len(missing_masks), missing_masks[:10]
            )
        )

    temporal_keys_in_full = {row["image_path"] for row in all_rows if row["is_temporal"] == "1"}
    not_in_full = sorted(set(temporal_lookup) - temporal_keys_in_full)
    if not_in_full:
        raise ValueError("Temporal images absent from full RGB manifest: {}".format(not_in_full[:10]))

    full_fields = ["sample_id", "facade_id", "year", "image_path", "mask_path", "is_temporal", "facade_source"]
    temporal_fields_out = ["temporal_row", "facade_id", "year", "image_path", "mask_path", "is_temporal", "facade_source", "parse_rule"]
    write_csv(out_dir / "segmentation_manifest_all.csv", full_fields, all_rows)
    write_csv(out_dir / "temporal_images_manifest.csv", temporal_fields_out, normalized_temporal)

    grouped: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n_all_images": 0, "n_temporal_images": 0})
    for row in all_rows:
        grouped[row["facade_id"]]["n_all_images"] += 1
        grouped[row["facade_id"]]["n_temporal_images"] += int(row["is_temporal"])
    summary_rows: List[Dict[str, object]] = []
    for facade_id in sorted(grouped):
        n_temporal = grouped[facade_id]["n_temporal_images"]
        summary_rows.append({
            "facade_id": facade_id,
            "n_all_images": grouped[facade_id]["n_all_images"],
            "n_temporal_images": n_temporal,
            "n_pairs_consecutive": max(0, n_temporal - 1),
            "has_temporal_pair": int(n_temporal >= 2),
        })
    write_csv(out_dir / "facade_summary_all.csv",
              ["facade_id", "n_all_images", "n_temporal_images", "n_pairs_consecutive", "has_temporal_pair"],
              summary_rows)

    report = {
        "dataset_root": str(root),
        "raw_images_dir": str(raw_dir),
        "masks_dir": str(masks_dir),
        "temporal_manifest_input": str(temporal_path),
        "temporal_rgb_source_column": temporal_image_column,
        "n_raw_images": len(raw_images),
        "n_masks": len(masks),
        "n_segmentation_rows": len(all_rows),
        "n_temporal_rgb_rows": len(normalized_temporal),
        "n_facade_groups": len(grouped),
        "n_temporal_facades": sum(1 for row in summary_rows if int(row["has_temporal_pair"]) == 1),
        "missing_masks": missing_masks,
        "notes": [
            "temporal manifest legacy mask_path was normalized to image_path when needed",
            "alignment/superpixel result files are intentionally not attached here",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest_build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote: {}".format(out_dir / "segmentation_manifest_all.csv"))
    print("Wrote: {}".format(out_dir / "temporal_images_manifest.csv"))
    print("Full RGB/mask samples: {}".format(len(all_rows)))
    print("Temporal RGB samples: {}".format(len(normalized_temporal)))
    print("Facade groups: {}".format(len(grouped)))


if __name__ == "__main__":
    main()
