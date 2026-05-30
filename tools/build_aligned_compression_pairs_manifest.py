#!/usr/bin/env python3
"""Build a pair-level RGB compression manifest from saved aligned pair outputs.

Expected local layout under --pairs-root::

    <facade_id>/<year_prev>_<year_curr>/facades/<facade_id>/ref_spx/<year_curr>/
        src_warp_<year_prev>_to_<year_curr>.png
        valid_mask_<year_prev>_to_<year_curr>.png

The warped source is the previous RGB observation geometrically mapped into
coordinates of the current observation. The valid mask marks pixels where the
residual is meaningful after warping. Split assignments are read from the
facade-level split produced by ``prepare_facade_group_splits.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PAIR_DIR_RE = re.compile(r"^(?P<prev>\d{4})_(?P<curr>\d{4})$")
WARP_RE = re.compile(r"^src_warp_(?P<prev>\d{4})_to_(?P<curr>\d{4})\.png$")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest of real aligned RGB temporal pairs.")
    parser.add_argument("--pairs-root", type=Path, required=True,
                        help="Path ending in ref_spx_batch_out/pairs.")
    parser.add_argument("--raw-images-dir", type=Path, required=True)
    parser.add_argument("--facade-assignments", type=Path, required=True,
                        help="CSV with facade_id and split columns.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--valid-threshold", type=int, default=0,
                        help="Mask pixels strictly larger than this value are counted as valid.")
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


def load_assignments(path: Path) -> Dict[str, str]:
    rows = read_csv(path)
    assignments: Dict[str, str] = {}
    for row in rows:
        facade_id = row.get("facade_id", "").strip()
        split = row.get("split", "").strip()
        if not facade_id or split not in SPLITS:
            raise ValueError("Invalid facade assignment row: {}".format(row))
        if facade_id in assignments and assignments[facade_id] != split:
            raise ValueError("Conflicting split assignments for facade: {}".format(facade_id))
        assignments[facade_id] = split
    if not assignments:
        raise ValueError("Empty assignments file: {}".format(path))
    return assignments


def index_raw_images(raw_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            index.setdefault(path.stem, []).append(path.resolve())
    return index


def find_raw_image(index: Dict[str, List[Path]], facade_id: str, year: str) -> Path:
    stem = "{}_{}".format(facade_id, year)
    candidates = index.get(stem, [])
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("Raw RGB image not found for temporal observation: {}".format(stem))
    raise ValueError("Ambiguous raw RGB images for {}: {}".format(stem, candidates))


def image_info_and_coverage(warp: Path, current: Path, valid: Path, threshold: int) -> Tuple[int, int, int, float]:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow and numpy are required for pair-manifest validation.") from exc

    with Image.open(warp) as image:
        warp_size = image.size
    with Image.open(current) as image:
        curr_size = image.size
    with Image.open(valid) as image:
        valid_img = image.convert("L")
        valid_size = valid_img.size
        valid_array = np.asarray(valid_img)
    if warp_size != curr_size or warp_size != valid_size:
        raise ValueError(
            "Shape mismatch: warp={} current={} valid={} for {}".format(warp_size, curr_size, valid_size, warp)
        )
    width, height = warp_size
    n_valid = int((valid_array > threshold).sum())
    total = int(width * height)
    return width, height, n_valid, float(n_valid / max(total, 1))


def main() -> None:
    args = parse_args()
    if not args.pairs_root.is_dir():
        raise FileNotFoundError("Pair result directory does not exist: {}".format(args.pairs_root))
    if not args.raw_images_dir.is_dir():
        raise FileNotFoundError("Raw image directory does not exist: {}".format(args.raw_images_dir))
    assignments = load_assignments(args.facade_assignments)
    raw_index = index_raw_images(args.raw_images_dir)
    rows: List[Dict[str, object]] = []
    ignored_warps: List[str] = []

    for warp_path in sorted(args.pairs_root.rglob("src_warp_*.png")):
        match = WARP_RE.match(warp_path.name)
        if match is None:
            ignored_warps.append(str(warp_path))
            continue
        year_prev = match.group("prev")
        year_curr = match.group("curr")
        # The standard layout stores the pair identifier at the first pair-root level.
        relative = warp_path.relative_to(args.pairs_root)
        if len(relative.parts) < 2:
            raise ValueError("Unexpected warp path layout: {}".format(warp_path))
        facade_id = relative.parts[0]
        pair_folder = relative.parts[1]
        pair_match = PAIR_DIR_RE.match(pair_folder)
        if pair_match is None or pair_match.group("prev") != year_prev or pair_match.group("curr") != year_curr:
            raise ValueError("Pair folder/warp name disagree: {}".format(warp_path))
        nested_facades = [part for part in relative.parts if part == facade_id]
        if len(nested_facades) < 2:
            raise ValueError("Warp path does not repeat the expected facade id: {}".format(warp_path))
        if facade_id not in assignments:
            raise ValueError("Pair facade has no train/val/test assignment: {}".format(facade_id))

        valid_path = warp_path.parent / "valid_mask_{}_to_{}.png".format(year_prev, year_curr)
        if not valid_path.exists():
            raise FileNotFoundError("Valid alignment mask not found for warp: {}".format(warp_path))
        prev_path = find_raw_image(raw_index, facade_id, year_prev)
        curr_path = find_raw_image(raw_index, facade_id, year_curr)
        width, height, n_valid, valid_ratio = image_info_and_coverage(
            warp_path, curr_path, valid_path, args.valid_threshold
        )
        pair_id = "{}_{}_{}".format(facade_id, year_prev, year_curr)
        rows.append({
            "pair_id": pair_id,
            "facade_id": facade_id,
            "year_prev": year_prev,
            "year_curr": year_curr,
            "prev_image_path": str(prev_path),
            "curr_image_path": str(curr_path),
            "prev_aligned_path": str(warp_path.resolve()),
            "valid_mask_path": str(valid_path.resolve()),
            "valid_threshold": args.valid_threshold,
            "valid_pixel_count": n_valid,
            "valid_ratio": "{:.8f}".format(valid_ratio),
            "height": height,
            "width": width,
            "split": assignments[facade_id],
            "alignment_source": "ref_spx_batch_out",
        })

    if not rows:
        raise ValueError("No src_warp_*.png pair outputs found under: {}".format(args.pairs_root))
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        duplicates = [pair_id for pair_id in sorted(set(pair_ids)) if pair_ids.count(pair_id) > 1]
        raise ValueError("Duplicate aligned pair IDs: {}".format(duplicates))

    fields = [
        "pair_id", "facade_id", "year_prev", "year_curr", "prev_image_path", "curr_image_path",
        "prev_aligned_path", "valid_mask_path", "valid_threshold", "valid_pixel_count", "valid_ratio",
        "height", "width", "split", "alignment_source",
    ]
    write_csv(args.out_dir / "pairs_all.csv", fields, rows)
    for split in SPLITS:
        write_csv(args.out_dir / "pairs_{}.csv".format(split), fields,
                  [row for row in rows if row["split"] == split])

    counts = {split: sum(1 for row in rows if row["split"] == split) for split in SPLITS}
    report = {
        "pairs_root": str(args.pairs_root),
        "raw_images_dir": str(args.raw_images_dir),
        "facade_assignments": str(args.facade_assignments),
        "n_pairs": len(rows),
        "pairs_by_split": counts,
        "mean_valid_ratio": sum(float(row["valid_ratio"]) for row in rows) / len(rows),
        "min_valid_ratio": min(float(row["valid_ratio"]) for row in rows),
        "max_valid_ratio": max(float(row["valid_ratio"]) for row in rows),
        "ignored_warps": ignored_warps,
        "residual_definition": "RGB current minus geometrically warped previous, restricted by valid_mask_path",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pairs_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Aligned RGB pairs: {}".format(len(rows)))
    print("Pairs by split: {}".format(counts))
    print("Manifest: {}".format(args.out_dir / "pairs_all.csv"))
    print("Mean valid alignment ratio: {:.4f}".format(report["mean_valid_ratio"]))


if __name__ == "__main__":
    main()
