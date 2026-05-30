#!/usr/bin/env python3
"""Create leakage-safe facade splits shared by segmentation and RGB compression.

Input is the linked full manifest produced by ``build_facade_master_manifest.py``.
For compression-aware balancing, pass the unsplit aligned pair manifest produced
by ``build_aligned_compression_pairs_manifest.py``. Every facade_id is assigned
to exactly one split; crops and augmentation are never created here.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class FacadeStats:
    facade_id: str
    n_all_images: int
    n_temporal_images: int
    n_aligned_pairs: int
    temporal_years: Tuple[int, ...]

    @property
    def has_pair(self) -> bool:
        return self.n_aligned_pairs > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split linked facade data by facade_id without leakage.")
    parser.add_argument("--master-manifest", type=Path, required=True,
                        help="segmentation_manifest_all.csv from build_facade_master_manifest.py")
    parser.add_argument("--compression-pairs-manifest", type=Path, default=None,
                        help="Optional unsplit aligned pairs_all.csv; strongly recommended.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=30000)
    parser.add_argument("--force-train-top-k", type=int, default=1,
                        help="Keep this many facades with most real aligned pairs as train anchors.")
    parser.add_argument("--min-eval-pairs", type=int, default=5,
                        help="Preferred minimum real aligned pairs in each of val and test.")
    parser.add_argument("--materialize-segmentation", choices=("none", "symlink", "copy"), default="none")
    parser.add_argument("--overwrite-materialized", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [{str(k): (v or "") for k, v in row.items()} for row in reader]
    return rows, fields


def read_master_manifest(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    rows, fields = read_csv(path)
    required = {"facade_id", "image_path", "mask_path", "is_temporal"}
    missing = required - set(fields)
    if missing:
        raise ValueError("Missing required master-manifest columns: {}".format(sorted(missing)))
    if not rows:
        raise ValueError("Empty manifest: {}".format(path))
    return rows, fields


def as_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def temporal_flag(row: Dict[str, str]) -> bool:
    return row.get("is_temporal", "").strip().lower() in {"1", "true", "yes"}


def load_real_pair_counts(path: Optional[Path]) -> Tuple[Dict[str, int], int]:
    if path is None:
        return {}, 0
    rows, fields = read_csv(path)
    if "facade_id" not in fields or "pair_id" not in fields:
        raise ValueError("Compression pair manifest must contain facade_id and pair_id.")
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Compression pair manifest contains duplicate pair_id values.")
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        facade_id = row["facade_id"].strip()
        if not facade_id:
            raise ValueError("Empty facade_id in compression pair row: {}".format(row))
        counts[facade_id] += 1
    return dict(counts), len(rows)


def build_stats(rows: Sequence[Dict[str, str]], pair_counts: Dict[str, int]) -> Dict[str, FacadeStats]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        facade_id = row["facade_id"].strip()
        if not facade_id:
            raise ValueError("Empty facade_id in row: {}".format(row))
        grouped[facade_id].append(row)
    unknown_pair_facades = sorted(set(pair_counts) - set(grouped))
    if unknown_pair_facades:
        raise ValueError("Aligned pairs refer to facades absent from master manifest: {}".format(unknown_pair_facades))
    stats: Dict[str, FacadeStats] = {}
    for facade_id, items in grouped.items():
        temporal_items = [row for row in items if temporal_flag(row)]
        years = tuple(sorted(
            year for year in (as_int(row.get("year", "")) for row in temporal_items)
            if year is not None
        ))
        if len(years) != len(temporal_items):
            raise ValueError("Temporal RGB row without valid year for facade: {}".format(facade_id))
        approximate_pairs = max(0, len(temporal_items) - 1)
        real_pairs = pair_counts.get(facade_id, approximate_pairs)
        stats[facade_id] = FacadeStats(facade_id, len(items), len(temporal_items), real_pairs, years)
    return stats


def validate_ratios(ratios: Dict[str, float]) -> None:
    if any(value < 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("Non-negative train/val/test ratios must sum to 1.0.")


def totals(assignment: Dict[str, str], stats: Dict[str, FacadeStats]) -> Dict[str, Dict[str, int]]:
    counts = {split: {"facades": 0, "all_images": 0, "temporal_images": 0, "aligned_pairs": 0}
              for split in SPLITS}
    for facade_id, split in assignment.items():
        item = stats[facade_id]
        counts[split]["facades"] += 1
        counts[split]["all_images"] += item.n_all_images
        counts[split]["temporal_images"] += item.n_temporal_images
        counts[split]["aligned_pairs"] += item.n_aligned_pairs
    return counts


def score_assignment(assignment: Dict[str, str], stats: Dict[str, FacadeStats],
                     ratios: Dict[str, float], min_eval_pairs: int) -> float:
    counts = totals(assignment, stats)
    total_images = sum(item.n_all_images for item in stats.values())
    total_pairs = sum(item.n_aligned_pairs for item in stats.values())
    score = 0.0
    for split in SPLITS:
        target_images = max(1.0, ratios[split] * total_images)
        target_pairs = max(1.0, ratios[split] * total_pairs)
        score += 0.35 * ((counts[split]["all_images"] - target_images) / target_images) ** 2
        score += 0.65 * ((counts[split]["aligned_pairs"] - target_pairs) / target_pairs) ** 2
    for split in ("val", "test"):
        score += 50.0 * max(0, min_eval_pairs - counts[split]["aligned_pairs"])
        if counts[split]["facades"] == 0:
            score += 1000.0
    return score


def assign_facades(stats: Dict[str, FacadeStats], ratios: Dict[str, float],
                   args: argparse.Namespace) -> Tuple[Dict[str, str], Dict[str, str]]:
    fixed: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    temporal = sorted((item for item in stats.values() if item.has_pair),
                      key=lambda item: (item.n_aligned_pairs, item.n_temporal_images, item.facade_id), reverse=True)
    for item in temporal[:max(0, args.force_train_top_k)]:
        fixed[item.facade_id] = "train"
        reasons[item.facade_id] = "largest_aligned_pair_history_train_anchor"
    free = [facade_id for facade_id in stats if facade_id not in fixed]
    if len(free) < 2:
        raise ValueError("Too few free facade groups for validation/test.")
    rng = random.Random(args.seed)
    weights = [ratios[split] for split in SPLITS]
    best: Optional[Dict[str, str]] = None
    best_score = float("inf")
    for _ in range(max(1, args.search_trials)):
        candidate = dict(fixed)
        for facade_id in free:
            candidate[facade_id] = rng.choices(SPLITS, weights=weights, k=1)[0]
        value = score_assignment(candidate, stats, ratios, args.min_eval_pairs)
        if value < best_score:
            best, best_score = candidate, value
    if best is None:
        raise RuntimeError("Could not construct facade assignment.")
    for facade_id in free:
        reasons[facade_id] = "balanced_group_assignment"
    return best, reasons


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def materialize_segmentation(rows: Sequence[Dict[str, str]], assignment: Dict[str, str],
                             out_dir: Path, mode: str, overwrite: bool) -> Dict[str, int]:
    counts = {split: 0 for split in SPLITS}
    if mode == "none":
        return counts
    for row in rows:
        image, mask = Path(row["image_path"]), Path(row["mask_path"])
        if not image.exists() or not mask.exists():
            raise FileNotFoundError("Cannot materialize RGB/mask pair: {}, {}".format(image, mask))
        split = assignment[row["facade_id"]]
        image_dst = out_dir / "segmentation_raw" / split / "images" / image.name
        mask_dst = out_dir / "segmentation_raw" / split / "masks" / image.with_suffix(mask.suffix).name
        for src, dst in ((image, image_dst), (mask, mask_dst)):
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                if not overwrite:
                    raise FileExistsError("Destination exists: {}".format(dst))
                dst.unlink()
            shutil.copy2(src, dst) if mode == "copy" else dst.symlink_to(src.resolve())
        counts[split] += 1
    return counts


def main() -> None:
    args = parse_args()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    validate_ratios(ratios)
    rows, fields = read_master_manifest(args.master_manifest)
    pair_counts, n_pair_rows = load_real_pair_counts(args.compression_pairs_manifest)
    stats = build_stats(rows, pair_counts)
    assignment, reasons = assign_facades(stats, ratios, args)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    assignment_rows: List[Dict[str, object]] = []
    for facade_id in sorted(stats):
        item = stats[facade_id]
        assignment_rows.append({"facade_id": facade_id, "split": assignment[facade_id], "reason": reasons[facade_id],
                                "n_all_images": item.n_all_images, "n_temporal_images": item.n_temporal_images,
                                "temporal_years": ";".join(str(y) for y in sorted(set(item.temporal_years))),
                                "n_aligned_pairs": item.n_aligned_pairs, "has_aligned_pair": int(item.has_pair)})
    write_csv(out / "facade_assignments.csv",
              ["facade_id", "split", "reason", "n_all_images", "n_temporal_images", "temporal_years",
               "n_aligned_pairs", "has_aligned_pair"], assignment_rows)

    out_fields = list(fields) if "split" in fields else list(fields) + ["split"]
    full_rows: List[Dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        updated["split"] = assignment[row["facade_id"]]
        full_rows.append(updated)
    temporal_rows = [row for row in full_rows if temporal_flag(row)]
    write_csv(out / "segmentation_all_with_splits.csv", out_fields, full_rows)
    write_csv(out / "temporal_all_with_splits.csv", out_fields, temporal_rows)
    for split in SPLITS:
        write_csv(out / "segmentation" / "segmentation_{}.csv".format(split), out_fields,
                  [row for row in full_rows if row["split"] == split])
        write_csv(out / "temporal_rgb" / "temporal_{}.csv".format(split), out_fields,
                  [row for row in temporal_rows if row["split"] == split])
        ids = sorted(facade_id for facade_id, assigned in assignment.items() if assigned == split)
        list_path = out / "facade_lists" / "facade_{}.txt".format(split)
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("".join("{}\n".format(value) for value in ids), encoding="utf-8")

    counts = totals(assignment, stats)
    materialized = materialize_segmentation(full_rows, assignment, out,
                                            args.materialize_segmentation, args.overwrite_materialized)
    report = {"input_master_manifest": str(args.master_manifest),
              "compression_pairs_manifest": str(args.compression_pairs_manifest) if args.compression_pairs_manifest else "",
              "real_pair_manifest_used": args.compression_pairs_manifest is not None,
              "n_real_pair_rows": n_pair_rows, "seed": args.seed, "ratios_requested": ratios,
              "policy": {"group_key": "facade_id", "segmentation_source": "all RGB images and target masks",
                         "compression_source": "real aligned temporal RGB pairs", "singletons_allowed_in_eval": True,
                         "long_temporal_train_anchors": args.force_train_top_k},
              "counts": counts, "materialized_segmentation_pairs": materialized,
              "checks": {"facade_overlap_absent": True,
                         "val_aligned_pairs_ge_min": counts["val"]["aligned_pairs"] >= args.min_eval_pairs,
                         "test_aligned_pairs_ge_min": counts["test"]["aligned_pairs"] >= args.min_eval_pairs}}
    (out / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Prepared linked group split: {}".format(out))
    for split in SPLITS:
        print("{}: facades={} all_images={} temporal_images={} aligned_pairs={}".format(
            split, counts[split]["facades"], counts[split]["all_images"],
            counts[split]["temporal_images"], counts[split]["aligned_pairs"]))
    print("Segmentation manifests: {}".format(out / "segmentation"))
    print("Rebuild aligned pair manifest with --facade-assignments {} to attach splits.".format(out / "facade_assignments.csv"))


if __name__ == "__main__":
    main()
