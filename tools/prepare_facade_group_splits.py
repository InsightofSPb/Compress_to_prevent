#!/usr/bin/env python3
"""Create leakage-safe facade-level splits for RGB temporal and segmentation experiments.

The primary entity is an RGB observation. All observations of one ``facade_id``
are assigned to the same main split, so neither segmentation nor RGB residual
compression can evaluate on a facade that appeared in training.

Expected manifest columns:
    facade_id, year, image_path, aligned_image_path, mask_path
Only ``facade_id``, ``year`` and ``image_path`` are required. ``aligned_image_path``
is used later by the RGB residual pipeline. ``mask_path`` is used only when raw
segmentation folders are materialized for supervised targets/evaluation.

Outputs:
    facade_assignments.csv
    manifest_with_splits.csv
    manifests/manifest_{train,val,test}.csv
    facade_lists/facade_{train,val,test}.txt
    split_report.json
    segmentation_raw/<split>/{images,masks}/   (optional)
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
from typing import Dict, Iterable, List, Sequence, Tuple

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class FacadeStats:
    facade_id: str
    n_images: int
    n_dated_images: int
    years: Tuple[int, ...]

    @property
    def n_pairs(self) -> int:
        """Number of consecutive temporal RGB pairs possible for this facade."""
        return max(0, self.n_dated_images - 1)

    @property
    def has_pair(self) -> bool:
        return self.n_pairs > 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split RGB facade observations by facade_id without leakage.")
    p.add_argument("--manifest-csv", type=Path, required=True, help="RGB observation manifest_images.csv.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--search-trials", type=int, default=30000)
    p.add_argument("--force-train-top-k", type=int, default=1,
                   help="Keep this many longest temporal facade histories in train as anchors.")
    p.add_argument("--allow-no-pair-in-eval", action="store_true",
                   help="Allow single-image/no-temporal-pair facade groups in val/test.")
    p.add_argument("--min-eval-pairs", type=int, default=5,
                   help="Preferred minimum consecutive RGB pairs in each of val and test.")
    p.add_argument("--materialize-segmentation", choices=("none", "symlink", "copy"), default="none",
                   help="Optionally create raw segmentation RGB/mask directories from the same split.")
    p.add_argument("--overwrite-materialized", action="store_true")
    return p.parse_args()


def read_manifest(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [{k: (v or "") for k, v in row.items()} for row in reader]
    required = {"facade_id", "year", "image_path"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Missing required manifest columns: {sorted(missing)}")
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows, fields


def as_year(value: str) -> int | None:
    try:
        return int(value.strip()) if value.strip() else None
    except ValueError:
        return None


def build_stats(rows: Sequence[Dict[str, str]]) -> Dict[str, FacadeStats]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        facade_id = row["facade_id"].strip()
        if not facade_id:
            raise ValueError(f"Empty facade_id in row: {row}")
        grouped[facade_id].append(row)
    stats: Dict[str, FacadeStats] = {}
    for facade_id, items in grouped.items():
        years = tuple(sorted(y for y in (as_year(r["year"]) for r in items) if y is not None))
        stats[facade_id] = FacadeStats(facade_id, len(items), len(years), years)
    return stats


def validate_ratios(ratios: Dict[str, float]) -> None:
    if any(v < 0 for v in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("Non-negative train/val/test ratios must sum to 1.0.")


def totals(assignment: Dict[str, str], stats: Dict[str, FacadeStats]) -> Dict[str, Dict[str, int]]:
    result = {s: {"facades": 0, "images": 0, "pairs": 0} for s in SPLITS}
    for facade_id, split in assignment.items():
        item = stats[facade_id]
        result[split]["facades"] += 1
        result[split]["images"] += item.n_images
        result[split]["pairs"] += item.n_pairs
    return result


def score_assignment(assignment: Dict[str, str], stats: Dict[str, FacadeStats],
                     ratios: Dict[str, float], min_eval_pairs: int) -> float:
    counts = totals(assignment, stats)
    n_images = sum(x.n_images for x in stats.values())
    n_pairs = sum(x.n_pairs for x in stats.values())
    score = 0.0
    for split in SPLITS:
        target_i = max(1.0, ratios[split] * n_images)
        target_p = max(1.0, ratios[split] * n_pairs)
        score += 0.35 * ((counts[split]["images"] - target_i) / target_i) ** 2
        score += 0.65 * ((counts[split]["pairs"] - target_p) / target_p) ** 2
    for split in ("val", "test"):
        score += 50.0 * max(0, min_eval_pairs - counts[split]["pairs"])
        score += 1000.0 if counts[split]["facades"] == 0 else 0.0
    return score


def assign_facades(stats: Dict[str, FacadeStats], ratios: Dict[str, float], args: argparse.Namespace
                   ) -> Tuple[Dict[str, str], Dict[str, str]]:
    assignment: Dict[str, str] = {}
    reason: Dict[str, str] = {}
    if not args.allow_no_pair_in_eval:
        for facade_id, item in stats.items():
            if not item.has_pair:
                assignment[facade_id] = "train"
                reason[facade_id] = "no_rgb_temporal_pair_train_only"
    temporal = sorted((x for x in stats.values() if x.has_pair and x.facade_id not in assignment),
                      key=lambda x: (x.n_pairs, x.n_images, x.facade_id), reverse=True)
    for item in temporal[:max(0, args.force_train_top_k)]:
        assignment[item.facade_id] = "train"
        reason[item.facade_id] = "long_history_train_anchor"
    free = [f for f in stats if f not in assignment]
    if len(free) < 2:
        raise ValueError("Too few facade groups left for val/test; reduce --force-train-top-k.")
    rng = random.Random(args.seed)
    best: Dict[str, str] | None = None
    best_score = float("inf")
    weights = [ratios[s] for s in SPLITS]
    for _ in range(max(1, args.search_trials)):
        candidate = dict(assignment)
        for facade_id in free:
            candidate[facade_id] = rng.choices(SPLITS, weights=weights, k=1)[0]
        value = score_assignment(candidate, stats, ratios, args.min_eval_pairs)
        if value < best_score:
            best, best_score = candidate, value
    assert best is not None
    for facade_id in free:
        reason[facade_id] = "balanced_facade_group_assignment"
    return best, reason


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def materialize_segmentation(rows: Sequence[Dict[str, str]], assignment: Dict[str, str],
                             out_dir: Path, mode: str, overwrite: bool) -> Dict[str, int]:
    counts = {s: 0 for s in SPLITS}
    if mode == "none":
        return counts
    for row in rows:
        image = Path(row["image_path"])
        mask_text = row.get("mask_path", "").strip()
        if not image.exists() or not mask_text or not Path(mask_text).exists():
            raise FileNotFoundError(f"Cannot materialize RGB/mask pair: {image}, {mask_text}")
        mask = Path(mask_text)
        split = assignment[row["facade_id"]]
        image_dst = out_dir / "segmentation_raw" / split / "images" / image.name
        mask_dst = out_dir / "segmentation_raw" / split / "masks" / image.with_suffix(mask.suffix).name
        for src, dst in ((image, image_dst), (mask, mask_dst)):
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                if not overwrite:
                    raise FileExistsError(f"Destination exists: {dst}")
                dst.unlink()
            if mode == "copy":
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())
        counts[split] += 1
    return counts


def main() -> None:
    args = parse_args()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    validate_ratios(ratios)
    rows, fields = read_manifest(args.manifest_csv)
    stats = build_stats(rows)
    assignment, reasons = assign_facades(stats, ratios, args)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    assignment_rows = []
    for facade_id in sorted(stats):
        item = stats[facade_id]
        assignment_rows.append({"facade_id": facade_id, "split": assignment[facade_id],
                                "reason": reasons[facade_id], "n_images": item.n_images,
                                "n_dated_images": item.n_dated_images,
                                "years": ";".join(map(str, sorted(set(item.years)))),
                                "n_pairs_consecutive": item.n_pairs})
    write_csv(out / "facade_assignments.csv",
              ["facade_id", "split", "reason", "n_images", "n_dated_images", "years", "n_pairs_consecutive"],
              assignment_rows)

    out_fields = list(fields) if "split" in fields else list(fields) + ["split"]
    split_rows = []
    for row in rows:
        updated = dict(row)
        updated["split"] = assignment[row["facade_id"]]
        split_rows.append(updated)
    write_csv(out / "manifest_with_splits.csv", out_fields, split_rows)
    for split in SPLITS:
        write_csv(out / "manifests" / f"manifest_{split}.csv", out_fields,
                  [r for r in split_rows if r["split"] == split])
        ids = sorted(f for f, s in assignment.items() if s == split)
        target = out / "facade_lists" / f"facade_{split}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(f"{x}\n" for x in ids), encoding="utf-8")

    counts = totals(assignment, stats)
    materialized = materialize_segmentation(rows, assignment, out,
                                            args.materialize_segmentation, args.overwrite_materialized)
    report = {"input_manifest": str(args.manifest_csv), "seed": args.seed,
              "policy": {"primary_data": "RGB images", "group_key": "facade_id",
                         "no_pair_to_train": not args.allow_no_pair_in_eval,
                         "force_train_top_k": args.force_train_top_k,
                         "compression_target": "RGB residual current-minus-aligned_previous"},
              "counts": counts,
              "materialized_segmentation_pairs": materialized,
              "checks": {"facade_overlap_absent": True,
                         "val_pairs_ge_min": counts["val"]["pairs"] >= args.min_eval_pairs,
                         "test_pairs_ge_min": counts["test"]["pairs"] >= args.min_eval_pairs}}
    (out / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared RGB facade group split: {out}")
    for split in SPLITS:
        print(f"{split}: facades={counts[split]['facades']} images={counts[split]['images']} pairs={counts[split]['pairs']}")
    print(f"Use for compression: {out / 'manifest_with_splits.csv'}")
    if args.materialize_segmentation != "none":
        print(f"Use for supervised segmentation: {out / 'segmentation_raw'}")


if __name__ == "__main__":
    main()
