#!/usr/bin/env python3
"""Create leakage-safe facade-level splits shared by segmentation and MasksComp.

The main split is group-based: one facade_id belongs to exactly one of train,
val, or test.  By default facades without at least one temporal pair stay in
train because they cannot contribute to pair-based compression evaluation.

Outputs:
  facade_assignments.csv                 one row per facade
  manifest_with_splits.csv               original observations + split
  manifests/manifest_{train,val,test}.csv
  maskscomp_splits/facade_{train,val,test}.txt
  split_report.json
  segmentation_raw/<split>/{images,masks}/   optional materialized paired data
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SPLITS = ("train", "val", "test")
YEAR_SUFFIX_RE = re.compile(r"^(?P<facade>.+)_(?P<year>\d{4})$")


@dataclass
class FacadeStats:
    facade_id: str
    n_items: int
    n_known_year_items: int
    n_unknown_year: int
    years: List[int]

    @property
    def n_unique_years(self) -> int:
        return len(set(self.years))

    @property
    def n_pairs(self) -> int:
        # Compatible with consecutive temporal pairing after rows without years
        # are ignored: repeated same-year captures are still separate items.
        return max(0, self.n_known_year_items - 1)

    @property
    def has_temporal_pair(self) -> bool:
        return self.n_pairs > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe train/val/test splits at facade_id level and "
            "optionally materialize raw image/mask folders for segmentation."
        )
    )
    parser.add_argument("--manifest-csv", type=Path, required=True,
                        help="Observation manifest, e.g. manifest_masks.csv.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=30000,
                        help="Number of deterministic random assignments evaluated.")
    parser.add_argument(
        "--force-train-top-k",
        type=int,
        default=1,
        help=(
            "Keep this many largest temporal facade histories in train. "
            "Use 0 to let balancing allocate all temporal histories."
        ),
    )
    parser.add_argument(
        "--allow-singletons-in-eval",
        action="store_true",
        help=(
            "Allow facades without a temporal pair in val/test. By default they "
            "remain in train so all held-out facades support compression evaluation."
        ),
    )
    parser.add_argument(
        "--min-eval-pairs",
        type=int,
        default=5,
        help="Preferred minimum number of temporal pairs in each val/test split.",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Source image directory used only with --materialize copy/symlink.",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help=(
            "Optional mask directory. If omitted, non-empty mask_path values "
            "from the manifest are used."
        ),
    )
    parser.add_argument(
        "--image-path-column",
        default="image_path",
        help="Manifest column containing source image paths, if available.",
    )
    parser.add_argument(
        "--mask-path-column",
        default="mask_path",
        help="Manifest column containing source mask paths.",
    )
    parser.add_argument(
        "--materialize",
        choices=("none", "symlink", "copy"),
        default="none",
        help="Create segmentation_raw/<split>/{images,masks} from the group split.",
    )
    parser.add_argument(
        "--image-exts",
        default=".png,.jpg,.jpeg,.tif,.tiff,.webp",
        help="Comma-separated extensions used to resolve an image by mask stem.",
    )
    parser.add_argument(
        "--overwrite-materialized",
        action="store_true",
        help="Replace existing destination links/files during materialization.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{str(k): (v or "") for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows, fieldnames


def parse_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def infer_facade_and_year(row: Dict[str, str], mask_path_column: str) -> Tuple[str, Optional[int]]:
    facade_id = row.get("facade_id", "").strip()
    year = parse_int(row.get("year", ""))
    path_like = (
        row.get(mask_path_column, "").strip()
        or row.get("mask_name", "").strip()
        or row.get("stem", "").strip()
    )
    stem = Path(path_like).stem if path_like else ""
    match = YEAR_SUFFIX_RE.match(stem)
    if not facade_id and match:
        facade_id = match.group("facade")
    if year is None and match:
        year = int(match.group("year"))
    if not facade_id:
        facade_id = stem
    if not facade_id:
        raise ValueError(f"Could not resolve facade_id from row: {row}")
    return facade_id, year


def normalize_rows(rows: List[Dict[str, str]], mask_path_column: str) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for row in rows:
        facade_id, year = infer_facade_and_year(row, mask_path_column)
        copied = dict(row)
        copied["facade_id"] = facade_id
        if year is not None:
            copied["year"] = str(year)
        normalized.append(copied)
    return normalized


def make_stats(rows: Sequence[Dict[str, str]]) -> Dict[str, FacadeStats]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["facade_id"]].append(row)

    stats: Dict[str, FacadeStats] = {}
    for facade_id, items in grouped.items():
        years = [year for row in items if (year := parse_int(row.get("year", ""))) is not None]
        stats[facade_id] = FacadeStats(
            facade_id=facade_id,
            n_items=len(items),
            n_known_year_items=len(years),
            n_unknown_year=len(items) - len(years),
            years=sorted(years),
        )
    return stats


def validate_ratios(ratios: Dict[str, float]) -> None:
    if any(value < 0.0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative.")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")
    if ratios["train"] <= 0.0:
        raise ValueError("train-ratio must be positive.")


def counts_for_assignment(
    assignment: Dict[str, str], stats: Dict[str, FacadeStats]
) -> Dict[str, Dict[str, int]]:
    counts = {split: {"facades": 0, "items": 0, "pairs": 0} for split in SPLITS}
    for facade_id, split in assignment.items():
        item = stats[facade_id]
        counts[split]["facades"] += 1
        counts[split]["items"] += item.n_items
        counts[split]["pairs"] += item.n_pairs
    return counts


def assignment_score(
    assignment: Dict[str, str],
    stats: Dict[str, FacadeStats],
    ratios: Dict[str, float],
    min_eval_pairs: int,
) -> float:
    counts = counts_for_assignment(assignment, stats)
    total_items = sum(v.n_items for v in stats.values())
    total_pairs = sum(v.n_pairs for v in stats.values())

    score = 0.0
    for split in SPLITS:
        target_items = max(1.0, ratios[split] * total_items)
        score += 0.55 * ((counts[split]["items"] - target_items) / target_items) ** 2
        if total_pairs > 0:
            target_pairs = max(1.0, ratios[split] * total_pairs)
            score += 0.45 * ((counts[split]["pairs"] - target_pairs) / target_pairs) ** 2

    for split in ("val", "test"):
        deficit = max(0, min_eval_pairs - counts[split]["pairs"])
        score += 20.0 * deficit
        if counts[split]["facades"] == 0:
            score += 1000.0
    return score


def build_group_assignment(
    stats: Dict[str, FacadeStats],
    ratios: Dict[str, float],
    seed: int,
    trials: int,
    force_train_top_k: int,
    allow_singletons_in_eval: bool,
    min_eval_pairs: int,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not stats:
        raise ValueError("No facade groups found.")

    reason: Dict[str, str] = {}
    fixed_train: Dict[str, str] = {}

    if not allow_singletons_in_eval:
        for facade_id, item in stats.items():
            if not item.has_temporal_pair:
                fixed_train[facade_id] = "train"
                reason[facade_id] = "train_no_temporal_pair"

    temporal_candidates = [
        item for item in stats.values()
        if item.has_temporal_pair and item.facade_id not in fixed_train
    ]
    temporal_candidates.sort(
        key=lambda item: (item.n_pairs, item.n_items, item.n_unique_years, item.facade_id),
        reverse=True,
    )
    for item in temporal_candidates[: max(0, force_train_top_k)]:
        fixed_train[item.facade_id] = "train"
        reason[item.facade_id] = "train_anchor_long_history"

    free_ids = [facade_id for facade_id in stats if facade_id not in fixed_train]
    if not free_ids:
        raise ValueError(
            "No facade remains available for validation/test. "
            "Reduce --force-train-top-k or pass --allow-singletons-in-eval."
        )

    rng = random.Random(seed)
    weights = [ratios[split] for split in SPLITS]
    best_assignment: Optional[Dict[str, str]] = None
    best_score = float("inf")

    for _ in range(max(1, trials)):
        trial_assignment = dict(fixed_train)
        for facade_id in free_ids:
            trial_assignment[facade_id] = rng.choices(SPLITS, weights=weights, k=1)[0]
        score = assignment_score(trial_assignment, stats, ratios, min_eval_pairs)
        if score < best_score:
            best_score = score
            best_assignment = trial_assignment

    assert best_assignment is not None
    for facade_id in free_ids:
        reason[facade_id] = "balanced_group_assignment"
    return best_assignment, reason


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_id_lists(out_dir: Path, assignment: Dict[str, str]) -> None:
    split_dir = out_dir / "maskscomp_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        ids = sorted(facade_id for facade_id, assigned in assignment.items() if assigned == split)
        (split_dir / f"facade_{split}.txt").write_text(
            "".join(f"{facade_id}\n" for facade_id in ids),
            encoding="utf-8",
        )


def write_outputs(
    out_dir: Path,
    rows: List[Dict[str, str]],
    original_fieldnames: List[str],
    stats: Dict[str, FacadeStats],
    assignment: Dict[str, str],
    reasons: Dict[str, str],
    ratios: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    assignment_rows: List[Dict[str, object]] = []
    for facade_id in sorted(stats):
        item = stats[facade_id]
        assignment_rows.append(
            {
                "facade_id": facade_id,
                "split": assignment[facade_id],
                "reason": reasons[facade_id],
                "n_items": item.n_items,
                "n_known_year_items": item.n_known_year_items,
                "n_unique_years": item.n_unique_years,
                "years": ";".join(str(year) for year in sorted(set(item.years))),
                "n_pairs_consecutive": item.n_pairs,
                "n_unknown_year": item.n_unknown_year,
            }
        )
    write_csv(
        out_dir / "facade_assignments.csv",
        [
            "facade_id", "split", "reason", "n_items", "n_known_year_items",
            "n_unique_years", "years", "n_pairs_consecutive", "n_unknown_year",
        ],
        assignment_rows,
    )

    fieldnames = list(original_fieldnames)
    for required in ("facade_id", "year"):
        if required not in fieldnames:
            fieldnames.append(required)
    if "split" not in fieldnames:
        fieldnames.append("split")
    rows_with_split: List[Dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        copied["split"] = assignment[row["facade_id"]]
        rows_with_split.append(copied)
    write_csv(out_dir / "manifest_with_splits.csv", fieldnames, rows_with_split)
    for split in SPLITS:
        subset = [row for row in rows_with_split if row["split"] == split]
        write_csv(out_dir / "manifests" / f"manifest_{split}.csv", fieldnames, subset)

    write_id_lists(out_dir, assignment)
    counts = counts_for_assignment(assignment, stats)
    total_items = sum(s.n_items for s in stats.values())
    total_pairs = sum(s.n_pairs for s in stats.values())
    overlap_ok = all(
        not (
            set(f for f, sp in assignment.items() if sp == a)
            & set(f for f, sp in assignment.items() if sp == b)
        )
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    report = {
        "manifest_csv": str(args.manifest_csv),
        "seed": args.seed,
        "ratios_requested": ratios,
        "policy": {
            "group_key": "facade_id",
            "singletons_and_no_pair_to_train": not args.allow_singletons_in_eval,
            "force_train_top_k": args.force_train_top_k,
            "min_eval_pairs_preferred": args.min_eval_pairs,
            "search_trials": args.search_trials,
        },
        "totals": {
            "facades": len(stats),
            "items": total_items,
            "consecutive_pairs": total_pairs,
        },
        "splits": {
            split: {
                **counts[split],
                "item_ratio": counts[split]["items"] / max(1, total_items),
                "pair_ratio": counts[split]["pairs"] / max(1, total_pairs),
            }
            for split in SPLITS
        },
        "checks": {
            "facade_overlap_absent": overlap_ok,
            "val_has_preferred_pairs": counts["val"]["pairs"] >= args.min_eval_pairs,
            "test_has_preferred_pairs": counts["test"]["pairs"] >= args.min_eval_pairs,
        },
        "maskscomp_split_dir": str(out_dir / "maskscomp_splits"),
    }
    with (out_dir / "split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def build_stem_index(root: Path, extensions: Sequence[str]) -> Dict[str, List[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {root}")
    allowed = {ext.lower() for ext in extensions}
    index: Dict[str, List[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            index[path.stem].append(path)
    return index


def resolve_existing_path(raw_value: str, base_dir: Optional[Path]) -> Optional[Path]:
    if not raw_value.strip():
        return None
    path = Path(raw_value)
    if path.exists():
        return path
    if base_dir is not None:
        candidate = base_dir / path.name
        if candidate.exists():
            return candidate
    return None


def choose_by_stem(
    stem: str, index: Dict[str, List[Path]], label: str, facade_id: str
) -> Path:
    candidates = index.get(stem, [])
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No {label} matching stem '{stem}' for facade '{facade_id}'.")
    raise ValueError(
        f"Ambiguous {label} matching stem '{stem}' for facade '{facade_id}': {candidates}"
    )


def materialize_segmentation_data(
    out_dir: Path,
    rows: List[Dict[str, str]],
    assignment: Dict[str, str],
    args: argparse.Namespace,
) -> Dict[str, int]:
    if args.materialize == "none":
        return {}
    if args.images_dir is None:
        raise ValueError("--images-dir is required when --materialize is copy or symlink.")

    exts = [ext.strip().lower() for ext in args.image_exts.split(",") if ext.strip()]
    image_index = build_stem_index(args.images_dir, exts)
    mask_index = build_stem_index(args.masks_dir, exts) if args.masks_dir else {}
    counters = {split: 0 for split in SPLITS}

    for row in rows:
        facade_id = row["facade_id"]
        split = assignment[facade_id]
        raw_mask = row.get(args.mask_path_column, "")
        mask_path = resolve_existing_path(raw_mask, args.masks_dir)
        stem = Path(raw_mask or row.get("mask_name", "") or row.get("stem", "")).stem
        if mask_path is None:
            if not args.masks_dir:
                raise FileNotFoundError(
                    f"Mask path is not readable and --masks-dir was not set: {raw_mask}"
                )
            mask_path = choose_by_stem(stem, mask_index, "mask", facade_id)

        raw_image = row.get(args.image_path_column, "")
        image_path = resolve_existing_path(raw_image, args.images_dir)
        if image_path is None:
            image_path = choose_by_stem(stem, image_index, "image", facade_id)

        destination_root = out_dir / "segmentation_raw" / split
        image_dst = destination_root / "images" / image_path.name
        mask_dst = destination_root / "masks" / image_path.with_suffix(mask_path.suffix).name
        ensure_dirs(image_dst.parent, mask_dst.parent)

        for src, dst in ((image_path, image_dst), (mask_path, mask_dst)):
            if dst.exists() or dst.is_symlink():
                if not args.overwrite_materialized:
                    raise FileExistsError(
                        f"Destination exists: {dst}. Pass --overwrite-materialized to replace it."
                    )
                dst.unlink()
            if args.materialize == "copy":
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())
        counters[split] += 1
    return counters


def main() -> None:
    args = parse_args()
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    validate_ratios(ratios)
    raw_rows, fieldnames = read_rows(args.manifest_csv)
    rows = normalize_rows(raw_rows, args.mask_path_column)
    stats = make_stats(rows)
    assignment, reasons = build_group_assignment(
        stats=stats,
        ratios=ratios,
        seed=args.seed,
        trials=args.search_trials,
        force_train_top_k=args.force_train_top_k,
        allow_singletons_in_eval=args.allow_singletons_in_eval,
        min_eval_pairs=args.min_eval_pairs,
    )
    write_outputs(args.out_dir, rows, fieldnames, stats, assignment, reasons, ratios, args)
    materialized = materialize_segmentation_data(args.out_dir, rows, assignment, args)

    counts = counts_for_assignment(assignment, stats)
    print(f"Prepared leakage-safe grouped split in: {args.out_dir}")
    for split in SPLITS:
        print(
            f"{split}: facades={counts[split]['facades']} "
            f"images={counts[split]['items']} pairs={counts[split]['pairs']}"
        )
    if materialized:
        print(f"Materialized segmentation pairs: {materialized}")
    print(f"MasksComp split lists: {args.out_dir / 'maskscomp_splits'}")
    print(f"Full report: {args.out_dir / 'split_report.json'}")


if __name__ == "__main__":
    main()
