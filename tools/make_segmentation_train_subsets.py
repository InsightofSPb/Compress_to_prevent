#!/usr/bin/env python3
"""Create reproducible train tile subset lists without duplicating prepared PNG files.

The prepared training directory contains one clean tile and N augmented variants
per spatial crop. This tool writes MMSeg ``split`` text files containing tile
stems only, allowing training to use:

* all clean tiles plus the first K augmented variants for every crop;
* an optional random subset of base clean crops for quick sanity runs.

The tool does not change or copy any RGB/mask data.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write selectable train tile lists for MMSeg CustomDataset.")
    parser.add_argument("--tiles-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--aug-copies", type=int, choices=(0, 1, 2, 3), required=True,
                        help="Number of aug variants retained per selected clean crop.")
    parser.add_argument("--max-clean-tiles", type=int, default=None,
                        help="Optional random limit on base clean crops before adding aug variants.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None,
                        help="Output stem; auto-generated when omitted.")
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [{str(k): (v or "") for k, v in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("Empty tiles manifest: {}".format(path))
    return rows


def base_key(row: Dict[str, str]) -> str:
    return "{}|{}|{}|{}".format(row["source_image"], row["tile_idx"], row["x"], row["y"])


def variant_rank(variant: str) -> int:
    if variant == "clean":
        return -1
    if not variant.startswith("aug"):
        raise ValueError("Unexpected variant value: {}".format(variant))
    return int(variant[3:])


def main() -> None:
    args = parse_args()
    rows = read_rows(args.tiles_manifest)
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[base_key(row)].append(row)

    clean_keys = []
    for key, group in groups.items():
        variants = {row["variant"] for row in group}
        if "clean" not in variants:
            raise ValueError("No clean variant for base tile: {}".format(key))
        clean_keys.append(key)
    clean_keys = sorted(clean_keys)

    selected_keys = list(clean_keys)
    if args.max_clean_tiles is not None:
        if args.max_clean_tiles <= 0:
            raise ValueError("max-clean-tiles must be positive.")
        rng = random.Random(args.seed)
        selected_keys = sorted(rng.sample(clean_keys, min(args.max_clean_tiles, len(clean_keys))))

    selected_stems: List[str] = []
    variant_counts: Dict[str, int] = defaultdict(int)
    for key in selected_keys:
        group = sorted(groups[key], key=lambda row: variant_rank(row["variant"]))
        wanted = [row for row in group if row["variant"] == "clean"]
        wanted.extend(
            row for row in group
            if row["variant"].startswith("aug") and variant_rank(row["variant"]) < args.aug_copies
        )
        expected = 1 + args.aug_copies
        if len(wanted) != expected:
            raise ValueError("Base tile {} has {} selected variants; expected {}.".format(key, len(wanted), expected))
        for row in wanted:
            stem = Path(row["image_path"]).stem
            selected_stems.append(stem)
            variant_counts[row["variant"]] += 1

    name = args.name
    if not name:
        clean_part = "allclean" if args.max_clean_tiles is None else "{}clean".format(args.max_clean_tiles)
        name = "train_{}_plus{}aug_seed{}".format(clean_part, args.aug_copies, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.out_dir / (name + ".txt")
    split_path.write_text("".join("{}\n".format(stem) for stem in selected_stems), encoding="utf-8")
    report = {
        "tiles_manifest": str(args.tiles_manifest),
        "split_path": str(split_path),
        "available_clean_tiles": len(clean_keys),
        "selected_clean_tiles": len(selected_keys),
        "aug_copies_per_clean": args.aug_copies,
        "total_selected_samples": len(selected_stems),
        "variant_counts": dict(sorted(variant_counts.items())),
        "max_clean_tiles": args.max_clean_tiles,
        "seed": args.seed,
    }
    report_path = args.out_dir / (name + ".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Available clean tiles: {}".format(len(clean_keys)))
    print("Selected clean tiles: {}".format(len(selected_keys)))
    print("Aug copies per clean: {}".format(args.aug_copies))
    print("Total train samples: {}".format(len(selected_stems)))
    print("Split file: {}".format(split_path))
    print("Report: {}".format(report_path))


if __name__ == "__main__":
    main()
