#!/usr/bin/env python3
"""Build valid-masked aligned RGB residual images for temporal compression."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression.residuals import build_residual_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RGB residual dataset from aligned temporal pairs.")
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_residual_dataset(args.pairs_manifest, args.out_dir)
    split_counts = {}
    valid_ratios = []
    for row in rows:
        split = str(row.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        valid_ratios.append(float(row.get("valid_ratio", 0.0)))
    report = {
        "pairs_manifest": str(args.pairs_manifest),
        "out_dir": str(args.out_dir),
        "n_residual_pairs": len(rows),
        "pairs_by_split": split_counts,
        "valid_ratio_mean": sum(valid_ratios) / max(len(valid_ratios), 1),
        "valid_ratio_min": min(valid_ratios) if valid_ratios else None,
        "valid_ratio_max": max(valid_ratios) if valid_ratios else None,
        "invalid_residual_policy": "zero_filled_and_excluded_from_scoring",
        "residual_definition": "current RGB minus geometrically warped previous RGB modulo 256 on valid alignment pixels",
    }
    report_path = args.out_dir / "residual_build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Built residual pairs: {}".format(len(rows)))
    print("Pairs by split: {}".format(split_counts))
    print("Valid ratio mean/min/max: {:.6f}/{:.6f}/{:.6f}".format(
        report["valid_ratio_mean"], report["valid_ratio_min"], report["valid_ratio_max"]
    ))
    print("Manifest: {}".format(args.out_dir / "residual_manifest.csv"))
    print("Report: {}".format(report_path))


if __name__ == "__main__":
    main()
