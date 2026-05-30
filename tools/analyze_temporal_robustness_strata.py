#!/usr/bin/env python3
"""Stratified robustness analysis for temporal heatmap scores.

This script reuses already computed tile scores and aligned GT reference maps.
It reports ranking metrics on subsets of the report split grouped by:

* alignment coverage proxy: pair-level valid overlap ratio after warping;
* temporal gap: elapsed years between the aligned RGB observations.

The pair-level valid ratio is intentionally described as a coverage/difficulty
proxy rather than a direct estimate of geometric registration accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze temporal heatmap metrics across robustness strata.")
    parser.add_argument(
        "--scores", action="append", required=True,
        help=("Score source as LABEL=PATH. LABEL 'basic' preserves method names contained in the CSV; "
              "any other label renames all score rows from that CSV, e.g. 'RGB/MSDZip=...'.")
    )
    parser.add_argument("--references-csv", type=Path, required=True)
    parser.add_argument("--pair-metadata-csv", type=Path, required=True)
    parser.add_argument(
        "--reference",
        choices=("inspection_relevant_change", "damage_or_repair_change", "damage_type_change",
                 "intervention_or_content_change", "any_semantic_change", "damage_presence_change"),
        default="inspection_relevant_change",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-min-change-ratio", type=float, default=0.05)
    parser.add_argument("--top-k-percent", type=float, default=10.0)
    parser.add_argument("--coverage-thresholds", type=str, default="0.75,0.90",
                        help="Two comma-separated boundaries for low/medium/high valid-overlap strata.")
    parser.add_argument("--out-dir", type=Path, required=True)
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


def load_score_sources(items: Sequence[str], split: str) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError("--scores must use LABEL=PATH syntax: {}".format(item))
        label, raw_path = item.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
        if not path.is_file():
            raise FileNotFoundError("Score file not found: {}".format(path))
        rows = [row for row in read_csv(path) if row.get("split", "") == split]
        for row in rows:
            converted = dict(row)
            if label.lower() != "basic":
                converted["method"] = label
            converted["score_source"] = str(path)
            output.append(converted)
    return output


def read_reference(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError("Cannot read reference map: {}".format(path))
    if image.ndim == 3:
        image = image[..., 0]
    return image


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def safe_corr(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def metrics(rows: List[Dict[str, object]], top_fraction: float) -> Dict[str, object]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception as exc:
        raise RuntimeError("This script requires scikit-learn: pip install scikit-learn") from exc
    y = np.asarray([int(row["target_positive"]) for row in rows], dtype=np.uint8)
    score = np.asarray([float(row["tile_score"]) for row in rows], dtype=np.float64)
    ratio = np.asarray([float(row["target_change_ratio"]) for row in rows], dtype=np.float64)
    prevalence = float(y.mean()) if len(y) else 0.0
    if len(np.unique(y)) < 2:
        auprc = auroc = None
    else:
        auprc = float(average_precision_score(y, score))
        auroc = float(roc_auc_score(y, score))
    n_top = max(1, int(np.ceil(len(y) * top_fraction)))
    order = np.argsort(-score)
    precision_top = float(y[order[:n_top]].mean()) if len(y) else None
    return {
        "n_pairs": len({str(row["pair_id"]) for row in rows}),
        "n_tiles": len(rows),
        "positive_ratio": prevalence,
        "AUPRC": auprc,
        "random_AUPRC_baseline": prevalence,
        "AUPRC_lift_over_random": (auprc / prevalence) if auprc is not None and prevalence > 0 else None,
        "AUROC": auroc,
        "Precision@Top{}%".format(int(round(top_fraction * 100))): precision_top,
        "Precision@Top{}%_lift".format(int(round(top_fraction * 100))): (precision_top / prevalence) if precision_top is not None and prevalence > 0 else None,
        "Spearman_target_ratio": safe_corr(average_ranks(ratio), average_ranks(score)),
        "mean_target_change_ratio": float(ratio.mean()) if len(ratio) else None,
    }


def coverage_stratum(value: float, low: float, high: float) -> str:
    if value < low:
        return "low_coverage_<{}".format(low)
    if value < high:
        return "medium_coverage_{}-{}".format(low, high)
    return "high_coverage_>={}".format(high)


def temporal_gap_stratum(gap: Optional[int]) -> str:
    if gap is None:
        return "unknown_gap"
    if gap <= 3:
        return "short_gap_1-3y"
    if gap <= 7:
        return "medium_gap_4-7y"
    return "long_gap_>=8y"


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target_min_change_ratio <= 1.0:
        raise ValueError("target-min-change-ratio must be in [0, 1]")
    if not 0.0 < args.top_k_percent <= 100.0:
        raise ValueError("top-k-percent must be in (0, 100]")
    thresholds = [float(item.strip()) for item in args.coverage_thresholds.split(",") if item.strip()]
    if len(thresholds) != 2 or not 0.0 <= thresholds[0] < thresholds[1] <= 1.0:
        raise ValueError("coverage-thresholds must be two ascending values in [0, 1]")
    low_coverage, high_coverage = thresholds

    refs = {row["pair_id"]: row for row in read_csv(args.references_csv)}
    metadata = {row["pair_id"]: row for row in read_csv(args.pair_metadata_csv)}
    ref_column = args.reference + "_path"
    score_rows = load_score_sources(args.scores, args.split)
    maps: Dict[str, np.ndarray] = {}
    enriched: List[Dict[str, object]] = []

    for row in score_rows:
        pair_id = row["pair_id"]
        if pair_id not in refs or pair_id not in metadata:
            raise KeyError("Missing reference or pair metadata for {}".format(pair_id))
        reference_path = refs[pair_id].get(ref_column, "")
        if not reference_path:
            raise KeyError("Missing reference {} for {}".format(ref_column, pair_id))
        if reference_path not in maps:
            maps[reference_path] = read_reference(Path(reference_path))
        reference = maps[reference_path]
        tile_size = int(float(row["tile_size"]))
        tile_x = int(float(row["tile_x"]))
        tile_y = int(float(row["tile_y"]))
        x0, y0 = tile_x * tile_size, tile_y * tile_size
        tile = reference[y0:min(y0 + tile_size, reference.shape[0]), x0:min(x0 + tile_size, reference.shape[1])]
        valid = tile != args.invalid_label
        if int(valid.sum()) == 0:
            continue
        target_ratio = float((tile[valid] == 1).mean())
        meta = metadata[pair_id]
        valid_ratio = float(meta["valid_ratio"])
        try:
            year_prev = int(float(meta.get("year_prev", "")))
            year_curr = int(float(meta.get("year_curr", "")))
            year_gap: Optional[int] = year_curr - year_prev
        except ValueError:
            year_gap = None
        enriched.append({
            "pair_id": pair_id,
            "method": row["method"],
            "tile_score": float(row["tile_score"]),
            "target_change_ratio": target_ratio,
            "target_positive": int(target_ratio >= args.target_min_change_ratio),
            "pair_valid_ratio": valid_ratio,
            "year_gap": year_gap if year_gap is not None else "",
            "coverage_stratum": coverage_stratum(valid_ratio, low_coverage, high_coverage),
            "temporal_gap_stratum": temporal_gap_stratum(year_gap),
        })

    if not enriched:
        raise ValueError("No test tiles available for robustness analysis")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_fields = ["pair_id", "method", "tile_score", "target_change_ratio", "target_positive", "pair_valid_ratio", "year_gap", "coverage_stratum", "temporal_gap_stratum"]
    write_csv(args.out_dir / "tile_scores_with_robustness_strata.csv", detail_fields, enriched)

    summary_rows: List[Dict[str, object]] = []
    top_fraction = args.top_k_percent / 100.0
    methods = sorted({str(row["method"]) for row in enriched})
    stratum_specs = [
        ("overall", ["overall"]),
        ("alignment_coverage_proxy", sorted({str(row["coverage_stratum"]) for row in enriched})),
        ("temporal_gap", sorted({str(row["temporal_gap_stratum"]) for row in enriched})),
    ]
    for method in methods:
        method_rows = [row for row in enriched if row["method"] == method]
        for analysis, labels in stratum_specs:
            for label in labels:
                if analysis == "overall":
                    selected = method_rows
                elif analysis == "alignment_coverage_proxy":
                    selected = [row for row in method_rows if row["coverage_stratum"] == label]
                else:
                    selected = [row for row in method_rows if row["temporal_gap_stratum"] == label]
                if not selected:
                    continue
                summary_rows.append({
                    "reference": args.reference,
                    "split": args.split,
                    "method": method,
                    "analysis": analysis,
                    "stratum": label,
                    **metrics(selected, top_fraction),
                })

    fields = list(summary_rows[0].keys())
    write_csv(args.out_dir / "robustness_strata_summary.csv", fields, summary_rows)
    report = {
        "reference": args.reference,
        "split": args.split,
        "target_min_change_ratio": args.target_min_change_ratio,
        "top_k_percent": args.top_k_percent,
        "coverage_thresholds": thresholds,
        "coverage_interpretation": "Pair-level valid overlap after alignment is a coverage/difficulty proxy, not a direct registration-accuracy measurement.",
        "temporal_gap_interpretation": "Elapsed years between paired observations; it may mix true change burden with acquisition-condition differences.",
        "n_enriched_tile_score_rows": len(enriched),
        "methods": methods,
        "outputs": {
            "detail": str(args.out_dir / "tile_scores_with_robustness_strata.csv"),
            "summary": str(args.out_dir / "robustness_strata_summary.csv"),
        },
    }
    (args.out_dir / "robustness_strata_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Reference:", args.reference)
    print("Split:", args.split)
    print("Methods:", methods)
    print("\nAlignment coverage proxy strata:")
    for row in summary_rows:
        if row["analysis"] == "alignment_coverage_proxy":
            print("{:<24s} {:<28s} pairs={:>2d} tiles={:>5d} AUPRC={} lift={} AUROC={}".format(
                str(row["method"]), str(row["stratum"]), int(row["n_pairs"]), int(row["n_tiles"]),
                "{:.4f}".format(row["AUPRC"]) if row["AUPRC"] is not None else "NA",
                "{:.4f}".format(row["AUPRC_lift_over_random"]) if row["AUPRC_lift_over_random"] is not None else "NA",
                "{:.4f}".format(row["AUROC"]) if row["AUROC"] is not None else "NA",
            ))
    print("\nTemporal gap strata:")
    for row in summary_rows:
        if row["analysis"] == "temporal_gap":
            print("{:<24s} {:<20s} pairs={:>2d} tiles={:>5d} AUPRC={} lift={} AUROC={}".format(
                str(row["method"]), str(row["stratum"]), int(row["n_pairs"]), int(row["n_tiles"]),
                "{:.4f}".format(row["AUPRC"]) if row["AUPRC"] is not None else "NA",
                "{:.4f}".format(row["AUPRC_lift_over_random"]) if row["AUPRC_lift_over_random"] is not None else "NA",
                "{:.4f}".format(row["AUROC"]) if row["AUROC"] is not None else "NA",
            ))
    print("\nSummary:", args.out_dir / "robustness_strata_summary.csv")
    print("Report:", args.out_dir / "robustness_strata_report.json")


if __name__ == "__main__":
    main()
