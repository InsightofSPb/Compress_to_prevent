#!/usr/bin/env python3
"""Pair-level bootstrap confidence intervals for temporal heatmap method comparisons.

The script joins tile score CSVs to an aligned GT reference map, restricts the
analysis to one report split (test by default), and resamples temporal pairs
with replacement. All tiles belonging to a sampled pair are retained together;
this avoids treating spatial tiles from one image pair as independent samples.

Primary output is the paired bootstrap distribution of metric differences
between an anchor method (typically RGB/MSDZip) and every comparator.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair-level bootstrap comparison of temporal heatmap methods.")
    parser.add_argument(
        "--scores", action="append", required=True,
        help=("Score source as LABEL=PATH. LABEL 'basic' preserves method names in the CSV; "
              "another label renames all rows in that source, e.g. 'RGB/MSDZip=...'.")
    )
    parser.add_argument("--references-csv", type=Path, required=True)
    parser.add_argument(
        "--reference",
        choices=("inspection_relevant_change", "damage_or_repair_change", "damage_type_change",
                 "intervention_or_content_change", "any_semantic_change", "damage_presence_change"),
        default="inspection_relevant_change",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-min-change-ratio", type=float, default=0.05)
    parser.add_argument("--top-k-percent", type=float, default=10.0)
    parser.add_argument("--anchor-method", default="RGB/MSDZip")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invalid-label", type=int, default=255)
    parser.add_argument("--out-dir", type=Path, required=True)
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
        label, path = label.strip(), Path(raw_path.strip())
        if not path.is_file():
            raise FileNotFoundError("Score file not found: {}".format(path))
        for row in read_csv(path):
            if row.get("split", "") != split:
                continue
            converted = dict(row)
            if label.lower() != "basic":
                converted["method"] = label
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


def metric_values(rows: List[Dict[str, object]], top_fraction: float) -> Dict[str, Optional[float]]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray([int(row["target_positive"]) for row in rows], dtype=np.uint8)
    score = np.asarray([float(row["tile_score"]) for row in rows], dtype=np.float64)
    ratio = np.asarray([float(row["target_change_ratio"]) for row in rows], dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {"AUPRC": None, "AUROC": None, "Precision@Top10%": None, "Spearman": None}
    n = max(1, int(np.ceil(len(y) * top_fraction)))
    top = np.argsort(-score)[:n]
    return {
        "AUPRC": float(average_precision_score(y, score)),
        "AUROC": float(roc_auc_score(y, score)),
        "Precision@Top10%": float(y[top].mean()),
        "Spearman": safe_corr(average_ranks(ratio), average_ranks(score)),
    }


def percentile_ci(values: List[float], confidence: float) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [alpha, 1.0 - alpha])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target_min_change_ratio <= 1.0:
        raise ValueError("target-min-change-ratio must be in [0, 1]")
    if not 0.0 < args.top_k_percent <= 100.0:
        raise ValueError("top-k-percent must be in (0, 100]")
    if args.n_bootstrap <= 0:
        raise ValueError("n-bootstrap must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    refs = {row["pair_id"]: row for row in read_csv(args.references_csv)}
    ref_column = args.reference + "_path"
    scores = load_score_sources(args.scores, args.split)
    maps: Dict[str, np.ndarray] = {}
    enriched: List[Dict[str, object]] = []
    for row in scores:
        pair_id = row["pair_id"]
        ref_path = refs.get(pair_id, {}).get(ref_column, "")
        if not ref_path:
            raise KeyError("Missing {} reference for pair {}".format(args.reference, pair_id))
        if ref_path not in maps:
            maps[ref_path] = read_reference(Path(ref_path))
        reference = maps[ref_path]
        tile_size = int(float(row["tile_size"]))
        tile_x, tile_y = int(float(row["tile_x"])), int(float(row["tile_y"]))
        x0, y0 = tile_x * tile_size, tile_y * tile_size
        tile = reference[y0:min(y0 + tile_size, reference.shape[0]), x0:min(x0 + tile_size, reference.shape[1])]
        valid = tile != args.invalid_label
        if not int(valid.sum()):
            continue
        change_ratio = float((tile[valid] == 1).mean())
        enriched.append({
            "pair_id": pair_id,
            "method": row["method"],
            "tile_x": tile_x,
            "tile_y": tile_y,
            "tile_size": tile_size,
            "tile_score": float(row["tile_score"]),
            "target_change_ratio": change_ratio,
            "target_positive": int(change_ratio >= args.target_min_change_ratio),
        })

    methods = sorted({str(row["method"]) for row in enriched})
    if args.anchor_method not in methods:
        raise ValueError("Anchor method {!r} absent; available methods: {}".format(args.anchor_method, methods))
    pair_ids = sorted({str(row["pair_id"]) for row in enriched})
    by_method_pair: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        method: {pair_id: [] for pair_id in pair_ids} for method in methods
    }
    for row in enriched:
        by_method_pair[str(row["method"])][str(row["pair_id"])].append(row)
    for method in methods:
        missing = [pair_id for pair_id in pair_ids if not by_method_pair[method][pair_id]]
        if missing:
            raise ValueError("Method {} missing pairs required for paired bootstrap: {}".format(method, missing))

    top_fraction = args.top_k_percent / 100.0
    observed = {method: metric_values([row for pair_id in pair_ids for row in by_method_pair[method][pair_id]], top_fraction) for method in methods}
    rng = np.random.default_rng(args.seed)
    distributions: Dict[str, Dict[str, List[float]]] = {}
    metric_names = ["AUPRC", "AUROC", "Precision@Top10%", "Spearman"]
    comparators = [method for method in methods if method != args.anchor_method]
    for method in comparators:
        distributions[method] = {metric: [] for metric in metric_names}

    for _ in range(args.n_bootstrap):
        selected = rng.choice(pair_ids, size=len(pair_ids), replace=True).tolist()
        anchor_rows = [row for pair_id in selected for row in by_method_pair[args.anchor_method][pair_id]]
        anchor_values = metric_values(anchor_rows, top_fraction)
        for method in comparators:
            compare_rows = [row for pair_id in selected for row in by_method_pair[method][pair_id]]
            compare_values = metric_values(compare_rows, top_fraction)
            for metric in metric_names:
                left, right = anchor_values.get(metric), compare_values.get(metric)
                if left is not None and right is not None:
                    distributions[method][metric].append(float(left - right))

    summary_rows: List[Dict[str, object]] = []
    replicate_rows: List[Dict[str, object]] = []
    for comparator in comparators:
        for metric in metric_names:
            values = distributions[comparator][metric]
            low, high = percentile_ci(values, args.confidence)
            observed_delta = None
            if observed[args.anchor_method].get(metric) is not None and observed[comparator].get(metric) is not None:
                observed_delta = float(observed[args.anchor_method][metric] - observed[comparator][metric])
            p_gt_zero = float(np.mean(np.asarray(values) > 0.0)) if values else None
            summary_rows.append({
                "reference": args.reference,
                "split": args.split,
                "anchor_method": args.anchor_method,
                "comparator": comparator,
                "metric": metric,
                "anchor_observed": observed[args.anchor_method].get(metric),
                "comparator_observed": observed[comparator].get(metric),
                "observed_delta": observed_delta,
                "bootstrap_mean_delta": float(np.mean(values)) if values else None,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_probability_delta_gt_zero": p_gt_zero,
                "n_valid_bootstrap_replicates": len(values),
            })
            for replicate_index, value in enumerate(values):
                replicate_rows.append({
                    "comparator": comparator,
                    "metric": metric,
                    "replicate": replicate_index,
                    "delta_anchor_minus_comparator": value,
                })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "pair_bootstrap_comparison_summary.csv"
    write_csv(summary_path, list(summary_rows[0].keys()), summary_rows)
    write_csv(args.out_dir / "pair_bootstrap_delta_replicates.csv", list(replicate_rows[0].keys()), replicate_rows)
    report = {
        "reference": args.reference,
        "split": args.split,
        "anchor_method": args.anchor_method,
        "methods": methods,
        "n_pairs": len(pair_ids),
        "n_bootstrap": args.n_bootstrap,
        "confidence": args.confidence,
        "bootstrap_unit": "temporal pair; all tiles of each sampled pair are resampled together",
        "target_min_change_ratio": args.target_min_change_ratio,
        "top_k_percent": args.top_k_percent,
        "summary_path": str(summary_path),
    }
    (args.out_dir / "pair_bootstrap_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Reference:", args.reference)
    print("Temporal pairs:", len(pair_ids))
    print("Anchor method:", args.anchor_method)
    for row in summary_rows:
        if row["metric"] == "AUPRC":
            print("AUPRC delta vs {:<24s}: observed={:+.4f}, {:.0f}% CI=[{:+.4f}, {:+.4f}], P(delta>0)={:.3f}".format(
                str(row["comparator"]), float(row["observed_delta"]), args.confidence * 100,
                float(row["ci_low"]), float(row["ci_high"]), float(row["bootstrap_probability_delta_gt_zero"])
            ))
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
