#!/usr/bin/env python3
"""Evaluate tile-level temporal heatmap scores against aligned GT references.

For each tile score, the reference target is positive when the fraction of
changed valid pixels in the corresponding GT reference tile is at least a
configurable threshold. Continuous ranking metrics (AUROC and AUPRC) are
reported for validation and test splits. A binary score threshold is selected
on validation by maximum F1 and then applied unchanged to test.

The report includes prevalence/random-ranking baselines, all-positive binary
baselines, lift over these baselines, and correlations between the raw score
and the continuous changed-pixel fraction. This prevents misleading
interpretation when maximum-F1 thresholds mark almost every tile as changed.
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
    parser = argparse.ArgumentParser(description="Evaluate temporal tile scores against GT change references.")
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--references-csv", type=Path, required=True)
    parser.add_argument("--reference", choices=(
        "inspection_relevant_change", "damage_or_repair_change", "damage_type_change",
        "intervention_or_content_change", "any_semantic_change", "damage_presence_change",
    ), default="inspection_relevant_change")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-min-change-ratio", type=float, default=0.01,
                        help="A tile is positive if at least this fraction of valid pixels changes.")
    parser.add_argument("--top-k-percent", type=str, default="5,10,20",
                        help="Comma-separated top-score fractions for Precision@Top-K metrics.")
    parser.add_argument("--invalid-label", type=int, default=255)
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--report-split", default="test")
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


def reference_column(name: str) -> str:
    return name + "_path"


def read_reference(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError("Could not read reference map: {}".format(path))
    if image.ndim == 3:
        image = image[..., 0]
    return image


def safe_ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Optional[float]]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception as exc:
        raise RuntimeError("This evaluator requires scikit-learn: pip install scikit-learn") from exc
    if len(np.unique(y_true)) < 2:
        return {"AUROC": None, "AUPRC": None}
    return {
        "AUROC": float(roc_auc_score(y_true, y_score)),
        "AUPRC": float(average_precision_score(y_true, y_score)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling, implemented without scipy."""
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


def _safe_corr(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def continuous_agreement(target_ratio: np.ndarray, score: np.ndarray) -> Dict[str, Optional[float]]:
    return {
        "Pearson_target_ratio": _safe_corr(target_ratio, score),
        "Spearman_target_ratio": _safe_corr(_average_ranks(target_ratio), _average_ranks(score)),
        "mean_target_change_ratio": float(target_ratio.mean()) if len(target_ratio) else None,
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int]:
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    predicted_positive_ratio = float(y_pred.mean()) if len(y_pred) else 0.0
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": float(precision), "recall": float(recall),
        "F1": float(f1), "IoU": float(iou),
        "predicted_positive_ratio": predicted_positive_ratio,
    }


def select_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float | int]]:
    candidates = np.unique(y_score)
    if len(candidates) == 0:
        raise ValueError("Cannot select threshold from empty score sequence")
    best_threshold = float(candidates[0])
    best_metrics = binary_metrics(y_true, (y_score >= best_threshold).astype(np.uint8))
    for threshold in candidates[1:]:
        metrics = binary_metrics(y_true, (y_score >= threshold).astype(np.uint8))
        if float(metrics["F1"]) > float(best_metrics["F1"]):
            best_threshold, best_metrics = float(threshold), metrics
    return best_threshold, best_metrics


def precision_at_top_k(y_true: np.ndarray, y_score: np.ndarray, fractions: Sequence[float]) -> Dict[str, float]:
    order = np.argsort(-y_score)
    output: Dict[str, float] = {}
    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    for fraction in fractions:
        label = int(round(fraction * 100))
        n = max(1, int(np.ceil(len(order) * fraction)))
        precision = float(y_true[order[:n]].mean())
        output["Precision@Top{}%".format(label)] = precision
        output["Precision@Top{}%_lift".format(label)] = precision / prevalence if prevalence > 0 else 0.0
    return output


def baseline_and_lift_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    binary: Dict[str, float | int],
    ranking: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    all_positive = binary_metrics(y_true, np.ones_like(y_true, dtype=np.uint8))
    auprc = ranking.get("AUPRC")
    return {
        "random_AUPRC_baseline": prevalence,
        "AUPRC_delta_over_random": (float(auprc) - prevalence) if auprc is not None else None,
        "AUPRC_lift_over_random": (float(auprc) / prevalence) if auprc is not None and prevalence > 0 else None,
        "all_positive_F1_baseline": float(all_positive["F1"]),
        "all_positive_IoU_baseline": float(all_positive["IoU"]),
        "F1_delta_over_all_positive": float(binary["F1"]) - float(all_positive["F1"]),
        "IoU_delta_over_all_positive": float(binary["IoU"]) - float(all_positive["IoU"]),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target_min_change_ratio <= 1.0:
        raise ValueError("target-min-change-ratio must be in [0, 1]")
    fractions = [float(item.strip()) / 100.0 for item in args.top_k_percent.split(",") if item.strip()]
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("top-k-percent values must be in (0, 100]")

    score_rows = read_csv(args.scores_csv)
    reference_rows = read_csv(args.references_csv)
    refs = {row["pair_id"]: row for row in reference_rows}
    map_cache: Dict[str, np.ndarray] = {}
    evaluation_rows: List[Dict[str, object]] = []
    ref_col = reference_column(args.reference)

    for row in score_rows:
        pair_id = row["pair_id"]
        if pair_id not in refs:
            raise KeyError("Score pair not present in reference manifest: {}".format(pair_id))
        reference_path = refs[pair_id].get(ref_col, "")
        if not reference_path:
            raise KeyError("Reference column {} absent for pair {}".format(ref_col, pair_id))
        if reference_path not in map_cache:
            map_cache[reference_path] = read_reference(Path(reference_path))
        reference = map_cache[reference_path]
        tile_size = int(float(row["tile_size"]))
        tile_x = int(float(row["tile_x"]))
        tile_y = int(float(row["tile_y"]))
        y0, x0 = tile_y * tile_size, tile_x * tile_size
        tile = reference[y0:min(y0 + tile_size, reference.shape[0]), x0:min(x0 + tile_size, reference.shape[1])]
        valid = tile != args.invalid_label
        valid_count = int(valid.sum())
        if valid_count == 0:
            continue
        change_ratio = float((tile[valid] == 1).mean())
        evaluation_rows.append({
            "pair_id": pair_id,
            "facade_id": row.get("facade_id", ""),
            "split": row["split"],
            "method": row["method"],
            "tile_x": tile_x,
            "tile_y": tile_y,
            "tile_size": tile_size,
            "tile_score": float(row["tile_score"]),
            "target_change_ratio": change_ratio,
            "target_positive": int(change_ratio >= args.target_min_change_ratio),
            "valid_pixel_count": valid_count,
        })

    if not evaluation_rows:
        raise ValueError("No evaluation tiles created")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "tile_scores_with_targets.csv",
        ["pair_id", "facade_id", "split", "method", "tile_x", "tile_y", "tile_size", "tile_score",
         "target_change_ratio", "target_positive", "valid_pixel_count"],
        evaluation_rows,
    )

    summary: Dict[str, object] = {
        "scores_csv": str(args.scores_csv),
        "references_csv": str(args.references_csv),
        "reference": args.reference,
        "target_min_change_ratio": args.target_min_change_ratio,
        "selection_split": args.selection_split,
        "report_split": args.report_split,
        "interpretation": {
            "primary_ranking_metrics": ["AUPRC", "AUROC", "Precision@TopK", "Spearman_target_ratio"],
            "threshold_metrics": "F1/IoU use a threshold selected on validation only; compare against all-positive baselines.",
        },
        "methods": {},
    }
    flat_rows: List[Dict[str, object]] = []
    methods = sorted({str(row["method"]) for row in evaluation_rows})
    for method in methods:
        method_rows = [row for row in evaluation_rows if row["method"] == method]
        selection = [row for row in method_rows if row["split"] == args.selection_split]
        report = [row for row in method_rows if row["split"] == args.report_split]
        if not selection or not report:
            raise ValueError("Method {} lacks rows for val/test threshold evaluation".format(method))
        y_val = np.asarray([row["target_positive"] for row in selection], dtype=np.uint8)
        s_val = np.asarray([row["tile_score"] for row in selection], dtype=np.float64)
        r_val = np.asarray([row["target_change_ratio"] for row in selection], dtype=np.float64)
        y_test = np.asarray([row["target_positive"] for row in report], dtype=np.uint8)
        s_test = np.asarray([row["tile_score"] for row in report], dtype=np.float64)
        r_test = np.asarray([row["target_change_ratio"] for row in report], dtype=np.float64)
        threshold, val_binary = select_f1_threshold(y_val, s_val)
        test_binary = binary_metrics(y_test, (s_test >= threshold).astype(np.uint8))
        val_rank = safe_ranking_metrics(y_val, s_val)
        test_rank = safe_ranking_metrics(y_test, s_test)
        val_values = {
            "n_tiles": len(selection), "positive_tiles": int(y_val.sum()), "positive_ratio": float(y_val.mean()),
            **val_rank, **val_binary, **precision_at_top_k(y_val, s_val, fractions),
            **continuous_agreement(r_val, s_val), **baseline_and_lift_metrics(y_val, s_val, val_binary, val_rank),
        }
        test_values = {
            "n_tiles": len(report), "positive_tiles": int(y_test.sum()), "positive_ratio": float(y_test.mean()),
            **test_rank, **test_binary, **precision_at_top_k(y_test, s_test, fractions),
            **continuous_agreement(r_test, s_test), **baseline_and_lift_metrics(y_test, s_test, test_binary, test_rank),
        }
        method_summary = {
            "threshold_selected_on_val": threshold,
            "val": val_values,
            "test": test_values,
        }
        summary["methods"][method] = method_summary
        for split_name, values in ((args.selection_split, val_values), (args.report_split, test_values)):
            flat_rows.append({
                "method": method, "reference": args.reference, "split": split_name,
                "target_min_change_ratio": args.target_min_change_ratio,
                "threshold_selected_on_val": threshold,
                **values,
            })

    (args.out_dir / "evaluation_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [
        "method", "reference", "split", "target_min_change_ratio", "threshold_selected_on_val",
        "n_tiles", "positive_tiles", "positive_ratio", "mean_target_change_ratio",
        "AUROC", "AUPRC", "random_AUPRC_baseline", "AUPRC_delta_over_random", "AUPRC_lift_over_random",
        "Pearson_target_ratio", "Spearman_target_ratio", "TP", "FP", "FN", "TN",
        "predicted_positive_ratio", "precision", "recall", "F1", "IoU",
        "all_positive_F1_baseline", "all_positive_IoU_baseline",
        "F1_delta_over_all_positive", "IoU_delta_over_all_positive",
    ]
    for value in fractions:
        label = int(round(value * 100))
        fields.extend(["Precision@Top{}%".format(label), "Precision@Top{}%_lift".format(label)])
    write_csv(args.out_dir / "evaluation_summary.csv", fields, flat_rows)
    print("Reference: {}".format(args.reference))
    print("Tile target positive threshold: {:.4f}".format(args.target_min_change_ratio))
    for row in flat_rows:
        print(
            "{method:22s} {split:5s} tiles={n_tiles:6d} pos={positive_ratio:.4f} "
            "AUPRC={AUPRC} lift={AUPRC_lift_over_random} AUROC={AUROC} "
            "F1={F1:.4f} (all+={all_positive_F1_baseline:.4f}) "
            "Spearman={Spearman_target_ratio}".format(**row)
        )
    print("Report: {}".format(args.out_dir / "evaluation_report.json"))
    print("Summary: {}".format(args.out_dir / "evaluation_summary.csv"))


if __name__ == "__main__":
    main()
