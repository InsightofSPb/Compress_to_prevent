from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .io import read_csv_rows, write_csv_rows


def _auc_roc(scores_labels: List[tuple[float, int]]) -> float:
    pos = sum(label for _, label in scores_labels)
    neg = len(scores_labels) - pos
    if pos == 0 or neg == 0:
        return 0.0

    ranked = sorted(scores_labels, key=lambda x: x[0])
    rank_sum = 0.0
    for rank_idx, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            rank_sum += rank_idx
    return float((rank_sum - (pos * (pos + 1) / 2.0)) / (pos * neg))


def _average_precision(scores_labels: List[tuple[float, int]]) -> float:
    positives = sum(label for _, label in scores_labels)
    if positives == 0:
        return 0.0

    ranked = sorted(scores_labels, key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    ap_acc = 0.0
    for _, label in ranked:
        if label == 1:
            tp += 1
            ap_acc += tp / (tp + fp)
        else:
            fp += 1
    return float(ap_acc / positives)


def evaluate_change_metrics(tile_scores_csv: Path, labels_csv: Path, output_csv: Path) -> Dict[str, float]:
    score_rows = read_csv_rows(tile_scores_csv)
    label_rows = read_csv_rows(labels_csv)

    labels: Dict[tuple[str, int, int], int] = {}
    for row in label_rows:
        key = (row["pair_id"], int(row["tile_x"]), int(row["tile_y"]))
        labels[key] = int(row["label"])

    joined: List[tuple[float, int]] = []
    for row in score_rows:
        key = (row["pair_id"], int(row["tile_x"]), int(row["tile_y"]))
        if key not in labels:
            continue
        joined.append((float(row["tile_score"]), labels[key]))

    roc_auc = _auc_roc(joined)
    ap = _average_precision(joined)
    summary = {
        "score_type": "change_score",
        "n_tiles": float(len(joined)),
        "roc_auc": roc_auc,
        "average_precision": ap,
    }

    write_csv_rows(output_csv, ["score_type", "n_tiles", "roc_auc", "average_precision"], [summary])
    return summary
