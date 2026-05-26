from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

from .io import read_csv_rows, write_csv_rows


def _auc_roc(scores_labels: List[tuple[float, int]]) -> float:
    pos = sum(label for _, label in scores_labels)
    neg = len(scores_labels) - pos
    if pos == 0 or neg == 0:
        return 0.0
    ranked = sorted(scores_labels, key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    n = len(ranked)
    while i < n:
        j = i
        score = ranked[i][0]
        while j < n and ranked[j][0] == score:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        n_pos_tie = sum(lbl for _, lbl in ranked[i:j])
        rank_sum += avg_rank * n_pos_tie
        i = j
    return float((rank_sum - (pos * (pos + 1) / 2.0)) / (pos * neg))


def _average_precision(scores_labels: List[tuple[float, int]]) -> float:
    positives = sum(label for _, label in scores_labels)
    if positives == 0:
        return 0.0
    ranked = sorted(scores_labels, key=lambda x: x[0], reverse=True)
    tp = fp = 0
    ap_acc = 0.0
    for _, label in ranked:
        if label == 1:
            tp += 1
            ap_acc += tp / (tp + fp)
        else:
            fp += 1
    return float(ap_acc / positives)


def _precision_recall_at_pct(ranked: List[Tuple[float, int]], pct: float) -> Tuple[float, float]:
    if not ranked:
        return 0.0, 0.0
    k = max(1, math.ceil(len(ranked) * pct))
    top = ranked[:k]
    tp = sum(l for _, l in top)
    precision = tp / k
    total_pos = sum(l for _, l in ranked)
    recall = tp / total_pos if total_pos else 0.0
    return float(precision), float(recall)


def _best_f1(ranked: List[Tuple[float, int]]) -> Tuple[float, float]:
    if not ranked:
        return 0.0, 0.0
    total_pos = sum(l for _, l in ranked)
    tp = fp = 0
    best_f1 = 0.0
    best_thr = ranked[0][0]
    for score, label in ranked:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total_pos if total_pos else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if f1 >= best_f1:
            best_f1 = f1
            best_thr = score
    return float(best_f1), float(best_thr)


def _topk_hit_rate(pair_rows: Dict[str, List[Tuple[float, int]]], k: int) -> float:
    eligible = {pid: rows for pid, rows in pair_rows.items() if any(lbl == 1 for _, lbl in rows)}
    if not eligible:
        return 0.0
    hits = 0
    for rows in eligible.values():
        ranked = sorted(rows, key=lambda x: x[0], reverse=True)[:k]
        if any(label == 1 for _, label in ranked):
            hits += 1
    return float(hits / len(eligible))


def evaluate_change_metrics(tile_scores_csv: Path, labels_csv: Path, output_csv: Path) -> Dict[str, float]:
    score_rows = read_csv_rows(tile_scores_csv)
    label_rows = read_csv_rows(labels_csv)

    labels: Dict[tuple[str, int, int], int] = {}
    for row in label_rows:
        labels[(row["pair_id"], int(row["tile_x"]), int(row["tile_y"]))] = int(row["label"])

    grouped: Dict[str, List[Tuple[str, float, int]]] = {}
    score_type_by_method: Dict[str, str] = {}
    for row in score_rows:
        method = row.get("method") or row.get("score_type") or "proposed_residual"
        key = (row["pair_id"], int(row["tile_x"]), int(row["tile_y"]))
        if key not in labels:
            continue
        grouped.setdefault(method, []).append((row["pair_id"], float(row["tile_score"]), labels[key]))
        score_type_by_method[method] = row.get("score_type", "change_score")

    summaries: List[Dict[str, object]] = []
    for method, rows in grouped.items():
        scores_labels = [(s, l) for _, s, l in rows]
        ranked = sorted(scores_labels, key=lambda x: x[0], reverse=True)
        n_tiles = len(rows)
        n_pos = sum(l for _, _, l in rows)
        n_neg = n_tiles - n_pos
        pair_map: Dict[str, List[Tuple[float, int]]] = {}
        for pid, s, l in rows:
            pair_map.setdefault(pid, []).append((s, l))

        p1, _ = _precision_recall_at_pct(ranked, 0.01)
        p5, r5 = _precision_recall_at_pct(ranked, 0.05)
        p10, r10 = _precision_recall_at_pct(ranked, 0.10)
        bf1, bthr = _best_f1(ranked)

        summaries.append({
            "method": method,
            "score_type": score_type_by_method.get(method, "change_score"),
            "n_pairs": len(pair_map),
            "n_tiles": n_tiles,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "roc_auc": _auc_roc(scores_labels),
            "average_precision": _average_precision(scores_labels),
            "precision_at_1pct": p1,
            "precision_at_5pct": p5,
            "precision_at_10pct": p10,
            "recall_at_5pct": r5,
            "recall_at_10pct": r10,
            "best_f1": bf1,
            "best_f1_threshold": bthr,
            "topk_hit_rate_5": _topk_hit_rate(pair_map, 5),
            "topk_hit_rate_10": _topk_hit_rate(pair_map, 10),
        })

    fields = ["method", "score_type", "n_pairs", "n_tiles", "n_pos", "n_neg", "roc_auc", "average_precision", "precision_at_1pct", "precision_at_5pct", "precision_at_10pct", "recall_at_5pct", "recall_at_10pct", "best_f1", "best_f1_threshold", "topk_hit_rate_5", "topk_hit_rate_10"]
    write_csv_rows(output_csv, fields, summaries)
    return summaries[0] if summaries else {"n_tiles": 0.0, "roc_auc": 0.0, "average_precision": 0.0}
