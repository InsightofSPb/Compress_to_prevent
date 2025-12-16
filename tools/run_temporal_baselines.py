import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

META_COLUMNS: Sequence[str] = (
    "facade_id",
    "step_idx",
    "target_step",
    "year_prev",
    "year_next",
    "year_t",
    "year_t1",
    "quality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal baselines for facade change prediction")
    parser.add_argument("--features-path", required=True, type=Path, help="Path to timeseries_features.parquet (CSV allowed)")
    parser.add_argument("--index-path", required=True, type=Path, help="Path to timeseries_index.csv")
    parser.add_argument("--target-column", default="delta_main_mean", type=str, help="Column name to predict on the t+1 step")
    parser.add_argument(
        "--output-dir", default=Path("outputs/temporal_baseline"), type=Path, help="Directory to save metrics and prediction tables"
    )
    return parser.parse_args()


def _coerce_value(value: str):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_table(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed = {k: _coerce_value(v) for k, v in row.items()}
            rows.append(parsed)
    return rows


def ensure_step_idx(rows: List[Dict[str, object]], index_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if rows and "step_idx" in rows[0]:
        return rows
    lookup: Dict[Tuple[object, object, object], int] = {}
    for idx_row in index_rows:
        key = (idx_row.get("facade_id"), idx_row.get("year_prev"), idx_row.get("year_next"))
        lookup[key] = int(idx_row.get("step_idx", len(lookup)))
    for row in rows:
        key = (row.get("facade_id"), row.get("year_prev"), row.get("year_next"))
        if key not in lookup:
            raise ValueError(f"Cannot infer step_idx for row {row}")
        row["step_idx"] = lookup[key]
    return rows


def merge_index(features: List[Dict[str, object]], index_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    index_by_key: Dict[Tuple[object, int], Dict[str, object]] = {}
    for idx_row in index_rows:
        facade = idx_row.get("facade_id")
        step = int(idx_row.get("step_idx", 0))
        index_by_key[(facade, step)] = idx_row

    merged: List[Dict[str, object]] = []
    for row in features:
        key = (row.get("facade_id"), int(row["step_idx"]))
        merged_row = dict(row)
        merged_row.update(index_by_key.get(key, {}))
        merged.append(merged_row)
    return merged


def pick_feature_columns(rows: List[Dict[str, object]], target_column: str) -> List[str]:
    if not rows:
        raise ValueError("Empty features table")
    feature_cols: List[str] = []
    for col, value in rows[0].items():
        if col == target_column or col in META_COLUMNS:
            continue
        if isinstance(value, (int, float)):
            feature_cols.append(col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found")
    return feature_cols


def build_supervised_table(rows: List[Dict[str, object]], target_column: str, feature_cols: List[str]) -> List[Dict[str, object]]:
    rows_sorted = sorted(rows, key=lambda r: (r.get("facade_id"), r.get("step_idx")))
    supervised: List[Dict[str, object]] = []
    from itertools import groupby

    for facade_id, group in groupby(rows_sorted, key=lambda r: r.get("facade_id")):
        group_list = list(group)
        for idx in range(len(group_list) - 1):
            src = group_list[idx]
            tgt = group_list[idx + 1]
            entry: Dict[str, object] = {
                "facade_id": facade_id,
                "step_idx": int(src["step_idx"]),
                "target_step": int(tgt.get("step_idx", src["step_idx"] + 1)),
                "year_t": tgt.get("year_prev", src.get("year_prev")),
                "year_t1": tgt.get("year_next", tgt.get("year")),
                "y_true": float(tgt[target_column]),
                "y_prev": float(src[target_column]),
            }
            for col in feature_cols:
                entry[col] = float(src[col]) if src[col] is not None else None
            supervised.append(entry)
    if not supervised:
        raise ValueError("Not enough steps to build supervised samples")
    return supervised


def split_by_facade(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    parts: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    from itertools import groupby

    rows_sorted = sorted(rows, key=lambda r: (r.get("facade_id"), r.get("target_step")))
    for _, group in groupby(rows_sorted, key=lambda r: r.get("facade_id")):
        g = list(group)
        if len(g) == 1:
            parts["test"].append(g[0])
        elif len(g) == 2:
            parts["val"].append(g[0])
            parts["test"].append(g[1])
        else:
            parts["test"].append(g[-1])
            parts["val"].append(g[-2])
            parts["train"].extend(g[:-2])
    return parts


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def rankdata(values: List[float]) -> List[float]:
    sorted_vals = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1][1] == sorted_vals[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearmanr(y_true: List[float], y_pred: List[float]) -> float:
    if len(y_true) <= 1:
        return float("nan")
    rank_true = rankdata(y_true)
    rank_pred = rankdata(y_pred)
    mu_true = mean(rank_true)
    mu_pred = mean(rank_pred)
    num = sum((a - mu_true) * (b - mu_pred) for a, b in zip(rank_true, rank_pred))
    den = math.sqrt(sum((a - mu_true) ** 2 for a in rank_true) * sum((b - mu_pred) ** 2 for b in rank_pred))
    return num / den if den != 0 else float("nan")


def compute_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    mae = mean([abs(a - b) for a, b in zip(y_true, y_pred)])
    rmse = math.sqrt(mean([(a - b) ** 2 for a, b in zip(y_true, y_pred)]))
    rho = spearmanr(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "spearman": rho}


def add_bias_and_scale(rows: List[Dict[str, object]], feature_cols: List[str], stats: Dict[str, Tuple[float, float]]):
    matrix: List[List[float]] = []
    for row in rows:
        vector: List[float] = [1.0]
        for col in feature_cols:
            mean_val, std_val = stats[col]
            val = row.get(col)
            if val is None:
                val = mean_val
            if std_val > 0:
                vector.append((val - mean_val) / std_val)
            else:
                vector.append(val - mean_val)
        matrix.append(vector)
    return matrix


def invert_matrix(mat: List[List[float]]) -> List[List[float]]:
    n = len(mat)
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(mat)]
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if abs(aug[r][col]) > 1e-12:
                pivot_row = r
                break
        if pivot_row is None:
            raise ValueError("Matrix is singular")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [rv - factor * cv for rv, cv in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for k in range(len(B)):
            for j in range(len(B[0])):
                result[i][j] += A[i][k] * B[k][j]
    return result


def matvec(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a * b for a, b in zip(row, v)) for row in A]


class RidgeModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.feature_stats: Dict[str, Tuple[float, float]] = {}
        self.weights: List[float] = []
        self.feature_cols: List[str] = []

    def fit(self, rows: List[Dict[str, object]], feature_cols: List[str]):
        self.feature_cols = feature_cols
        stats: Dict[str, Tuple[float, float]] = {}
        for col in feature_cols:
            vals = [float(r[col]) for r in rows if r[col] is not None]
            stats[col] = (mean(vals), std(vals))
        self.feature_stats = stats

        X = add_bias_and_scale(rows, feature_cols, stats)
        y = [float(r["y_true"]) for r in rows]

        Xt = list(map(list, zip(*X)))
        XtX = matmul(Xt, X)
        for i in range(len(XtX)):
            XtX[i][i] += self.alpha
        XtY = matvec(Xt, y)
        XtY_col = [[val] for val in XtY]
        inv = invert_matrix(XtX)
        weights_matrix = matmul(inv, XtY_col)
        self.weights = [w[0] for w in weights_matrix]

    def predict(self, rows: List[Dict[str, object]]) -> List[float]:
        X = add_bias_and_scale(rows, self.feature_cols, self.feature_stats)
        preds: List[float] = []
        for vec in X:
            preds.append(sum(w * x for w, x in zip(self.weights, vec)))
        return preds


def evaluate_models(splits: Dict[str, List[Dict[str, object]]], ridge_model: RidgeModel, feature_cols: List[str], global_mean: float):
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split_name, rows in splits.items():
        if not rows:
            metrics[split_name] = {}
            continue
        y_true = [float(r["y_true"]) for r in rows]
        preds_ridge = ridge_model.predict(rows)
        preds_persist = [float(r["y_prev"]) for r in rows]
        preds_mean = [global_mean for _ in rows]
        metrics[split_name] = {
            "ridge": compute_metrics(y_true, preds_ridge),
            "persistence": compute_metrics(y_true, preds_persist),
            "global_mean": compute_metrics(y_true, preds_mean),
        }
    return metrics


def save_predictions(rows: List[Dict[str, object]], ridge_model: RidgeModel, feature_cols: List[str], global_mean: float, output_dir: Path):
    y_true = [float(r["y_true"]) for r in rows]
    preds_ridge = ridge_model.predict(rows)
    preds_persist = [float(r["y_prev"]) for r in rows]
    preds_mean = [global_mean for _ in rows]

    output_path = output_dir / "temporal_predictions_test.csv"
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["facade_id", "year_t", "year_t1", "y_true", "y_pred_ridge", "y_pred_persist", "y_pred_mean", "abs_residual"])
        for row, y, pr, pp, pm in zip(rows, y_true, preds_ridge, preds_persist, preds_mean):
            writer.writerow([
                row.get("facade_id"),
                row.get("year_t"),
                row.get("year_t1"),
                y,
                pr,
                pp,
                pm,
                abs(y - pr),
            ])

    leaderboard_path = output_dir / "temporal_leaderboard_test.csv"
    by_facade: Dict[object, List[Tuple[float, float]]] = {}
    for row, pred in zip(rows, preds_ridge):
        by_facade.setdefault(row.get("facade_id"), []).append((float(row["y_true"]), pred))

    leaderboard_rows: List[List[object]] = []
    for facade_id, vals in by_facade.items():
        abs_residuals = [abs(y - p) for y, p in vals]
        leaderboard_rows.append([
            facade_id,
            mean(abs_residuals),
            max(abs_residuals),
            mean([y for y, _ in vals]),
            mean([p for _, p in vals]),
            len(vals),
        ])
    leaderboard_rows.sort(key=lambda r: r[1], reverse=True)

    with leaderboard_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "facade_id",
            "mean_abs_residual",
            "max_abs_residual",
            "mean_y_true",
            "mean_y_pred_ridge",
            "n_samples",
            "risk_rank",
            "risk_label",
        ])
        total = len(leaderboard_rows)
        for rank, row in enumerate(leaderboard_rows, start=1):
            if total <= 3:
                label = "HIGH" if rank == 1 else "MED"
            else:
                tertile = max(1, total // 3)
                if rank <= tertile:
                    label = "HIGH"
                elif rank <= tertile * 2:
                    label = "MED"
                else:
                    label = "LOW"
            writer.writerow(row + [rank, label])


if __name__ == "__main__":
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = read_table(args.features_path)
    index_rows = read_table(args.index_path)
    feature_rows = ensure_step_idx(feature_rows, index_rows)
    merged_rows = merge_index(feature_rows, index_rows)

    feature_cols = pick_feature_columns(merged_rows, args.target_column)
    supervised = build_supervised_table(merged_rows, args.target_column, feature_cols)
    splits = split_by_facade(supervised)

    if not splits["train"]:
        raise ValueError("Train split is empty; need at least 3 transitions per facade")

    ridge = RidgeModel(alpha=1.0)
    ridge.fit(splits["train"], feature_cols)
    global_mean = mean([float(r["y_true"]) for r in splits["train"]])

    metrics = evaluate_models(splits, ridge, feature_cols, global_mean)
    metrics_path = output_dir / "temporal_eval_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    if not splits["test"]:
        raise ValueError("Test split is empty; cannot produce predictions")

    save_predictions(splits["test"], ridge, feature_cols, global_mean, output_dir)
