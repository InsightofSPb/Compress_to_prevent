import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import pandas as pd

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

DEFAULT_ALPHAS: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal baselines for facade change prediction")
    parser.add_argument("--features-path", required=True, type=Path, help="Path to timeseries_features.parquet (CSV allowed)")
    parser.add_argument("--index-path", required=True, type=Path, help="Path to timeseries_index.csv")
    parser.add_argument("--target-column", default="delta_main_mean", type=str, help="Column name to predict on the t+1 step")
    parser.add_argument(
        "--output-dir", default=Path("outputs/temporal_baseline"), type=Path, help="Directory to save metrics and prediction tables"
    )

    # New: split & model selection knobs (safe defaults preserve old behavior)
    parser.add_argument(
        "--split-mode",
        default="within_facade",
        choices=("within_facade", "facade_disjoint"),
        help="within_facade: last step per facade goes to test (old behavior). "
             "facade_disjoint: facades are split into train/val/test with no overlap.",
    )
    parser.add_argument("--train-frac", default=0.7, type=float, help="Used only for split-mode=facade_disjoint")
    parser.add_argument("--val-frac", default=0.15, type=float, help="Used only for split-mode=facade_disjoint")
    parser.add_argument("--split-seed", default=13, type=int, help="Seed for split-mode=facade_disjoint")

    parser.add_argument(
        "--scaler",
        default="robust",
        choices=("standard", "robust"),
        help="Feature scaling. robust uses median/IQR and is usually safer for outliers.",
    )
    parser.add_argument(
        "--alphas",
        default=",".join(str(a) for a in DEFAULT_ALPHAS),
        type=str,
        help="Comma-separated list of ridge alphas to try (used by internal CV).",
    )
    parser.add_argument("--cv-folds", default=5, type=int, help="Number of group folds (by facade_id) for alpha selection")
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
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        df = df.where(pd.notnull(df), None)
        return df.to_dict("records")
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed = {k: _coerce_value(v) for k, v in row.items()}
            rows.append(parsed)
    return rows


def ensure_step_idx(rows: List[Dict[str, object]], index_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    # Be robust: treat step_idx as "present" only if it's actually filled.
    if rows and "step_idx" in rows[0] and rows[0].get("step_idx") is not None:
        return rows

    lookup: Dict[Tuple[object, object, object], int] = {}
    for idx_row in index_rows:
        if "step_idx" not in idx_row or idx_row.get("step_idx") is None:
            continue
        key = (idx_row.get("facade_id"), idx_row.get("year_prev"), idx_row.get("year_next"))
        lookup[key] = int(idx_row.get("step_idx", len(lookup)))

    if not lookup:
        grouped: Dict[object, List[Dict[str, object]]] = {}
        for row in rows:
            grouped.setdefault(row.get("facade_id"), []).append(row)
        for facade_id, group in grouped.items():
            group_sorted = sorted(group, key=lambda r: (r.get("year_prev"), r.get("year_next")))
            for idx, row in enumerate(group_sorted):
                lookup[(facade_id, row.get("year_prev"), row.get("year_next"))] = idx

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


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def pick_feature_columns(rows: List[Dict[str, object]], target_column: str) -> List[str]:
    if not rows:
        raise ValueError("Empty features table")

    # Old version looked at only the first row. Here we scan and keep columns that are numeric at least somewhere.
    candidate_cols = [c for c in rows[0].keys() if c != target_column and c not in META_COLUMNS]
    feature_cols: List[str] = []
    for col in candidate_cols:
        seen_numeric = False
        for r in rows[: min(200, len(rows))]:
            v = r.get(col)
            if v is None:
                continue
            if _is_number(v):
                seen_numeric = True
                break
        if seen_numeric:
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No numeric feature columns found")
    return feature_cols


def _to_float_or_none(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _quality_score(q: object) -> Optional[float]:
    if q is None:
        return None
    s = str(q).strip().lower()
    if s == "strong":
        return 1.0
    if s == "weak":
        return 0.0
    return None


def build_supervised_table(rows: List[Dict[str, object]], target_column: str, feature_cols: List[str]) -> List[Dict[str, object]]:
    """
    Build samples:
      X = features at step t (src)
      y_true = target at step t+1 (tgt)
    Also adds:
      - dt_prev: year_next - year_prev for src step
      - quality_score: {strong=1, weak=0, else None} for src step
    """
    rows_sorted = sorted(rows, key=lambda r: (r.get("facade_id"), r.get("step_idx")))
    supervised: List[Dict[str, object]] = []
    from itertools import groupby

    for facade_id, group in groupby(rows_sorted, key=lambda r: r.get("facade_id")):
        group_list = list(group)
        for idx in range(len(group_list) - 1):
            src = group_list[idx]
            tgt = group_list[idx + 1]

            y_tgt = _to_float_or_none(tgt.get(target_column))
            y_src = _to_float_or_none(src.get(target_column))
            if y_tgt is None or y_src is None:
                continue

            src_year_prev = _to_float_or_none(src.get("year_prev"))
            src_year_next = _to_float_or_none(src.get("year_next"))
            dt_prev = None
            if src_year_prev is not None and src_year_next is not None:
                dt_prev = src_year_next - src_year_prev

            entry: Dict[str, object] = {
                "facade_id": facade_id,
                "step_idx": int(src["step_idx"]),
                "target_step": int(tgt.get("step_idx", int(src["step_idx"]) + 1)),
                "year_t": tgt.get("year_prev", src.get("year_prev")),
                "year_t1": tgt.get("year_next", tgt.get("year")),
                "y_true": float(y_tgt),
                "y_prev": float(y_src),
                # engineered:
                "dt_prev": dt_prev,
                "quality_score": _quality_score(src.get("quality")),
            }
            for col in feature_cols:
                entry[col] = _to_float_or_none(src.get(col))
            supervised.append(entry)

    if not supervised:
        raise ValueError("Not enough steps (or missing targets) to build supervised samples")
    return supervised


def split_within_facade(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    """Old behavior: per facade, last sample -> test, previous -> val, rest -> train."""
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


def split_facade_disjoint(
    rows: List[Dict[str, object]], train_frac: float, val_frac: float, seed: int
) -> Dict[str, List[Dict[str, object]]]:
    """Facades do not overlap across splits."""
    facades = sorted({r.get("facade_id") for r in rows})
    rng = random.Random(seed)
    rng.shuffle(facades)

    n = len(facades)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(0, int(round(val_frac * n)))
    # ensure at least one test facade when possible
    if n_train + n_val >= n and n > 1:
        n_val = max(0, n - n_train - 1)

    train_set = set(facades[:n_train])
    val_set = set(facades[n_train : n_train + n_val])
    test_set = set(facades[n_train + n_val :])

    parts: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for r in rows:
        fid = r.get("facade_id")
        if fid in train_set:
            parts["train"].append(r)
        elif fid in val_set:
            parts["val"].append(r)
        else:
            parts["test"].append(r)
    return parts


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def median(values: List[float]) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    m = len(vals) // 2
    if len(vals) % 2 == 1:
        return vals[m]
    return 0.5 * (vals[m - 1] + vals[m])


def quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    if q <= 0:
        return vals[0]
    if q >= 1:
        return vals[-1]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def iqr(values: List[float]) -> float:
    return quantile(values, 0.75) - quantile(values, 0.25)


def rankdata(values: List[float]) -> List[float]:
    sorted_vals = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_vals):
        j = i
        # treat exact ties only (OK for our use; avoids extra deps)
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
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    mae = mean([abs(a - b) for a, b in zip(y_true, y_pred)])
    rmse = math.sqrt(mean([(a - b) ** 2 for a, b in zip(y_true, y_pred)]))
    rho = spearmanr(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "spearman": rho}


def compute_feature_stats(
    rows: List[Dict[str, object]],
    feature_cols: List[str],
    scaler: str,
) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for col in feature_cols:
        vals = [float(r[col]) for r in rows if r.get(col) is not None]
        if not vals:
            stats[col] = (0.0, 1.0)
            continue
        if scaler == "standard":
            mu = mean(vals)
            sd = std(vals)
            stats[col] = (mu, sd if sd > 0 else 1.0)
        elif scaler == "robust":
            med = median(vals)
            s = iqr(vals)
            # iqr can be 0 if constant; keep scale=1 to avoid div-by-zero
            stats[col] = (med, s if s > 0 else 1.0)
        else:
            raise ValueError(f"Unknown scaler: {scaler}")
    return stats


def add_bias_and_scale(rows: List[Dict[str, object]], feature_cols: List[str], stats: Dict[str, Tuple[float, float]]):
    matrix: List[List[float]] = []
    for row in rows:
        vector: List[float] = [1.0]
        for col in feature_cols:
            center, scale = stats[col]
            val = row.get(col)
            if val is None:
                val = center
            vector.append((float(val) - center) / scale if scale > 0 else float(val) - center)
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
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(len(B[0])):
                result[i][j] += aik * B[k][j]
    return result


def matvec(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a * b for a, b in zip(row, v)) for row in A]


class RidgeModel:
    def __init__(self, alpha: float = 1.0, scaler: str = "robust"):
        self.alpha = float(alpha)
        self.scaler = scaler
        self.feature_stats: Dict[str, Tuple[float, float]] = {}
        self.weights: List[float] = []
        self.feature_cols: List[str] = []

    def fit(self, rows: List[Dict[str, object]], feature_cols: List[str]):
        self.feature_cols = feature_cols
        stats = compute_feature_stats(rows, feature_cols, self.scaler)
        self.feature_stats = stats

        X = add_bias_and_scale(rows, feature_cols, stats)
        y = [float(r["y_true"]) for r in rows]

        Xt = list(map(list, zip(*X)))
        XtX = matmul(Xt, X)
        # IMPORTANT: do NOT regularize bias term
        for i in range(1, len(XtX)):
            XtX[i][i] += self.alpha

        XtY = matvec(Xt, y)
        XtY_col = [[val] for val in XtY]
        inv = invert_matrix(XtX)
        weights_matrix = matmul(inv, XtY_col)
        self.weights = [w[0] for w in weights_matrix]

    def predict(self, rows: List[Dict[str, object]]) -> List[float]:
        X = add_bias_and_scale(rows, self.feature_cols, self.feature_stats)
        return [sum(w * x for w, x in zip(self.weights, vec)) for vec in X]


class PersistenceCalibrator:
    """Fits y ≈ a*y_prev + b on train and applies it to persistence predictions."""
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.fitted = False

    def fit(self, rows: List[Dict[str, object]]):
        xs = [float(r["y_prev"]) for r in rows]
        ys = [float(r["y_true"]) for r in rows]
        if not xs:
            self.a, self.b, self.fitted = 1.0, 0.0, False
            return
        mx = mean(xs)
        my = mean(ys)
        varx = mean([(x - mx) ** 2 for x in xs])
        if varx <= 1e-12:
            self.a = 0.0
            self.b = my
        else:
            cov = mean([(x - mx) * (y - my) for x, y in zip(xs, ys)])
            self.a = cov / varx
            self.b = my - self.a * mx
        self.fitted = True

    def predict(self, rows: List[Dict[str, object]]) -> List[float]:
        return [self.a * float(r["y_prev"]) + self.b for r in rows]


def make_group_folds(rows: List[Dict[str, object]], n_folds: int, seed: int = 13) -> List[Tuple[List[int], List[int]]]:
    """
    Deterministic GroupKFold-like split by facade_id.
    Returns list of (train_indices, val_indices).
    """
    if n_folds < 2:
        return [([i for i in range(len(rows))], [])]

    # Group -> row indices
    groups: Dict[object, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get("facade_id"), []).append(i)

    facades = sorted(groups.keys(), key=lambda x: str(x))
    rng = random.Random(seed)
    rng.shuffle(facades)

    folds: List[List[object]] = [[] for _ in range(n_folds)]
    for i, fid in enumerate(facades):
        folds[i % n_folds].append(fid)

    splits: List[Tuple[List[int], List[int]]] = []
    for k in range(n_folds):
        val_facades = set(folds[k])
        tr_idx: List[int] = []
        va_idx: List[int] = []
        for fid, idxs in groups.items():
            if fid in val_facades:
                va_idx.extend(idxs)
            else:
                tr_idx.extend(idxs)
        if va_idx:
            splits.append((tr_idx, va_idx))
    return splits


def tune_ridge_alpha(
    train_rows: List[Dict[str, object]],
    feature_cols: List[str],
    alphas: Sequence[float],
    scaler: str,
    n_folds: int,
    seed: int,
) -> Tuple[float, Dict[str, float]]:
    folds = make_group_folds(train_rows, n_folds=n_folds, seed=seed)
    # If too few facades -> folds may collapse. Fall back to alpha=1.
    if not folds:
        return 1.0, {}

    alpha_scores: Dict[float, float] = {}
    for a in alphas:
        fold_maes: List[float] = []
        for tr_idx, va_idx in folds:
            tr = [train_rows[i] for i in tr_idx]
            va = [train_rows[i] for i in va_idx]
            if not tr or not va:
                continue
            m = RidgeModel(alpha=float(a), scaler=scaler)
            m.fit(tr, feature_cols)
            y_true = [float(r["y_true"]) for r in va]
            y_pred = m.predict(va)
            fold_maes.append(compute_metrics(y_true, y_pred)["mae"])
        alpha_scores[float(a)] = mean(fold_maes) if fold_maes else float("inf")

    best_alpha = min(alpha_scores.items(), key=lambda kv: kv[1])[0]
    return best_alpha, {str(k): v for k, v in sorted(alpha_scores.items(), key=lambda kv: kv[0])}


def evaluate_models(
    splits: Dict[str, List[Dict[str, object]]],
    ridge_model: RidgeModel,
    persist_cal: PersistenceCalibrator,
    global_mean_value: float,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split_name, rows in splits.items():
        if not rows:
            metrics[split_name] = {}
            continue
        y_true = [float(r["y_true"]) for r in rows]
        preds_ridge = ridge_model.predict(rows)
        preds_persist = [float(r["y_prev"]) for r in rows]
        preds_persist_cal = persist_cal.predict(rows)
        preds_mean = [global_mean_value for _ in rows]
        metrics[split_name] = {
            "ridge": compute_metrics(y_true, preds_ridge),
            "persistence": compute_metrics(y_true, preds_persist),
            "persistence_calibrated": compute_metrics(y_true, preds_persist_cal),
            "global_mean": compute_metrics(y_true, preds_mean),
        }
    return metrics


def save_predictions(
    rows: List[Dict[str, object]],
    ridge_model: RidgeModel,
    persist_cal: PersistenceCalibrator,
    global_mean_value: float,
    output_dir: Path,
):
    y_true = [float(r["y_true"]) for r in rows]
    preds_ridge = ridge_model.predict(rows)
    preds_persist = [float(r["y_prev"]) for r in rows]
    preds_persist_cal = persist_cal.predict(rows)
    preds_mean = [global_mean_value for _ in rows]

    output_path = output_dir / "temporal_predictions_test.csv"
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "facade_id",
                "year_t",
                "year_t1",
                "y_true",
                "y_pred_ridge",
                "y_pred_persist",
                "y_pred_persist_cal",
                "y_pred_mean",
                "abs_residual_ridge",
            ]
        )
        for row, y, pr, pp, ppc, pm in zip(rows, y_true, preds_ridge, preds_persist, preds_persist_cal, preds_mean):
            writer.writerow(
                [
                    row.get("facade_id"),
                    row.get("year_t"),
                    row.get("year_t1"),
                    y,
                    pr,
                    pp,
                    ppc,
                    pm,
                    abs(y - pr),
                ]
            )

    leaderboard_path = output_dir / "temporal_leaderboard_test.csv"
    by_facade: Dict[object, List[Tuple[float, float]]] = {}
    for row, pred in zip(rows, preds_ridge):
        by_facade.setdefault(row.get("facade_id"), []).append((float(row["y_true"]), float(pred)))

    leaderboard_rows: List[List[object]] = []
    for facade_id, vals in by_facade.items():
        abs_residuals = [abs(y - p) for y, p in vals]
        leaderboard_rows.append(
            [
                facade_id,
                mean(abs_residuals),
                max(abs_residuals),
                mean([y for y, _ in vals]),
                mean([p for _, p in vals]),
                len(vals),
            ]
        )
    leaderboard_rows.sort(key=lambda r: r[1], reverse=True)

    with leaderboard_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "facade_id",
                "mean_abs_residual",
                "max_abs_residual",
                "mean_y_true",
                "mean_y_pred_ridge",
                "n_samples",
                "risk_rank",
                "risk_label",
            ]
        )
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


def parse_alphas(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out if out else list(DEFAULT_ALPHAS)


if __name__ == "__main__":
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = read_table(args.features_path)
    index_rows = read_table(args.index_path)
    feature_rows = ensure_step_idx(feature_rows, index_rows)
    merged_rows = merge_index(feature_rows, index_rows)

    base_feature_cols = pick_feature_columns(merged_rows, args.target_column)
    supervised = build_supervised_table(merged_rows, args.target_column, base_feature_cols)

    # Add engineered numeric features into the feature list explicitly
    feature_cols = base_feature_cols + ["dt_prev", "quality_score"]

    if args.split_mode == "within_facade":
        splits = split_within_facade(supervised)
    else:
        splits = split_facade_disjoint(supervised, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.split_seed)

    if not splits["train"]:
        raise ValueError("Train split is empty; check your split parameters/data.")

    # 1) Fit persistence calibrator on train (very strong baseline when Spearman is good)
    persist_cal = PersistenceCalibrator()
    persist_cal.fit(splits["train"])

    # 2) Tune ridge alpha on train with group folds (by facade_id)
    alphas = parse_alphas(args.alphas)
    best_alpha, cv_mae_by_alpha = tune_ridge_alpha(
        splits["train"],
        feature_cols,
        alphas=alphas,
        scaler=args.scaler,
        n_folds=args.cv_folds,
        seed=args.split_seed,
    )

    ridge = RidgeModel(alpha=best_alpha, scaler=args.scaler)
    ridge.fit(splits["train"], feature_cols)
    global_mean_value = mean([float(r["y_true"]) for r in splits["train"]])

    metrics = evaluate_models(splits, ridge, persist_cal, global_mean_value)
    # include model selection info
    metrics["_model_selection"] = {
        "split_mode": args.split_mode,
        "scaler": args.scaler,
        "alphas": [float(a) for a in alphas],
        "best_alpha": float(best_alpha),
        "cv_folds": int(args.cv_folds),
        "cv_mae_by_alpha": cv_mae_by_alpha,
    }

    metrics_path = output_dir / "temporal_eval_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    if not splits["test"]:
        raise ValueError("Test split is empty; cannot produce predictions")

    save_predictions(splits["test"], ridge, persist_cal, global_mean_value, output_dir)
