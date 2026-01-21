import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute patch-level change metrics and risk scores.")
    parser.add_argument("--features-path", required=True, type=Path, help="Path to timeseries_features_patch.parquet (CSV allowed)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for patch changes and leaderboard")
    parser.add_argument(
        "--split-mode",
        default="within_patch",
        choices=("within_patch", "within_facade", "patch_disjoint", "facade_disjoint"),
        help="Splitting strategy for scaling/thresholds.",
    )
    parser.add_argument("--train-frac", default=0.7, type=float, help="Used only for disjoint split modes")
    parser.add_argument("--val-frac", default=0.15, type=float, help="Used only for disjoint split modes")
    parser.add_argument("--split-seed", default=13, type=int, help="Seed for disjoint split modes")
    parser.add_argument("--risk-weight-semantic", default=1.0, type=float)
    parser.add_argument("--risk-weight-compression", default=1.0, type=float)
    parser.add_argument("--high-quantile", default=0.9, type=float, help="Quantile for HIGH risk (unsupervised)")
    parser.add_argument("--med-quantile", default=0.7, type=float, help="Quantile for MED risk (unsupervised)")
    parser.add_argument("--supervised-thresholds", action="store_true", help="Tune thresholds using target-column")
    parser.add_argument("--target-column", default=None, type=str, help="Optional target for supervised threshold selection")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def rankdata(values: Iterable[float]) -> List[float]:
    sorted_vals = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * len(sorted_vals)
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


def spearmanr(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) <= 1:
        return float("nan")
    rank_true = rankdata(y_true)
    rank_pred = rankdata(y_pred)
    mu_true = sum(rank_true) / len(rank_true)
    mu_pred = sum(rank_pred) / len(rank_pred)
    num = sum((a - mu_true) * (b - mu_pred) for a, b in zip(rank_true, rank_pred))
    den = math.sqrt(
        sum((a - mu_true) ** 2 for a in rank_true) * sum((b - mu_pred) ** 2 for b in rank_pred)
    )
    return num / den if den != 0 else float("nan")


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if p.size == 0:
        return float("nan")
    p = np.clip(p.astype(float), 0, None)
    q = np.clip(q.astype(float), 0, None)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * (_kl(p, m) + _kl(q, m))


def _group_key(row: pd.Series, mode: str):
    if mode in {"within_patch", "patch_disjoint"}:
        return (row.get("facade_id"), row.get("patch_id"))
    return row.get("facade_id")


def _order_key(row: pd.Series) -> Tuple:
    if "step_idx" in row:
        return (row.get("step_idx"), row.get("year_next"))
    return (row.get("year_next"), row.get("year_prev"))


def split_within_group(df: pd.DataFrame, mode: str) -> pd.Series:
    split_labels = pd.Series(index=df.index, dtype=object)
    df_sorted = df.sort_values(["facade_id", "patch_id", "year_next", "year_prev"], kind="mergesort")
    for _, group in df_sorted.groupby(lambda idx: _group_key(df_sorted.loc[idx], mode)):
        idxs = list(group.index)
        if len(idxs) == 1:
            split_labels.loc[idxs[0]] = "test"
        elif len(idxs) == 2:
            split_labels.loc[idxs[0]] = "val"
            split_labels.loc[idxs[1]] = "test"
        else:
            split_labels.loc[idxs[:-2]] = "train"
            split_labels.loc[idxs[-2]] = "val"
            split_labels.loc[idxs[-1]] = "test"
    return split_labels.reindex(df.index)


def split_disjoint(df: pd.DataFrame, mode: str, train_frac: float, val_frac: float, seed: int) -> pd.Series:
    groups = sorted({_group_key(row, mode) for _, row in df.iterrows()}, key=lambda x: str(x))
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(0, int(round(val_frac * n)))
    if n_train + n_val >= n and n > 1:
        n_val = max(0, n - n_train - 1)

    train_set = set(groups[:n_train])
    val_set = set(groups[n_train : n_train + n_val])
    split_labels = []
    for _, row in df.iterrows():
        key = _group_key(row, mode)
        if key in train_set:
            split_labels.append("train")
        elif key in val_set:
            split_labels.append("val")
        else:
            split_labels.append("test")
    return pd.Series(split_labels, index=df.index)


def compute_robust_stats(values: Iterable[float]) -> Tuple[float, float]:
    vals = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if vals.size == 0:
        return 0.0, 1.0
    med = float(np.median(vals))
    q75 = float(np.percentile(vals, 75))
    q25 = float(np.percentile(vals, 25))
    iqr = q75 - q25
    return med, iqr if iqr > 0 else 1.0


def ensure_semantic_components(df: pd.DataFrame) -> pd.DataFrame:
    share_cols_t = sorted([c for c in df.columns if c.startswith("share_") and c.endswith("_t")])
    share_cols_t1 = sorted([c for c in df.columns if c.startswith("share_") and c.endswith("_t1")])
    delta_share_cols = [c for c in df.columns if c.startswith("delta_share_")]

    if not delta_share_cols and share_cols_t and share_cols_t1:
        for col_t in share_cols_t:
            col_t1 = col_t.replace("_t", "_t1")
            if col_t1 in df.columns:
                base = col_t.replace("share_", "").replace("_t", "")
                delta_col = f"delta_share_{base}"
                df[delta_col] = df[col_t1].fillna(0.0) - df[col_t].fillna(0.0)
        delta_share_cols = [c for c in df.columns if c.startswith("delta_share_")]

    if "l1_share_change" not in df.columns and delta_share_cols:
        df["l1_share_change"] = df[delta_share_cols].abs().sum(axis=1)

    if "max_delta_share" not in df.columns and delta_share_cols:
        df["max_delta_share"] = df[delta_share_cols].abs().max(axis=1)

    if "js_divergence" not in df.columns and share_cols_t and share_cols_t1:
        js_vals = []
        for _, row in df.iterrows():
            p = np.array([row.get(c, 0.0) or 0.0 for c in share_cols_t], dtype=float)
            q = np.array([row.get(c.replace("_t", "_t1"), 0.0) or 0.0 for c in share_cols_t], dtype=float)
            js_vals.append(js_divergence(p, q))
        df["js_divergence"] = js_vals

    return df


def assign_labels(values: pd.Series, med_threshold: float, high_threshold: float) -> pd.Series:
    labels = []
    for v in values:
        if v >= high_threshold:
            labels.append("HIGH")
        elif v >= med_threshold:
            labels.append("MED")
        else:
            labels.append("LOW")
    return pd.Series(labels, index=values.index)


def tune_thresholds(
    train_df: pd.DataFrame,
    risk_values: pd.Series,
    target_col: str,
    med_candidates: Iterable[float],
    high_candidates: Iterable[float],
) -> Tuple[float, float]:
    best_score = float("-inf")
    best = (np.quantile(risk_values, 0.7), np.quantile(risk_values, 0.9))
    target = train_df[target_col].astype(float)
    for med_q in med_candidates:
        for high_q in high_candidates:
            if med_q >= high_q:
                continue
            med_thr = float(np.quantile(risk_values, med_q))
            high_thr = float(np.quantile(risk_values, high_q))
            labels = assign_labels(risk_values, med_thr, high_thr).map({"LOW": 0, "MED": 1, "HIGH": 2})
            score = spearmanr(target, labels)
            if not math.isnan(score) and score > best_score:
                best_score = score
                best = (med_thr, high_thr)
    return best


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.features_path)
    required = {"facade_id", "patch_id", "year_prev", "year_next"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"features-path missing required columns: {sorted(missing)}")

    df = ensure_semantic_components(df)

    if "abs_delta_comp_signal" not in df.columns and "delta_comp_signal" in df.columns:
        df["abs_delta_comp_signal"] = df["delta_comp_signal"].abs()

    if args.split_mode in {"within_patch", "within_facade"}:
        df["split"] = split_within_group(df, args.split_mode)
    else:
        df["split"] = split_disjoint(df, args.split_mode, args.train_frac, args.val_frac, args.split_seed)

    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("Train split is empty; check split parameters/data.")

    semantic_cols = [c for c in ("l1_share_change", "js_divergence", "max_delta_share") if c in df.columns]
    df["risk_semantic_raw"] = df[semantic_cols].sum(axis=1, skipna=True) if semantic_cols else 0.0
    if "abs_delta_comp_signal" in df.columns:
        df["risk_compression_raw"] = df["abs_delta_comp_signal"].fillna(0.0)
    else:
        df["risk_compression_raw"] = 0.0

    sem_med, sem_iqr = compute_robust_stats(df.loc[df["split"] == "train", "risk_semantic_raw"])
    comp_med, comp_iqr = compute_robust_stats(df.loc[df["split"] == "train", "risk_compression_raw"])

    df["risk_semantic"] = (df["risk_semantic_raw"] - sem_med) / sem_iqr
    df["risk_compression"] = (df["risk_compression_raw"] - comp_med) / comp_iqr
    df["risk_total"] = args.risk_weight_semantic * df["risk_semantic"] + args.risk_weight_compression * df["risk_compression"]

    thresholds = {}
    risk_train = df.loc[df["split"] == "train", "risk_total"]
    if args.supervised_thresholds and args.target_column and args.target_column in df.columns:
        med_thr, high_thr = tune_thresholds(
            train_df,
            risk_train,
            args.target_column,
            med_candidates=[0.6, 0.65, 0.7, 0.75],
            high_candidates=[0.8, 0.85, 0.9, 0.95],
        )
        thresholds.update({"mode": "supervised", "med_threshold": float(med_thr), "high_threshold": float(high_thr)})
    else:
        med_thr = float(np.quantile(risk_train, args.med_quantile))
        high_thr = float(np.quantile(risk_train, args.high_quantile))
        thresholds.update({"mode": "unsupervised", "med_quantile": args.med_quantile, "high_quantile": args.high_quantile})
        thresholds.update({"med_threshold": med_thr, "high_threshold": high_thr})

    df["risk_label_patch"] = assign_labels(df["risk_total"], thresholds["med_threshold"], thresholds["high_threshold"])

    thresholds.update(
        {
            "risk_weight_semantic": args.risk_weight_semantic,
            "risk_weight_compression": args.risk_weight_compression,
            "semantic_median": sem_med,
            "semantic_iqr": sem_iqr,
            "compression_median": comp_med,
            "compression_iqr": comp_iqr,
        }
    )

    thresholds_path = args.output_dir / "patch_risk_thresholds.json"
    with thresholds_path.open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2)

    changes_path = args.output_dir / "patch_changes.parquet"
    df.to_parquet(changes_path, index=False)
    df.to_csv(args.output_dir / "patch_changes.csv", index=False)

    leaderboard = (
        df.groupby(["facade_id", "patch_id"], as_index=False)
        .agg(
            max_risk_total=("risk_total", "max"),
            mean_risk_total=("risk_total", "mean"),
            n_samples=("risk_total", "size"),
        )
        .sort_values("max_risk_total", ascending=False)
    )
    leaderboard["risk_label_patch"] = assign_labels(
        leaderboard["max_risk_total"], thresholds["med_threshold"], thresholds["high_threshold"]
    )
    leaderboard["risk_rank"] = range(1, len(leaderboard) + 1)
    leaderboard_path = args.output_dir / "patch_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print(f"[OK] Saved patch changes to {changes_path}")
    print(f"[OK] Saved patch leaderboard to {leaderboard_path}")
    print(f"[OK] Saved risk thresholds to {thresholds_path}")


if __name__ == "__main__":
    main()
