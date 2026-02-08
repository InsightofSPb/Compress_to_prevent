#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train patch-level baselines (Ridge / HistGradientBoosting) for patch_changes.*.

Expected columns (minimum):
  - facade_id
  - patch_id
  - year_a/year_b  (or year_prev/year_next)
  - target column (e.g., delta_mean_w or risk_patch, etc.)

The script:
  - loads patch changes (csv/parquet)
  - creates facade-disjoint (default) or patch-disjoint split
  - imputes numeric features using train medians
  - trains Ridge+Scaler and HGB
  - reports MAE/RMSE/Spearman on train/val/test
  - (optional) aggregates patch predictions to facade-level and reports metrics too
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)

DEFAULT_PATCH_CHANGES = (
    "/home/sasha/LPOSS/datasets/SPb_facades/facades_with_years/"
    "patch_changes/patch_changes.csv"
)
DEFAULT_OUT_DIR = (
    "/home/sasha/LPOSS/datasets/SPb_facades/facades_with_years/patch_baseline"
)

KEY_COLUMNS = ("facade_id", "patch_id", "year_a", "year_b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train patch-level baselines.")
    p.add_argument("--patch-changes", type=Path, default=Path(DEFAULT_PATCH_CHANGES),
                   help="patch_changes.csv/parquet produced by tools/build_patch_changes.py")
    p.add_argument("--target-col", type=str, default="delta_mean_w",
                   help="Target column name in patch_changes (e.g., delta_mean_w or risk).")
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)

    p.add_argument("--split-mode", type=str, default="facade_disjoint",
                   choices=["facade_disjoint", "patch_disjoint"],
                   help="How to split groups to avoid leakage.")
    p.add_argument("--compute-permutation-importance", action="store_true",
                   help="If set: compute permutation importance for HGB on val split.")
    p.add_argument("--facade-agg", type=str, default="none",
                   choices=["none", "mean", "max", "p95mean"],
                   help="Also report facade-level metrics by aggregating patch preds per facade/year pair.")
    return p.parse_args()


def _coerce_years(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in [".csv", ".tsv"]:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported file format: {path}")


def _ensure_year_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "year_a" in df.columns and "year_b" in df.columns:
        pass
    elif "year_prev" in df.columns and "year_next" in df.columns:
        df = df.rename(columns={"year_prev": "year_a", "year_next": "year_b"})
    else:
        raise ValueError("Expected year_a/year_b or year_prev/year_next in patch_changes.")
    df = _coerce_years(df, ["year_a", "year_b"])
    return df


def _select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    exclude = set(KEY_COLUMNS)
    exclude.add("split")
    exclude.add(target_col)

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def _split_groups(df: pd.DataFrame, seed: int, train_frac: float, val_frac: float, mode: str) -> pd.DataFrame:
    if mode == "facade_disjoint":
        group_col = "facade_id"
    else:
        group_col = "patch_id"

    groups = df[group_col].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_total = len(groups)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))

    train_set = set(groups[:n_train])
    val_set = set(groups[n_train:n_train + n_val])

    def assign_split(x):
        if x in train_set:
            return "train"
        if x in val_set:
            return "val"
        return "test"

    out = df.copy()
    out["split"] = out[group_col].apply(assign_split)
    return out


def _impute_from_train(train_df: pd.DataFrame, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    medians = {}
    for col in feature_cols:
        s = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = float(s.median()) if s.notna().any() else 0.0
        if not np.isfinite(med):
            med = 0.0
        medians[col] = med

    filled = df.copy()
    for col in feature_cols:
        filled[col] = (
            pd.to_numeric(filled[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(medians[col])
        )

    X = filled[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, filled


def _evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    corr = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    if corr is None or (isinstance(corr, float) and not np.isfinite(corr)):
        corr = float("nan")
    return {"mae": float(mae), "rmse": float(rmse), "spearman": float(corr)}


def _facade_aggregate(pred_df: pd.DataFrame, how: str) -> pd.DataFrame:
    """
    pred_df columns: facade_id, patch_id, year_a, year_b, y_true, y_pred, split, model
    """
    gcols = ["facade_id", "year_a", "year_b", "split", "model"]

    def agg_p95mean(x: pd.Series) -> float:
        if x.empty:
            return float("nan")
        q = np.nanpercentile(x.to_numpy(dtype=float), 95)
        sel = x[x >= q]
        return float(sel.mean()) if len(sel) else float(np.nanmean(x.to_numpy(dtype=float)))

    if how == "mean":
        out = pred_df.groupby(gcols, as_index=False).agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
    elif how == "max":
        out = pred_df.groupby(gcols, as_index=False).agg(y_true=("y_true", "max"), y_pred=("y_pred", "max"))
    elif how == "p95mean":
        out = pred_df.groupby(gcols, as_index=False).agg(
            y_true=("y_true", agg_p95mean),
            y_pred=("y_pred", agg_p95mean),
        )
    else:
        raise ValueError(how)
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_df(args.patch_changes)
    df = _ensure_year_cols(df)

    # sanity checks
    need = {"facade_id", "patch_id", "year_a", "year_b", args.target_col}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"patch_changes missing columns: {missing}")

    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df = df[df[args.target_col].notna()].copy()
    if df.empty:
        raise ValueError("Dataset is empty after dropping NaN targets.")

    df = _split_groups(df, args.seed, args.train_frac, args.val_frac, args.split_mode)
    split_path = args.out_dir / "split.parquet"
    df.to_parquet(split_path, index=False)

    feature_cols = _select_feature_columns(df, args.target_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train, train_f = _impute_from_train(train_df, train_df, feature_cols)
    y_train = pd.to_numeric(train_f[args.target_col], errors="coerce").to_numpy(dtype=float)

    X_val, val_f = _impute_from_train(train_df, val_df, feature_cols)
    y_val = pd.to_numeric(val_f[args.target_col], errors="coerce").to_numpy(dtype=float)

    X_test, test_f = _impute_from_train(train_df, test_df, feature_cols)
    y_test = pd.to_numeric(test_f[args.target_col], errors="coerce").to_numpy(dtype=float)

    model_defs = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        "hgb": HistGradientBoostingRegressor(random_state=args.seed),
    }

    metrics_out: Dict[str, Dict[str, Dict[str, float]]] = {}
    preds_out = []
    imps_out = []

    for name, model in model_defs.items():
        model.fit(X_train, y_train)

        metrics_out[name] = {
            "train": _evaluate_metrics(y_train, model.predict(X_train)),
            "val": _evaluate_metrics(y_val, model.predict(X_val)),
        }
        if len(test_df) > 0:
            metrics_out[name]["test"] = _evaluate_metrics(y_test, model.predict(X_test))

        # predictions table
        for split_name, split_df, X, y in [
            ("train", train_f, X_train, y_train),
            ("val", val_f, X_val, y_val),
            ("test", test_f, X_test, y_test),
        ]:
            if split_df.empty:
                continue
            p = model.predict(X)
            t = split_df[list(KEY_COLUMNS)].copy()
            t["y_true"] = y
            t["y_pred"] = p
            t["model"] = name
            t["split"] = split_name
            preds_out.append(t)

        # permutation importance (optional)
        if args.compute_permutation_importance and name == "hgb" and X_val.size > 0:
            perm = permutation_importance(
                model,
                X_val,
                y_val,
                n_repeats=5,
                random_state=args.seed,
                scoring="neg_mean_absolute_error",
            )
            imp = pd.DataFrame({
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
                "model": name,
            })
            imps_out.append(imp)

    # save
    metrics_path = args.out_dir / "metrics_patch.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    pred_path = None
    if preds_out:
        pred_df = pd.concat(preds_out, ignore_index=True)
        pred_path = args.out_dir / "predictions_patch.parquet"
        pred_df.to_parquet(pred_path, index=False)

        if args.facade_agg != "none":
            fac_df = _facade_aggregate(pred_df, args.facade_agg)
            facade_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
            for model_name in fac_df["model"].unique():
                facade_metrics[model_name] = {}
                for split_name in ["train", "val", "test"]:
                    sub = fac_df[(fac_df["model"] == model_name) & (fac_df["split"] == split_name)]
                    y_t = sub["y_true"].to_numpy(dtype=float)
                    y_p = sub["y_pred"].to_numpy(dtype=float)
                    facade_metrics[model_name][split_name] = _evaluate_metrics(y_t, y_p)

            facade_metrics_path = args.out_dir / f"metrics_facade_from_patches_{args.facade_agg}.json"
            with facade_metrics_path.open("w", encoding="utf-8") as f:
                json.dump(facade_metrics, f, indent=2, ensure_ascii=False)

    if imps_out:
        imp_path = args.out_dir / "feature_importance_patch.csv"
        pd.concat(imps_out, ignore_index=True).to_csv(imp_path, index=False)

    LOGGER.info("Saved split: %s", split_path)
    LOGGER.info("Saved patch metrics: %s", metrics_path)
    if pred_path is not None:
        LOGGER.info("Saved predictions: %s", pred_path)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
