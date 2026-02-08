#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 3:
        return float("nan")
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return float("nan")

    if spearmanr is None:
        rt = pd.Series(y_true).rank().to_numpy()
        rp = pd.Series(y_pred).rank().to_numpy()
        return float(np.corrcoef(rt, rp)[0, 1])

    return float(spearmanr(y_true, y_pred).correlation)


def infer_feature_sets(df, target, temporal_only: bool):
    drop_always = {"facade_id", "patch_id", "risk_label_patch", "split"}

    semantic_prefixes = (
        "entropy_", "max_share_",
        "share_class_", "delta_share_class_",
        "area_px_class_", "delta_area_px_class_",
        "l1_share_change", "max_delta_share", "js_divergence",
    )

    semantic_cols = []
    for c in df.columns:
        if c in drop_always or c == target:
            continue
        if c.startswith("risk_"):
            continue
        if c.startswith(semantic_prefixes) or c in {"entropy_t", "entropy_t1", "max_share_t", "max_share_t1"}:
            semantic_cols.append(c)

    compress_candidates = ["comp_signal_t", "comp_signal_t1", "delta_comp_signal", "abs_delta_comp_signal"]
    compress_cols = [c for c in compress_candidates if c in df.columns and c != target]

    time_cols = [c for c in ["step_idx", "year_prev", "year_next"] if c in df.columns and c != target]

    if temporal_only:
        semantic_cols = [c for c in semantic_cols if c.endswith("_t") or c in {"step_idx", "year_prev", "year_next"}]
        compress_cols = [c for c in compress_cols if c.endswith("_t")]
        semantic_cols = [c for c in semantic_cols if (not c.endswith("_t1")) and (not c.startswith("delta_"))]

    cat_cols = []
    if "quality" in df.columns and "quality" != target:
        cat_cols.append("quality")

    lposs_only = sorted(set(semantic_cols + time_cols + cat_cols))
    compress_only = sorted(set(compress_cols + time_cols + cat_cols))
    full = sorted(set(semantic_cols + compress_cols + time_cols + cat_cols))

    def clean(cols):
        out = []
        for c in cols:
            if c in drop_always or c == target:
                continue
            if c.startswith("risk_"):
                continue
            out.append(c)
        return out

    return {
        "lposs_only": clean(lposs_only),
        "compress_only": clean(compress_only),
        "full": clean(full),
    }


def build_model(kind: str, num_cols, cat_cols):
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    if kind == "ridge":
        numeric_tf = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if kind == "ridge":
        reg = Ridge(alpha=1.0, random_state=42)
    elif kind == "hgb":
        reg = HistGradientBoostingRegressor(
            random_state=42,
            max_depth=None,
            learning_rate=0.05,
            max_iter=500,
        )
    else:
        raise ValueError(kind)

    return Pipeline(steps=[("pre", pre), ("reg", reg)])


def eval_split(model, X, y):
    pred = model.predict(X)
    mae = float(mean_absolute_error(y, pred))
    rmse = float(mean_squared_error(y, pred) ** 0.5)
    sp = safe_spearman(y, pred)
    return {"mae": mae, "rmse": rmse, "spearman": float(sp)}


def assign_split_by_year(df: pd.DataFrame, year_col: str, n_test_years: int, n_val_years: int):
    if year_col not in df.columns:
        raise SystemExit(f"--year-col {year_col} not found in CSV columns.")

    y = pd.to_numeric(df[year_col], errors="coerce")
    ok = y.notna()
    df = df.loc[ok].copy()
    y = y.loc[ok].astype(int)

    years = sorted(y.unique().tolist())
    need = n_test_years + n_val_years + 1
    if len(years) < need:
        raise SystemExit(
            f"Not enough unique years in {year_col}: {years}. "
            f"Need at least {need} (train + val + test)."
        )

    test_years = set(years[-n_test_years:])
    val_years = set(years[-(n_test_years + n_val_years):-n_test_years])
    # остальные = train
    split = np.where(y.isin(test_years), "test", np.where(y.isin(val_years), "val", "train"))
    df["split"] = split
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="patch_changes.csv")
    ap.add_argument("--target", default="risk_total",
                    help="что предсказываем: js_divergence / l1_share_change / max_delta_share / share_class_3_t1 ...")
    ap.add_argument("--temporal-only", action="store_true",
                    help="использовать только признаки на t (без *_t1 и delta_*)")

    # NEW: split by year
    ap.add_argument("--split-by-year", action="store_true",
                    help="пересобрать split по годам (игнорирует колонку split в CSV)")
    ap.add_argument("--year-col", default="year_next",
                    help="по какой колонке делить годы: year_next (default) или year_prev")
    ap.add_argument("--n-test-years", type=int, default=1,
                    help="сколько самых поздних лет отдать в test")
    ap.add_argument("--n-val-years", type=int, default=1,
                    help="сколько лет перед test отдать в val")

    args = ap.parse_args()

    df = pd.read_csv(args.csv, skipinitialspace=True)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    if args.split_by_year:
        df = assign_split_by_year(df, args.year_col, args.n_test_years, args.n_val_years)
    else:
        if "split" not in df.columns:
            raise SystemExit("Column 'split' not found. Либо добавь split, либо используй --split-by-year.")

    # normalize split ALWAYS
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found in columns.")

    # target
    y = pd.to_numeric(df[args.target], errors="coerce")
    ok = y.notna()
    df = df.loc[ok].copy()
    y = y.loc[ok].to_numpy(dtype=float)

    # debug: split counts after filtering by target
    counts = df["split"].value_counts().to_dict()
    print("Split counts AFTER target filtering:", counts)

    if not (df["split"] == "train").any():
        raise SystemExit(
            f"No train rows after filtering target '{args.target}'. "
            f"Try another target (e.g. js_divergence) or recompute/fill '{args.target}'. "
            f"Counts: {counts}"
        )

    feature_sets = infer_feature_sets(df, target=args.target, temporal_only=args.temporal_only)

    results = {}
    for mode, cols in feature_sets.items():
        cat_cols = [c for c in cols if df[c].dtype == "object"]
        num_cols = [c for c in cols if c not in cat_cols]

        X = df[cols].copy()
        split = df["split"].to_numpy()

        res_mode = {}
        for model_kind in ["ridge", "hgb"]:
            model = build_model(model_kind, num_cols=num_cols, cat_cols=cat_cols)

            m_train = (split == "train")
            m_val = (split == "val")
            m_test = (split == "test")

            X_train, y_train = X.loc[m_train], y[m_train]
            X_val, y_val = X.loc[m_val], y[m_val]
            X_test, y_test = X.loc[m_test], y[m_test]

            model.fit(X_train, y_train)

            res_mode[model_kind] = {
                "train": eval_split(model, X_train, y_train),
                "val": eval_split(model, X_val, y_val) if len(X_val) else {},
                "test": eval_split(model, X_test, y_test) if len(X_test) else {},
            }

        results[mode] = res_mode

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
