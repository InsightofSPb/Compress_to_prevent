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

DEFAULT_PAIR_AGG = (
    "/home/sasha/LPOSS/datasets/SPb_facades/facades_with_years/"
    "compress_features_lposs/pair_agg.parquet"
)
DEFAULT_TARGET_FEATURES = (
    "/home/sasha/LPOSS/datasets/SPb_facades/facades_with_years/"
    "timeseries_features/timeseries_features.parquet"
)
DEFAULT_TARGET_INDEX = (
    "/home/sasha/LPOSS/datasets/SPb_facades/facades_with_years/"
    "timeseries_features/timeseries_index_steps.csv"
)
DEFAULT_OUT_DIR = (
    "/home/sasha/LPOSS/datasets/SPb_facades/facades_with_years/pair_baseline"
)


KEY_COLUMNS = ("facade_id", "year_a", "year_b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pair-level baselines for delta_mean_w.")
    parser.add_argument("--pair-agg", type=Path, default=Path(DEFAULT_PAIR_AGG))
    parser.add_argument("--target-features", type=Path, default=Path(DEFAULT_TARGET_FEATURES))
    parser.add_argument("--target-index", type=Path, default=Path(DEFAULT_TARGET_INDEX))
    parser.add_argument("--target-col", type=str, default="delta_mean_w")
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    return parser.parse_args()


def _coerce_years(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _load_target_features(
    target_path: Path,
    index_path: Optional[Path],
    target_col: str,
) -> pd.DataFrame:
    target_df = pd.read_parquet(target_path)
    if target_col not in target_df.columns:
        raise ValueError(f"Target column {target_col} not found in {target_path}")

    if "year_prev" in target_df.columns and "year_next" in target_df.columns:
        target_df = target_df.rename(columns={"year_prev": "year_a", "year_next": "year_b"})
    elif "year_a" in target_df.columns and "year_b" in target_df.columns:
        pass
    elif "step_idx" in target_df.columns:
        if index_path is None or not index_path.exists():
            raise ValueError("Target features missing years; index file required for step mapping.")
        index_df = pd.read_csv(index_path)
        if "step_idx" not in index_df.columns:
            raise ValueError("Index file missing step_idx for step mapping.")
        year_cols = None
        if {"year_prev", "year_next"}.issubset(index_df.columns):
            year_cols = ("year_prev", "year_next")
        elif {"year_a", "year_b"}.issubset(index_df.columns):
            year_cols = ("year_a", "year_b")
        if year_cols is None:
            raise ValueError("Index file missing year_prev/year_next or year_a/year_b columns.")
        index_df = index_df.rename(columns={year_cols[0]: "year_a", year_cols[1]: "year_b"})
        target_df = target_df.merge(index_df[["facade_id", "step_idx", "year_a", "year_b"]], on=["facade_id", "step_idx"])
    else:
        raise ValueError("Target features must include year_prev/year_next or step_idx columns.")

    target_df = _coerce_years(target_df, ["year_a", "year_b"])
    return target_df


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = set(KEY_COLUMNS)
    exclude.add("split")
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def _split_facades(df: pd.DataFrame, seed: int, train_frac: float, val_frac: float) -> pd.DataFrame:
    unique_facades = df["facade_id"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_facades)

    n_total = len(unique_facades)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))

    train_set = set(unique_facades[:n_train])
    val_set = set(unique_facades[n_train : n_train + n_val])

    def assign_split(facade_id: object) -> str:
        if facade_id in train_set:
            return "train"
        if facade_id in val_set:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df["facade_id"].apply(assign_split)
    return df


def _impute_from_train(
    train_df: pd.DataFrame, df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, pd.DataFrame]:
    medians: Dict[str, float] = {}
    for col in feature_cols:
        medians[col] = float(train_df[col].median()) if col in train_df.columns else 0.0
    filled = df.copy()
    for col in feature_cols:
        filled[col] = pd.to_numeric(filled[col], errors="coerce").fillna(medians[col])
    return filled[feature_cols].to_numpy(dtype=float), filled


def _evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    spearman = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return {"mae": float(mae), "rmse": float(rmse), "spearman": float(spearman)}


def _build_modes(df: pd.DataFrame) -> Dict[str, List[str]]:
    lposs_cols = [
        "p_damage_mean",
        "p_damage_p95",
        "entropy_norm_mean_mean",
        "entropy_norm_mean_p95",
        "margin_mean_mean",
        "margin_mean_p10",
        "mean_p_REPAIRS_mean",
        "mean_p_TEXT_OR_IMAGES_mean",
        "mean_p_ORNAMENT_INTACT_mean",
    ]
    compress_cols = [
        "support_area_ratio_mean",
        "support_area_ratio_p10",
        "delta_bpp_mean",
        "delta_bpp_median",
        "delta_bpp_p75",
        "delta_bpp_p95",
        "delta_bpp_rel_mean",
        "delta_bpp_rel_p95",
        "bpp_excess_b_p95",
    ]

    lposs_only = [c for c in lposs_cols if c in df.columns]
    compress_only = [c for c in compress_cols if c in df.columns]
    full = _select_feature_columns(df)
    full = [c for c in full if c != "y_true"]

    return {
        "lposs_only": lposs_only,
        "compress_only": compress_only,
        "full": full,
    }


def _train_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], pd.DataFrame, Optional[pd.DataFrame]]:
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    predictions: List[pd.DataFrame] = []
    importances: List[pd.DataFrame] = []

    if not feature_cols:
        return metrics, pd.DataFrame(), None

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    x_train, train_filled = _impute_from_train(train_df, train_df, feature_cols)
    y_train = pd.to_numeric(train_filled[target_col], errors="coerce").to_numpy(dtype=float)

    x_val, val_filled = _impute_from_train(train_df, val_df, feature_cols)
    y_val = pd.to_numeric(val_filled[target_col], errors="coerce").to_numpy(dtype=float)

    x_test, test_filled = _impute_from_train(train_df, test_df, feature_cols)
    y_test = pd.to_numeric(test_filled[target_col], errors="coerce").to_numpy(dtype=float)

    model_defs = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        "hgb": HistGradientBoostingRegressor(random_state=seed),
    }

    for model_name, model in model_defs.items():
        model.fit(x_train, y_train)

        metrics[model_name] = {
            "train": _evaluate_metrics(y_train, model.predict(x_train)),
            "val": _evaluate_metrics(y_val, model.predict(x_val)),
        }
        if len(test_df) > 0:
            metrics[model_name]["test"] = _evaluate_metrics(y_test, model.predict(x_test))

        for split_name, split_df, x_split, y_split in [
            ("train", train_filled, x_train, y_train),
            ("val", val_filled, x_val, y_val),
            ("test", test_filled, x_test, y_test),
        ]:
            if split_df.empty:
                continue
            preds = model.predict(x_split)
            pred_df = split_df[list(KEY_COLUMNS)].copy()
            pred_df["y_true"] = y_split
            pred_df["y_pred"] = preds
            pred_df["model"] = model_name
            pred_df["split"] = split_name
            predictions.append(pred_df)

        if model_name == "hgb" and x_val.size > 0:
            perm = permutation_importance(
                model,
                x_val,
                y_val,
                n_repeats=5,
                random_state=seed,
                scoring="neg_mean_absolute_error",
            )
            imp_df = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                }
            )
            imp_df["model"] = model_name
            importances.append(imp_df)

    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    importance_df = pd.concat(importances, ignore_index=True) if importances else None
    return metrics, pred_df, importance_df


def main() -> None:
    args = parse_args()

    pair_df = pd.read_parquet(args.pair_agg)
    pair_df = _coerce_years(pair_df, ["year_a", "year_b"])

    target_df = _load_target_features(args.target_features, args.target_index, args.target_col)
    target_df = target_df.rename(columns={args.target_col: "y_true"})

    merged = pair_df.merge(
        target_df[["facade_id", "year_a", "year_b", "y_true"]],
        on=["facade_id", "year_a", "year_b"],
        how="inner",
    )
    merged["y_true"] = pd.to_numeric(merged["y_true"], errors="coerce")
    merged = merged[merged["y_true"].notna()].copy()
    if merged.empty:
        raise ValueError("Merged dataset is empty after joining target.")

    merged = _split_facades(merged, args.seed, args.train_frac, args.val_frac)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.out_dir / "split.parquet"
    merged.to_parquet(split_path, index=False)

    modes = _build_modes(merged)
    metrics_out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    predictions_out: List[pd.DataFrame] = []
    importance_out: List[pd.DataFrame] = []

    for mode_name, feature_cols in modes.items():
        metrics, preds, importances = _train_models(merged, feature_cols, "y_true", args.seed)
        metrics_out[mode_name] = metrics
        if not preds.empty:
            preds = preds.copy()
            preds["mode"] = mode_name
            predictions_out.append(preds)
        if importances is not None and not importances.empty:
            importances = importances.copy()
            importances["mode"] = mode_name
            importance_out.append(importances)

    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_out, handle, indent=2, ensure_ascii=False)

    if predictions_out:
        pred_path = args.out_dir / "predictions.parquet"
        pd.concat(predictions_out, ignore_index=True).to_parquet(pred_path, index=False)

    if importance_out:
        importance_path = args.out_dir / "feature_importance.csv"
        pd.concat(importance_out, ignore_index=True).to_csv(importance_path, index=False)

    LOGGER.info("Saved split to %s", split_path)
    LOGGER.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
