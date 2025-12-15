"""Aggregate ordered facade pairs into time-series features for temporal modeling.

The script consumes outputs of ``run_ref_spx_pipeline`` where pair directories follow the
``<year_prev>_<year_next>`` convention (pairs_consecutive → temporal modeling) and builds
z_t feature vectors per facade.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build time-series features from facade pairs.")
    parser.add_argument("--batch-out-dir", type=Path, required=True, help="Output dir from run_ref_spx_pipeline")
    parser.add_argument("--pairs-summary", type=Path, required=True, help="pairs_summary.csv from the batch run")
    parser.add_argument("--min-coverage", type=float, default=0.8, help="Minimum coverage_mean to mark a pair as valid")
    parser.add_argument(
        "--delta-threshold",
        type=str,
        default="p90",
        help="Threshold for strong changes: either a float or percentile label like p90",
    )
    return parser.parse_args()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights)
    if not np.any(mask):
        return float("nan")

    order = np.argsort(values[mask])
    sorted_vals = values[mask][order]
    sorted_weights = weights[mask][order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    idx = np.searchsorted(cumulative, percentile / 100.0)
    idx = min(idx, sorted_vals.size - 1)
    return float(sorted_vals[idx])


def choose_area_weights(df: pd.DataFrame) -> np.ndarray:
    for candidate in ("area_px_ref", "area_px_lbl", "area_px_src"):
        if candidate in df.columns:
            weights = df[candidate].to_numpy(dtype=float)
            if np.isfinite(weights).any():
                return weights
    return np.ones(len(df), dtype=float)


def parse_threshold_arg(arg: str) -> Tuple[str, Optional[float]]:
    try:
        return "fixed", float(arg)
    except ValueError:
        pass

    if arg.lower().startswith("p"):
        try:
            return "percentile", float(arg[1:])
        except ValueError:
            pass
    raise ValueError(f"Unrecognized delta-threshold: {arg}")


def compute_features(df: pd.DataFrame, threshold_spec: Tuple[str, Optional[float]]) -> Tuple[dict, bool]:
    weights = choose_area_weights(df)
    deltas = df["delta_main"].to_numpy(dtype=float)
    coverage = df["coverage"].to_numpy(dtype=float) if "coverage" in df.columns else np.full_like(deltas, np.nan)

    thr_kind, thr_value = threshold_spec
    if thr_kind == "percentile":
        threshold = weighted_percentile(deltas, weights, thr_value or 0.0)
    else:
        threshold = thr_value or 0.0

    mean_delta = weighted_mean(deltas, weights)
    p90_delta = weighted_percentile(deltas, weights, 90.0)
    p95_delta = weighted_percentile(deltas, weights, 95.0)
    max_delta = float(np.nanmax(deltas)) if np.isfinite(deltas).any() else float("nan")

    valid_delta = np.isfinite(deltas)
    strong_mask = valid_delta & np.isfinite(threshold) & (deltas > threshold)
    total_weight = float(weights[valid_delta].sum())
    frac_strong = float(weights[strong_mask].sum() / total_weight) if total_weight > 0 else float("nan")

    coverage_mean = weighted_mean(coverage, weights)
    coverage_p10 = weighted_percentile(coverage, weights, 10.0)

    features = {
        "z_mean_delta": mean_delta,
        "z_p90_delta": p90_delta,
        "z_p95_delta": p95_delta,
        "z_max_delta": max_delta,
        "z_frac_strong": frac_strong,
        "z_coverage_mean": coverage_mean,
        "z_coverage_p10": coverage_p10,
    }

    is_valid = np.isfinite(mean_delta) and np.isfinite(coverage_mean)
    return features, is_valid


def load_delta_full(delta_path: Path) -> Optional[pd.DataFrame]:
    if not delta_path.exists():
        return None
    df = pd.read_parquet(delta_path)
    if "delta_main" not in df.columns:
        return None
    return df


def build_timeseries(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs_df = pd.read_csv(args.pairs_summary)
    required_cols = {"facade_id", "year_prev", "year_next", "dt_years"}
    if not required_cols.issubset(pairs_df.columns):
        missing = required_cols - set(pairs_df.columns)
        raise ValueError(f"pairs_summary is missing columns: {missing}")

    thr_spec = parse_threshold_arg(args.delta_threshold)
    feature_rows = []
    index_rows = []

    for facade_id, group in pairs_df.groupby("facade_id"):
        group_sorted = group.sort_values("year_prev")
        total_pairs = len(group_sorted)
        valid_pairs = 0

        for _, row in group_sorted.iterrows():
            year_prev = int(row["year_prev"])
            year_next = int(row["year_next"])
            dt_years = float(row["dt_years"])

            delta_path = (
                args.batch_out_dir
                / "pairs"
                / str(facade_id)
                / f"{year_prev}_{year_next}"
                / "delta_full.parquet"
            )

            df = load_delta_full(delta_path)
            if df is None or df.empty:
                feature_rows.append(
                    {
                        "facade_id": facade_id,
                        "year_prev": year_prev,
                        "year_next": year_next,
                        "dt_years": dt_years,
                        "valid_pair": False,
                        "z_mean_delta": np.nan,
                        "z_p90_delta": np.nan,
                        "z_p95_delta": np.nan,
                        "z_max_delta": np.nan,
                        "z_frac_strong": np.nan,
                        "z_coverage_mean": np.nan,
                        "z_coverage_p10": np.nan,
                    }
                )
                continue

            features, valid = compute_features(df, thr_spec)
            valid &= features["z_coverage_mean"] >= args.min_coverage
            valid_pairs += int(valid)

            feature_rows.append(
                {
                    "facade_id": facade_id,
                    "year_prev": year_prev,
                    "year_next": year_next,
                    "dt_years": dt_years,
                    "valid_pair": valid,
                    **features,
                }
            )

        pct_valid = valid_pairs / total_pairs if total_pairs else 0.0
        index_rows.append(
            {
                "facade_id": facade_id,
                "num_pairs": total_pairs,
                "num_valid_pairs": valid_pairs,
                "pct_valid_pairs": pct_valid,
                "min_year": int(group_sorted["year_prev"].min()),
                "max_year": int(group_sorted["year_next"].max()),
                "usable_for_temporal_model": bool(valid_pairs > 0 and pct_valid >= args.min_coverage),
            }
        )

    features_df = pd.DataFrame(feature_rows)
    index_df = pd.DataFrame(index_rows)
    return features_df, index_df


def main() -> None:
    args = parse_args()
    features_df, index_df = build_timeseries(args)

    out_parquet = args.batch_out_dir / "timeseries_features.parquet"
    out_index = args.batch_out_dir / "timeseries_index.csv"

    features_df.to_parquet(out_parquet, index=False)
    index_df.to_csv(out_index, index=False)

    print(f"Saved time-series features to {out_parquet}")
    print(f"Saved time-series index to {out_index}")


if __name__ == "__main__":
    main()
