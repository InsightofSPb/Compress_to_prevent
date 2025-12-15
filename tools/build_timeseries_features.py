import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_DELTA_CANDIDATES = [
    "delta_main",
    "delta_lab_ab_aligned",
    "delta_rgb",
    "delta_lab_ab",
    "delta_rgb_std",
    "delta_lab_all",
]
DEFAULT_COVERAGE_CANDIDATES = ["coverage", "coverage_src"]
DEFAULT_AREA_CANDIDATES = ["area_px_lbl", "area_px_ref"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build time-series features from ref_spx_batch_out/pairs parquet files "
            "for temporal modeling."
        )
    )
    parser.add_argument("--pairs-summary", required=True, type=Path, help="Path to pairs_summary.csv")
    parser.add_argument(
        "--pairs-root",
        required=True,
        type=Path,
        help="Root directory with pair folders (facade_id/year_a_year_b)",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for features")
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.85,
        help="Minimum coverage to keep a superpixel in aggregation",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=10.0,
        help="Threshold for delta exceedance metrics",
    )
    parser.add_argument(
        "--delta-column",
        type=str,
        default="delta_main",
        help="Preferred delta column; falls back to common variants if missing",
    )
    parser.add_argument(
        "--coverage-column",
        type=str,
        default=None,
        help="Preferred coverage column; falls back to common variants if missing",
    )
    parser.add_argument(
        "--min-quality",
        type=str,
        choices=["strong", "weak", "any"],
        default="strong",
        help="Minimum pair quality to include (weak allows weak+strong)",
    )
    parser.add_argument(
        "--min-coverage-median",
        type=float,
        default=None,
        help="Optional minimum median coverage to keep the pair",
    )
    return parser.parse_args()


def quality_ok(quality: str, min_quality: str) -> bool:
    quality = (quality or "").strip().lower()
    if min_quality == "any":
        return True
    if min_quality == "weak":
        return quality in {"weak", "strong"}
    return quality == "strong"


def pick_column(df: pd.DataFrame, preferred: Optional[str], candidates: Iterable[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for col in candidates:
        if col in df.columns:
            return col
    return None


def weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    if values.size == 0:
        return float("nan")
    if weights is None:
        return float(np.nanmean(values))
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def aggregate_pair(
    df: pd.DataFrame,
    coverage_col: str,
    delta_col: str,
    area_col: Optional[str],
    coverage_threshold: float,
    delta_threshold: float,
) -> Dict[str, float]:
    n_spx_total = int(len(df))

    valid_mask = df[coverage_col] >= coverage_threshold
    df_valid = df.loc[valid_mask].copy()

    if delta_col not in df_valid.columns:
        raise KeyError(f"Delta column {delta_col} not found in dataframe")

    coverage_vals = df_valid[coverage_col].to_numpy()
    delta_vals = df_valid[delta_col].to_numpy()

    area_vals = df_valid[area_col].to_numpy() if area_col else None

    n_spx_valid = int(len(df_valid))

    coverage_mean = float(np.nanmean(coverage_vals)) if coverage_vals.size else float("nan")
    coverage_median = float(np.nanmedian(coverage_vals)) if coverage_vals.size else float("nan")
    coverage_p10 = float(np.nanpercentile(coverage_vals, 10)) if coverage_vals.size else float("nan")

    delta_clean = delta_vals[~np.isnan(delta_vals)]
    delta_median = float(np.median(delta_clean)) if delta_clean.size else float("nan")
    delta_p90 = float(np.percentile(delta_clean, 90)) if delta_clean.size else float("nan")
    delta_p95 = float(np.percentile(delta_clean, 95)) if delta_clean.size else float("nan")
    delta_max = float(np.max(delta_clean)) if delta_clean.size else float("nan")
    delta_mean_w = weighted_mean(delta_vals, area_vals)

    if delta_clean.size:
        frac_delta_gt_T = float(np.mean(delta_clean > delta_threshold))
    else:
        frac_delta_gt_T = float("nan")

    if area_vals is not None:
        mask = (~np.isnan(delta_vals)) & (~np.isnan(area_vals))
        valid_weights = area_vals[mask]
        if valid_weights.size:
            area_frac_delta_gt_T = float(
                np.sum(valid_weights[delta_vals[mask] > delta_threshold]) / np.sum(valid_weights)
            )
        else:
            area_frac_delta_gt_T = float("nan")
    else:
        area_frac_delta_gt_T = float("nan")

    return {
        "n_spx_total": n_spx_total,
        "n_spx_valid": n_spx_valid,
        "coverage_mean": coverage_mean,
        "coverage_median": coverage_median,
        "coverage_p10": coverage_p10,
        "delta_mean_w": delta_mean_w,
        "delta_median": delta_median,
        "delta_p90": delta_p90,
        "delta_p95": delta_p95,
        "delta_max": delta_max,
        "frac_delta_gt_T": frac_delta_gt_T,
        "area_frac_delta_gt_T": area_frac_delta_gt_T,
    }


def build_features(args: argparse.Namespace) -> pd.DataFrame:
    pairs_df = pd.read_csv(args.pairs_summary)
    required_cols = {"facade_id", "year_a", "year_b"}
    missing = required_cols - set(pairs_df.columns)
    if missing:
        raise ValueError(f"pairs_summary missing columns: {missing}")

    rows: List[Dict[str, object]] = []
    pairs_root = args.pairs_root

    for _, row in pairs_df.iterrows():
        quality = str(row.get("quality", "")).strip().lower()
        if not quality_ok(quality, args.min_quality):
            continue

        facade_id = row["facade_id"]
        year_a = int(row["year_a"])
        year_b = int(row["year_b"])
        year_prev = min(year_a, year_b)
        year_next = max(year_a, year_b)
        dt_years = year_next - year_prev

        status = row.get("status", "")

        pair_dir = pairs_root / str(facade_id) / f"{year_a}_{year_b}"
        delta_path = pair_dir / "delta_full.parquet"
        if not delta_path.exists():
            delta_path = pair_dir / "delta.parquet"
        if not delta_path.exists():
            print(f"[WARN] Missing delta parquet for pair {facade_id} {year_a}_{year_b}")
            continue

        df = pd.read_parquet(delta_path)
        coverage_col = pick_column(df, args.coverage_column, DEFAULT_COVERAGE_CANDIDATES)
        if coverage_col is None:
            print(f"[WARN] No coverage column found for pair {facade_id} {year_a}_{year_b}")
            continue

        delta_col = pick_column(df, args.delta_column, DEFAULT_DELTA_CANDIDATES)
        if delta_col is None:
            print(f"[WARN] No delta column found for pair {facade_id} {year_a}_{year_b}")
            continue

        area_col = pick_column(df, None, DEFAULT_AREA_CANDIDATES)

        stats = aggregate_pair(
            df,
            coverage_col=coverage_col,
            delta_col=delta_col,
            area_col=area_col,
            coverage_threshold=args.coverage_threshold,
            delta_threshold=args.delta_threshold,
        )

        if args.min_coverage_median is not None:
            med = stats.get("coverage_median", float("nan"))
            if np.isnan(med) or med < args.min_coverage_median:
                continue

        rows.append(
            {
                "facade_id": facade_id,
                "year_prev": year_prev,
                "year_next": year_next,
                "dt_years": dt_years,
                "quality": quality,
                "status": status,
                **stats,
            }
        )

    features = pd.DataFrame(rows)
    if not features.empty:
        features.sort_values(["facade_id", "year_prev", "year_next"], inplace=True)
    return features


def build_index(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(
            columns=["facade_id", "n_steps", "min_year", "max_year", "pct_steps_ok", "usable_for_temporal_model"]
        )

    def summarize_group(df: pd.DataFrame) -> Dict[str, object]:
        n_steps = len(df)
        min_year = int(df["year_prev"].min())
        max_year = int(df["year_next"].max())
        ok_mask = df["n_spx_valid"] > 0
        pct_steps_ok = float(ok_mask.mean()) if n_steps else float("nan")
        usable = n_steps >= 2 and pct_steps_ok >= 0.7
        return {
            "n_steps": n_steps,
            "min_year": min_year,
            "max_year": max_year,
            "pct_steps_ok": pct_steps_ok,
            "usable_for_temporal_model": usable,
        }

    grouped = features.groupby("facade_id", as_index=False).apply(lambda g: pd.Series(summarize_group(g)))
    grouped.reset_index(drop=True, inplace=True)
    return grouped


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = build_features(args)
    features_path = args.out_dir / "timeseries_features.parquet"
    features.to_parquet(features_path, index=False)
    print(f"[OK] Saved time-series features to {features_path}")

    index_df = build_index(features)
    index_path = args.out_dir / "timeseries_index.csv"
    index_df.to_csv(index_path, index=False)
    print(f"[OK] Saved time-series index to {index_path}")


if __name__ == "__main__":
    main()
