import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

DEFAULT_DELTA_CANDIDATES: Sequence[str] = (
    "delta_p95",
    "delta_main",
    "delta_lab_ab_aligned",
    "delta_rgb",
    "delta_lab_ab",
    "delta_rgb_std",
    "delta_lab_all",
)
DEFAULT_COVERAGE_CANDIDATES: Sequence[str] = ("coverage_median", "coverage", "coverage_src")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leaderboard of facades and intervals")
    parser.add_argument("--timeseries-features", required=True, type=Path, help="Path to timeseries_features.parquet")
    parser.add_argument("--pairs-root", required=True, type=Path, help="Root directory with pair folders")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--score", choices=["p95", "area_frac", "combo"], default="combo")
    parser.add_argument("--min-steps", type=int, default=1, help="Minimum number of intervals per facade")
    parser.add_argument("--min-coverage-median", type=float, default=None, help="Filter intervals by median coverage")
    parser.add_argument("--top-k-facades", type=int, default=20)
    parser.add_argument("--top-k-intervals", type=int, default=50)
    parser.add_argument("--topk-gallery-pattern", type=str, default="gallery_top*.png", help="Glob for gallery images")
    return parser.parse_args()


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_scores(df: pd.DataFrame, score_type: str) -> pd.DataFrame:
    working = df.copy()
    if "dt_years" not in working.columns:
        working["dt_years"] = working["year_next"] - working["year_prev"]

    coverage_col = pick_column(working, DEFAULT_COVERAGE_CANDIDATES)
    if coverage_col is None:
        raise ValueError("Coverage column not found in features")

    if score_type == "p95":
        working["score"] = working["delta_p95"]
    elif score_type == "area_frac":
        if "area_frac_delta_gt_T" in working.columns:
            frac_col = "area_frac_delta_gt_T"
        elif "frac_delta_gt_T" in working.columns:
            frac_col = "frac_delta_gt_T"
        else:
            raise ValueError("No fraction delta columns available for scoring")
        working["score"] = working[frac_col]
    else:
        if "frac_delta_gt_T" not in working.columns:
            raise ValueError("frac_delta_gt_T is required for combo score")
        working["score"] = (
            working["delta_p95"]
            * (working["frac_delta_gt_T"].fillna(0.0).add(1e-6).pow(0.5))
            * working[coverage_col].fillna(0.0).map(clamp01)
        )

    working["score_per_year"] = working["score"] / working["dt_years"].clip(lower=1)
    return working


def summarize_facades(df: pd.DataFrame, min_steps: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["facade_id", "max_score_per_year", "mean_score_per_year", "n_steps", "n_steps_ok", "last_year"]
        )

    def _agg(group: pd.DataFrame) -> Dict[str, object]:
        n_steps = len(group)
        quality = group.get("quality", pd.Series(dtype=object))
        n_steps_ok = int((quality.str.lower() == "strong").sum()) if not quality.empty else 0
        return {
            "max_score_per_year": float(group["score_per_year"].max()),
            "mean_score_per_year": float(group["score_per_year"].mean()),
            "n_steps": int(n_steps),
            "n_steps_ok": n_steps_ok,
            "last_year": int(group["year_next"].max()),
        }

    grouped = df.groupby("facade_id", as_index=False).apply(lambda g: pd.Series(_agg(g)))
    grouped.reset_index(drop=True, inplace=True)
    filtered = grouped[grouped["n_steps"] >= min_steps].copy()
    filtered.sort_values(["max_score_per_year", "mean_score_per_year"], ascending=False, inplace=True)
    return filtered


def locate_pair_dir(pairs_root: Path, facade_id: str, year_prev: int, year_next: int) -> Optional[Path]:
    candidates = [f"{year_prev}_{year_next}", f"{year_next}_{year_prev}"]
    facade_dir = pairs_root / str(facade_id)
    for name in candidates:
        candidate = facade_dir / name
        if candidate.exists():
            return candidate
    return None


def compute_sky_flag(pair_dir: Path, delta_col_candidates: Sequence[str]) -> Optional[bool]:
    delta_path = pair_dir / "delta_full.parquet"
    if not delta_path.exists():
        delta_path = pair_dir / "delta.parquet"
    if not delta_path.exists():
        return None

    df = pd.read_parquet(delta_path)
    delta_col = pick_column(df, delta_col_candidates)
    if delta_col is None or delta_col not in df.columns:
        return None
    if "cy_norm" in df.columns:
        cy_norm = df["cy_norm"]
    elif "cy" in df.columns and "H" in df.columns:
        cy_norm = df["cy"] / df["H"].replace(0, pd.NA)
    elif "cy" in df.columns:
        max_cy = df["cy"].max()
        if pd.isna(max_cy) or max_cy <= 0:
            return None
        cy_norm = df["cy"] / float(max_cy)
    else:
        return None

    if cy_norm.isna().all():
        return None

    delta_vals = df[delta_col]
    valid_mask = ~(delta_vals.isna() | cy_norm.isna())
    df_valid = df.loc[valid_mask].copy()
    if df_valid.empty:
        return None

    df_valid["cy_norm"] = cy_norm.loc[df_valid.index]
    n_top = max(1, int(len(df_valid) * 0.1))
    top_changes = df_valid.nlargest(n_top, columns=[delta_col])
    share_top_sky = (top_changes["cy_norm"] < 0.25).mean()
    return bool(share_top_sky > 0.6)


def copy_debug_artifacts(pair_dir: Path, debug_dir: Path, gallery_pattern: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name in ["delta_heatmap.png", "coverage_heatmap.png", "top_changed.csv", "ref_overlay.png"]:
        src = pair_dir / name
        if src.exists():
            shutil.copy(src, debug_dir / name)

    for overlay in pair_dir.glob("warped_overlay_*.png"):
        shutil.copy(overlay, debug_dir / overlay.name)
    for gallery in pair_dir.glob(gallery_pattern):
        shutil.copy(gallery, debug_dir / gallery.name)


def add_flags(df: pd.DataFrame, pairs_root: Path, top_k: int) -> pd.DataFrame:
    if df.empty:
        return df

    delta_cols = list(DEFAULT_DELTA_CANDIDATES)

    records: List[pd.Series] = []
    for rank, (_, row) in enumerate(df.head(top_k).iterrows(), start=1):
        coverage_p10 = row.get("coverage_p10", float("nan"))
        coverage_median = row.get("coverage_median", float("nan"))
        n_valid_spx = row.get("n_spx_valid", float("nan"))

        flag_low_coverage = bool(
            (not math.isnan(coverage_p10) and coverage_p10 < 0.6)
            or (not math.isnan(coverage_median) and coverage_median < 0.8)
        )
        flag_too_sparse = bool(not math.isnan(n_valid_spx) and n_valid_spx < 50)

        pair_dir = locate_pair_dir(pairs_root, row["facade_id"], int(row["year_prev"]), int(row["year_next"]))
        flag_sky_dominated: Optional[bool] = None
        if pair_dir is not None:
            flag_sky_dominated = compute_sky_flag(pair_dir, delta_cols)

        n_flags = sum(flag is True for flag in [flag_low_coverage, flag_too_sparse, flag_sky_dominated])
        if n_flags == 0:
            human_priority = "HIGH"
        elif n_flags == 1:
            human_priority = "MED"
        else:
            human_priority = "LOW"

        rec = row.copy()
        rec["rank"] = rank
        rec["flag_low_coverage"] = flag_low_coverage
        rec["flag_too_sparse"] = flag_too_sparse
        rec["flag_sky_dominated"] = flag_sky_dominated
        rec["human_priority"] = human_priority
        rec["pair_dir"] = str(pair_dir) if pair_dir is not None else ""
        records.append(rec)

    result = pd.DataFrame(records)
    result.sort_values("rank", inplace=True)
    return result


def build_leaderboards(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.timeseries_features)
    if args.min_coverage_median is not None and "coverage_median" in df.columns:
        df = df[df["coverage_median"] >= args.min_coverage_median].copy()

    scored = compute_scores(df, args.score)

    leaderboard_facades = summarize_facades(scored, args.min_steps)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    facades_path = args.out_dir / "leaderboard_facades.csv"
    leaderboard_facades.head(args.top_k_facades).to_csv(facades_path, index=False)

    intervals_sorted = scored.sort_values("score_per_year", ascending=False)
    top_intervals = intervals_sorted.head(args.top_k_intervals).copy()

    top_intervals_with_flags = add_flags(top_intervals, args.pairs_root, args.top_k_intervals)

    debug_base = args.out_dir / "debug_top_intervals"
    debug_base.mkdir(parents=True, exist_ok=True)
    debug_dirs: List[str] = []
    for _, row in top_intervals_with_flags.iterrows():
        pair_dir = Path(row["pair_dir"]) if row.get("pair_dir") else None
        if pair_dir is None or not pair_dir.exists():
            debug_dirs.append("")
            continue

        target_dir = debug_base / f"{int(row['rank']):03d}_{row['facade_id']}_{int(row['year_prev'])}_{int(row['year_next'])}"
        copy_debug_artifacts(pair_dir, target_dir, args.topk_gallery_pattern)
        debug_dirs.append(str(target_dir))

    top_intervals_with_flags["debug_dir"] = debug_dirs

    intervals_path = args.out_dir / "leaderboard_intervals.csv"
    top_intervals_with_flags.to_csv(intervals_path, index=False)

    print(f"[OK] Saved facade leaderboard to {facades_path}")
    print(f"[OK] Saved interval leaderboard to {intervals_path}")
    print(f"[OK] Debug artifacts saved under {debug_base}")


def main() -> None:
    args = parse_args()
    build_leaderboards(args)


if __name__ == "__main__":
    main()
