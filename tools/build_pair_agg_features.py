import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)


DEFAULT_SPX_FEATURES = (
    "/home/sasha/LPOSS/datasets/SPb_facades/facades_with_years/"
    "compress_features_lposs/spx_features.parquet"
)
DEFAULT_PAIR_FEATURES = (
    "/home/sasha/LPOSS/datasets/SPb_facades/facades_with_years/"
    "compress_features_lposs/pair_features.parquet"
)
DEFAULT_OUT = (
    "/home/sasha/LPOSS/datasets/SPb_facades/facades_with_years/"
    "compress_features_lposs/pair_agg.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate pair-level features with LPOSS per-year statistics."
    )
    parser.add_argument("--spx-features", type=Path, default=Path(DEFAULT_SPX_FEATURES))
    parser.add_argument("--pair-features", type=Path, default=Path(DEFAULT_PAIR_FEATURES))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument(
        "--only-strong",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only strong-quality pair rows.",
    )
    return parser.parse_args()


def _safe_percentile(values: Iterable[object], q: float) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
    array = array[~np.isnan(array)]
    if array.size == 0:
        return float("nan")
    return float(np.percentile(array, q))


def _add_agg(
    agg_defs: Dict[str, Tuple[str, Callable[[pd.Series], float]]],
    df: pd.DataFrame,
    source_col: str,
    out_col: str,
    func: Callable[[pd.Series], float],
) -> None:
    if source_col in df.columns:
        agg_defs[out_col] = (source_col, func)
    else:
        LOGGER.warning("Missing column for aggregation: %s", source_col)


def _prepare_spx_features(spx: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "year" in spx.columns and "year_b" not in spx.columns:
        rename_map["year"] = "year_b"
    if "obj_id" in spx.columns and "obj_id_b" not in spx.columns:
        rename_map["obj_id"] = "obj_id_b"
    if rename_map:
        spx = spx.rename(columns=rename_map)
    return spx


def build_aggregates(merged: pd.DataFrame) -> pd.DataFrame:
    keys = ["facade_id", "year_a", "year_b"]
    grouped = merged.groupby(keys, dropna=False)

    size_df = grouped.size().rename("n_obj").to_frame()

    agg_defs: Dict[str, Tuple[str, Callable[[pd.Series], float]]] = {}

    _add_agg(agg_defs, merged, "support_area_ratio", "support_area_ratio_mean", "mean")
    _add_agg(agg_defs, merged, "support_area_ratio", "support_area_ratio_p10", lambda s: _safe_percentile(s, 10))

    _add_agg(agg_defs, merged, "delta_bpp", "delta_bpp_mean", "mean")
    _add_agg(agg_defs, merged, "delta_bpp", "delta_bpp_median", "median")
    _add_agg(agg_defs, merged, "delta_bpp", "delta_bpp_p75", lambda s: _safe_percentile(s, 75))
    _add_agg(agg_defs, merged, "delta_bpp", "delta_bpp_p95", lambda s: _safe_percentile(s, 95))

    _add_agg(agg_defs, merged, "delta_bpp_rel", "delta_bpp_rel_mean", "mean")
    _add_agg(agg_defs, merged, "delta_bpp_rel", "delta_bpp_rel_p95", lambda s: _safe_percentile(s, 95))

    if "bpp_excess_b" in merged.columns:
        _add_agg(agg_defs, merged, "bpp_excess_b", "bpp_excess_b_p95", lambda s: _safe_percentile(s, 95))

    _add_agg(agg_defs, merged, "p_damage", "p_damage_mean", "mean")
    _add_agg(agg_defs, merged, "p_damage", "p_damage_p95", lambda s: _safe_percentile(s, 95))

    _add_agg(agg_defs, merged, "entropy_norm_mean", "entropy_norm_mean_mean", "mean")
    _add_agg(agg_defs, merged, "entropy_norm_mean", "entropy_norm_mean_p95", lambda s: _safe_percentile(s, 95))

    _add_agg(agg_defs, merged, "margin_mean", "margin_mean_mean", "mean")
    _add_agg(agg_defs, merged, "margin_mean", "margin_mean_p10", lambda s: _safe_percentile(s, 10))

    optional_mean_p = [
        "mean_p_REPAIRS",
        "mean_p_TEXT_OR_IMAGES",
        "mean_p_ORNAMENT_INTACT",
    ]
    for col in optional_mean_p:
        if col in merged.columns:
            _add_agg(agg_defs, merged, col, f"{col}_mean", "mean")

    agg_df = grouped.agg(**agg_defs) if agg_defs else pd.DataFrame(index=grouped.size().index)

    result = size_df.join(agg_df)
    result = result.reset_index()
    return result


def main() -> None:
    args = parse_args()

    pair_df = pd.read_parquet(args.pair_features)
    if args.only_strong and "status_quality" in pair_df.columns:
        pair_df = pair_df[pair_df["status_quality"] == "strong"].copy()

    spx_df = pd.read_parquet(args.spx_features)
    spx_df = _prepare_spx_features(spx_df)

    required_keys = {"facade_id", "year_b", "obj_id_b"}
    if not required_keys.issubset(pair_df.columns):
        missing = required_keys - set(pair_df.columns)
        raise ValueError(f"pair_features missing required columns: {sorted(missing)}")
    if not required_keys.issubset(spx_df.columns):
        missing = required_keys - set(spx_df.columns)
        raise ValueError(f"spx_features missing required columns: {sorted(missing)}")

    merged = pair_df.merge(spx_df, on=["facade_id", "year_b", "obj_id_b"], how="left", suffixes=("", "_spx"))
    agg_df = build_aggregates(merged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_parquet(args.out, index=False)
    preview_path = args.out.with_name("pair_agg_preview.csv")
    agg_df.head(200).to_csv(preview_path, index=False)

    LOGGER.info("Saved pair aggregates to %s", args.out)
    LOGGER.info("Saved preview CSV to %s", preview_path)


if __name__ == "__main__":
    main()
