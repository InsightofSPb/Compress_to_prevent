import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tools.ref_spx_features import load_manifest_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch orchestrator for ref-SPX pipeline across facade pairs.",
    )

    parser.add_argument("--temporal-manifest", type=Path, required=True, help="CSV with facade_id, year, image_path")
    parser.add_argument("--pairs", type=Path, required=True, help="CSV with facade_id, year_a, year_b")
    parser.add_argument("--geom-dir", type=Path, required=True, help="Directory containing geom/<facade_id>/<year_a>_<year_b>.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="Root output directory")

    # ref_spx_features flags
    parser.add_argument("--n-segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=10.0)

    # ref_spx_change_map flags
    parser.add_argument("--coverage-threshold", type=float, default=0.85)
    parser.add_argument("--include-std", action="store_true")
    parser.add_argument("--std-weight", type=float, default=0.3)
    parser.add_argument("--exclude-top-pct", type=float, default=0.20)
    parser.add_argument("--exclude-bottom-pct", type=float, default=0.08)
    parser.add_argument("--exclude-border-px", type=int, default=10)
    parser.add_argument("--min-ref-std", type=float, default=8.0)
    parser.add_argument("--disable-sky-filter", action="store_true")
    parser.add_argument("--sky-h-low", type=int, default=80)
    parser.add_argument("--sky-h-high", type=int, default=140)
    parser.add_argument("--sky-s-min", type=int, default=25)
    parser.add_argument("--sky-v-min", type=int, default=140)
    parser.add_argument("--metric", type=str, default="lab_ab_aligned")
    parser.add_argument("--global-color-align", action="store_true")
    parser.add_argument("--norm-pct-low", type=float, default=2.0)
    parser.add_argument("--norm-pct-high", type=float, default=98.0)
    parser.add_argument("--heat-alpha", type=float, default=0.6)
    parser.add_argument("--gallery-k", type=int, default=30)
    parser.add_argument("--gallery-pad", type=int, default=12)
    parser.add_argument("--gallery-tile", type=int, default=224)
    parser.add_argument("--gallery-gap", type=int, default=10)
    parser.add_argument("--gallery-margin", type=int, default=10)

    return parser.parse_args()


def read_geom(geom_path: Path) -> Tuple[str, np.ndarray]:
    with geom_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    quality = data.get("quality") or data.get("status_quality") or ""
    H = data.get("H")
    if H is not None:
        H = np.array(H, dtype=float)
        if H.shape != (3, 3):
            H = None
    return quality, H


def ensure_geom(geom_root: Path, facade_id: str, year_a: int, year_b: int) -> Tuple[Path, str]:
    geom_path = geom_root / str(facade_id) / f"{year_a}_{year_b}.json"
    if not geom_path.exists():
        return geom_path, "geom_missing"

    quality, H = read_geom(geom_path)
    if quality not in {"weak", "strong"}:
        return geom_path, f"bad_quality:{quality}"
    if H is None:
        return geom_path, "no_H"
    return geom_path, "ok"


def run_ref_spx_features(
    args: argparse.Namespace,
    pair_dir: Path,
    geom_json: Path,
    facade_id: str,
    ref_year: int,
    src_year: int,
) -> Tuple[bool, str, Path]:
    pair_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path("tools") / "ref_spx_features.py"),
        "--temporal-manifest",
        str(args.temporal_manifest),
        "--geom-json",
        str(geom_json),
        "--facade-id",
        str(facade_id),
        "--ref-year",
        str(ref_year),
        "--src-year",
        str(src_year),
        "--out-dir",
        str(pair_dir),
        "--n-segments",
        str(args.n_segments),
        "--compactness",
        str(args.compactness),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        reason = f"ref_spx_features_failed:{result.stderr.strip() or result.stdout.strip()}"
        return False, reason, pair_dir

    base = pair_dir / "facades" / str(facade_id) / "ref_spx" / str(ref_year)
    return True, "ok", base


def run_ref_spx_change_map(
    args: argparse.Namespace,
    pair_dir: Path,
    base_dir: Path,
    facade_id: str,
    ref_year: int,
    src_year: int,
    ref_image_path: Path,
) -> Tuple[bool, str]:
    ref_features = base_dir / f"features_{ref_year}.parquet"
    src_features = base_dir / f"features_{src_year}_warped.parquet"
    ref_labels = base_dir / "ref_labels.npz"
    src_warp = base_dir / f"src_warp_{src_year}_to_{ref_year}.png"

    if not (ref_features.exists() and src_features.exists() and ref_labels.exists() and src_warp.exists()):
        return False, "missing_inputs_for_change_map"

    cmd = [
        sys.executable,
        str(Path("tools") / "ref_spx_change_map.py"),
        "--ref-features",
        str(ref_features),
        "--src-features",
        str(src_features),
        "--ref-labels",
        str(ref_labels),
        "--ref-image",
        str(ref_image_path),
        "--out",
        str(pair_dir),
        "--coverage-threshold",
        str(args.coverage_threshold),
        "--std-weight",
        str(args.std_weight),
        "--exclude-top-pct",
        str(args.exclude_top_pct),
        "--exclude-bottom-pct",
        str(args.exclude_bottom_pct),
        "--exclude-border-px",
        str(args.exclude_border_px),
        "--min-ref-std",
        str(args.min_ref_std),
        "--sky-h-low",
        str(args.sky_h_low),
        "--sky-h-high",
        str(args.sky_h_high),
        "--sky-s-min",
        str(args.sky_s_min),
        "--sky-v-min",
        str(args.sky_v_min),
        "--metric",
        str(args.metric),
        "--norm-pct-low",
        str(args.norm_pct_low),
        "--norm-pct-high",
        str(args.norm_pct_high),
        "--heat-alpha",
        str(args.heat_alpha),
        "--gallery-k",
        str(args.gallery_k),
        "--gallery-pad",
        str(args.gallery_pad),
        "--gallery-tile",
        str(args.gallery_tile),
        "--gallery-gap",
        str(args.gallery_gap),
        "--gallery-margin",
        str(args.gallery_margin),
    ]

    if args.include_std:
        cmd.append("--include-std")
    if args.disable_sky_filter:
        cmd.append("--disable-sky-filter")
    if args.global_color_align:
        cmd.append("--global-color-align")
    cmd.extend(["--src-warp-image", str(src_warp)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        reason = f"ref_spx_change_map_failed:{result.stderr.strip() or result.stdout.strip()}"
        return False, reason

    return True, "ok"


def collect_stats(pair_dir: Path, base_dir: Path, metric: str) -> Tuple[int, int, float, float, float]:
    delta_path = pair_dir / "delta.parquet"
    labels_path = base_dir / "ref_labels.npz"

    if not delta_path.exists() or not labels_path.exists():
        return 0, 0, float("nan"), float("nan"), float("nan")

    df = pd.read_parquet(delta_path)
    vals = df["delta_main"].dropna().to_numpy()
    num_valid = int(vals.size)
    median = float(np.median(vals)) if num_valid else float("nan")
    p95 = float(np.percentile(vals, 95)) if num_valid else float("nan")
    max_v = float(np.max(vals)) if num_valid else float("nan")

    labels = np.load(labels_path)["labels"]
    num_labels = int(np.unique(labels).size)

    return num_labels, num_valid, median, p95, max_v


def main() -> None:
    args = parse_args()
    pairs_df = pd.read_csv(args.pairs)
    required_cols = {"facade_id", "year_a", "year_b"}
    if not required_cols.issubset(pairs_df.columns):
        missing = required_cols - set(pairs_df.columns)
        raise ValueError(f"Pairs CSV missing columns: {missing}")

    summary_rows: List[Dict[str, object]] = []
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for _, row in pairs_df.iterrows():
        facade_id = row["facade_id"]
        year_a = int(row["year_a"])
        year_b = int(row["year_b"])

        if year_a > year_b:
            year_a, year_b = year_b, year_a

        year_prev, year_next = year_a, year_b
        ref_year, src_year = year_prev, year_next
        dt_years = src_year - ref_year

        pair_dir = out_root / "pairs" / str(facade_id) / f"{ref_year}_{src_year}"

        geom_path, status = ensure_geom(args.geom_dir, str(facade_id), ref_year, src_year)
        quality = ""
        if status != "ok":
            summary_rows.append(
                {
                    "facade_id": facade_id,
                    "year_prev": year_prev,
                    "year_next": year_next,
                    "dt_years": dt_years,
                    "quality": quality,
                    "num_labels": 0,
                    "num_valid": 0,
                    "median_delta": np.nan,
                    "p95_delta": np.nan,
                    "max_delta": np.nan,
                    "gallery_path": "",
                    "delta_heatmap_path": "",
                    "status": "skipped",
                    "skip_reason": status,
                }
            )
            continue
        quality, _ = read_geom(geom_path)

        try:
            ref_img_path, _ = load_manifest_image(
                args.temporal_manifest, str(facade_id), ref_year
            )
        except Exception as e:  # noqa: BLE001
            summary_rows.append(
                {
                    "facade_id": facade_id,
                    "year_prev": year_prev,
                    "year_next": year_next,
                    "dt_years": dt_years,
                    "quality": quality,
                    "num_labels": 0,
                    "num_valid": 0,
                    "median_delta": np.nan,
                    "p95_delta": np.nan,
                    "max_delta": np.nan,
                    "gallery_path": "",
                    "delta_heatmap_path": "",
                    "status": "skipped",
                    "skip_reason": f"manifest_error:{e}",
                }
            )
            continue

        ok_features, reason, base_dir = run_ref_spx_features(
            args=args,
            pair_dir=pair_dir,
            geom_json=geom_path,
            facade_id=str(facade_id),
            ref_year=ref_year,
            src_year=src_year,
        )
        if not ok_features:
            summary_rows.append(
                {
                    "facade_id": facade_id,
                    "year_prev": year_prev,
                    "year_next": year_next,
                    "dt_years": dt_years,
                    "quality": quality,
                    "num_labels": 0,
                    "num_valid": 0,
                    "median_delta": np.nan,
                    "p95_delta": np.nan,
                    "max_delta": np.nan,
                    "gallery_path": "",
                    "delta_heatmap_path": "",
                    "status": "failed",
                    "skip_reason": reason,
                }
            )
            continue

        ok_change, reason_change = run_ref_spx_change_map(
            args=args,
            pair_dir=pair_dir,
            base_dir=base_dir,
            facade_id=str(facade_id),
            ref_year=ref_year,
            src_year=src_year,
            ref_image_path=ref_img_path,
        )
        if not ok_change:
            summary_rows.append(
                {
                    "facade_id": facade_id,
                    "year_prev": year_prev,
                    "year_next": year_next,
                    "dt_years": dt_years,
                    "quality": quality,
                    "num_labels": 0,
                    "num_valid": 0,
                    "median_delta": np.nan,
                    "p95_delta": np.nan,
                    "max_delta": np.nan,
                    "gallery_path": "",
                    "delta_heatmap_path": "",
                    "status": "failed",
                    "skip_reason": reason_change,
                }
            )
            continue

        shutil.copy2(geom_path, pair_dir / geom_path.name)

        num_labels, num_valid, med, p95, max_v = collect_stats(pair_dir, base_dir, args.metric)
        gallery_path = pair_dir / f"gallery_top{args.gallery_k}_{args.metric}.png"
        delta_heatmap_path = pair_dir / "delta_heatmap.png"

        summary_rows.append(
            {
                "facade_id": facade_id,
                "year_prev": year_prev,
                "year_next": year_next,
                "dt_years": dt_years,
                "quality": quality,
                "num_labels": num_labels,
                "num_valid": num_valid,
                "median_delta": med,
                "p95_delta": p95,
                "max_delta": max_v,
                "gallery_path": str(gallery_path) if gallery_path.exists() else "",
                "delta_heatmap_path": str(delta_heatmap_path) if delta_heatmap_path.exists() else "",
                "status": "ok",
                "skip_reason": "",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_root / "pairs_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

