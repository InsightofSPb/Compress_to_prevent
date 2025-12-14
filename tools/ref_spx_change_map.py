import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute simple change map between reference and warped features.")
    parser.add_argument("--ref-features", required=True, type=Path, help="Path to reference features parquet (features_{ref_year}.parquet)")
    parser.add_argument(
        "--src-features",
        required=True,
        type=Path,
        help="Path to warped source features parquet (features_{src_year}_warped.parquet)",
    )
    parser.add_argument("--ref-labels", required=True, type=Path, help="Path to reference labels npz file")
    parser.add_argument("--ref-image", required=True, type=Path, help="Path to reference image")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.6,
        help="Minimum coverage of warped superpixel to consider for change metric",
    )
    parser.add_argument("--include-std", action="store_true", help="Include std deviation term in delta")
    parser.add_argument(
        "--std-weight",
        type=float,
        default=0.3,
        help="Weight for std deviation component when --include-std is enabled",
    )
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_labels(labels_path: Path) -> np.ndarray:
    data = np.load(labels_path)
    if "labels" not in data:
        raise KeyError(f"`labels` key not found in {labels_path}")
    return data["labels"]


def compute_deltas(ref_df: pd.DataFrame, src_df: pd.DataFrame) -> pd.DataFrame:
    merged = ref_df.merge(src_df, on="ref_label_id", suffixes=("_ref", "_src"))

    delta_rgb = np.sqrt(
        (merged["mean_r_ref"] - merged["mean_r_src"]) ** 2
        + (merged["mean_g_ref"] - merged["mean_g_src"]) ** 2
        + (merged["mean_b_ref"] - merged["mean_b_src"]) ** 2
    )

    delta_std = np.sqrt(
        (merged["std_r_ref"] - merged["std_r_src"]) ** 2
        + (merged["std_g_ref"] - merged["std_g_src"]) ** 2
        + (merged["std_b_ref"] - merged["std_b_src"]) ** 2
    )

    merged["delta_rgb"] = delta_rgb
    merged["delta_std"] = delta_std
    return merged


def normalize_values(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.zeros_like(values), 0.0, 0.0
    valid_values = values[valid_mask]
    vmin, vmax = float(valid_values.min()), float(valid_values.max())
    if vmax - vmin < 1e-8:
        norm = np.zeros_like(values)
    else:
        norm = (values - vmin) / (vmax - vmin)
    norm[~valid_mask] = 0.0
    return norm, vmin, vmax


def build_heatmap_image(labels: np.ndarray, ref_image: np.ndarray, delta_map: Dict[int, float]) -> np.ndarray:
    heat_values = np.full(labels.shape, np.nan, dtype=float)
    for lbl, delta in delta_map.items():
        heat_values[labels == lbl] = delta

    norm_values, _, _ = normalize_values(heat_values)
    cmap = plt.get_cmap("inferno")
    heat_colors = cmap(norm_values)[..., :3]  # drop alpha channel

    ref_float = ref_image.astype(np.float32) / 255.0
    result = ref_float.copy()
    valid_mask = ~np.isnan(heat_values)
    result[valid_mask] = 0.4 * ref_float[valid_mask] + 0.6 * heat_colors[valid_mask]

    return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)


def save_histogram(values: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values.dropna(), bins=30, color="steelblue", edgecolor="black")
    plt.xlabel("Delta")
    plt.ylabel("Count")
    plt.title("Distribution of delta")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()

    ref_df = pd.read_parquet(args.ref_features)
    src_df = pd.read_parquet(args.src_features)

    merged = compute_deltas(ref_df, src_df)
    merged["coverage"] = merged["coverage_src"]

    coverage_mask = merged["coverage"] >= args.coverage_threshold
    filtered = merged[coverage_mask].copy()

    delta = filtered["delta_rgb"].copy()
    if args.include_std:
        delta += args.std_weight * filtered["delta_std"]
    filtered["delta"] = delta

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    save_columns = [
        "ref_label_id",
        "coverage",
        "delta",
        "delta_rgb",
        "delta_std",
        "mean_r_ref",
        "mean_g_ref",
        "mean_b_ref",
        "mean_r_src",
        "mean_g_src",
        "mean_b_src",
    ]
    filtered.to_parquet(out_dir / "delta.parquet", index=False, columns=save_columns)

    top_changed = filtered.sort_values("delta", ascending=False).head(50)
    top_changed[["ref_label_id", "coverage", "delta", "delta_rgb", "delta_std"]].to_csv(
        out_dir / "top_changed.csv", index=False
    )

    labels = load_labels(args.ref_labels)
    ref_image = load_image(args.ref_image)

    delta_map = dict(zip(filtered["ref_label_id"], filtered["delta"]))
    heatmap_img = build_heatmap_image(labels, ref_image, delta_map)
    cv2.imwrite(str(out_dir / "delta_heatmap.png"), cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))

    save_histogram(filtered["delta"], out_dir / "delta_hist.png")


if __name__ == "__main__":
    main()
