import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute change maps between reference and warped features + optional top-K gallery."
    )

    parser.add_argument("--ref-features", required=True, type=Path, help="features_{ref_year}.parquet")
    parser.add_argument("--src-features", required=True, type=Path, help="features_{src_year}_warped.parquet")
    parser.add_argument("--ref-labels", required=True, type=Path, help="ref_labels.npz (key: labels)")
    parser.add_argument("--ref-image", required=True, type=Path, help="reference image path")
    parser.add_argument("--out", required=True, type=Path, help="output directory")

    # Coverage / std
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.85,
        help="Minimum coverage of warped superpixel to consider for delta_main (others become NaN)",
    )
    parser.add_argument("--include-std", action="store_true", help="Include std deviation term in RGB delta")
    parser.add_argument("--std-weight", type=float, default=0.3, help="Weight for std term when --include-std is enabled")

    # Metrics
    parser.add_argument(
        "--metric",
        type=str,
        default="lab_ab_aligned",
        choices=["rgb", "rgb_std", "lab_all", "lab_ab", "lab_ab_aligned"],
        help="Which delta metric to use as delta_main",
    )
    parser.add_argument(
        "--global-color-align",
        action="store_true",
        help="Align src Lab a,b globally to ref (median shift on valid superpixels). "
             "Used for lab_ab_aligned and can also help others.",
    )

    # Heatmap normalization
    parser.add_argument("--norm-pct-low", type=float, default=2.0, help="Low percentile for heatmap normalization")
    parser.add_argument("--norm-pct-high", type=float, default=98.0, help="High percentile for heatmap normalization")
    parser.add_argument("--heat-alpha", type=float, default=0.6, help="Alpha for heat overlay blending [0..1]")

    # Optional gallery
    parser.add_argument(
        "--src-warp-image",
        type=Path,
        default=None,
        help="Path to src image already warped into ref frame (same HxW as ref). "
             "If provided, script will build a top-K gallery (ref vs warped).",
    )
    parser.add_argument("--gallery-k", type=int, default=10, help="How many top superpixels to show in gallery")
    parser.add_argument("--gallery-pad", type=int, default=12, help="Padding (px) around superpixel bbox for crops")
    parser.add_argument("--gallery-tile", type=int, default=224, help="Tile size (px) for each crop in gallery")
    parser.add_argument("--gallery-gap", type=int, default=10, help="Gap (px) between ref and warped tiles")
    parser.add_argument("--gallery-margin", type=int, default=10, help="Outer margin for gallery image")

    return parser.parse_args()


# -----------------------------
# IO utils
# -----------------------------
def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_labels_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    if "labels" not in data:
        raise KeyError(f"`labels` key not found in {path}")
    return data["labels"].astype(np.int32)


# -----------------------------
# Color / metric utils
# -----------------------------
def rgb_means_to_lab(means_rgb_0_255: np.ndarray) -> np.ndarray:
    """
    Convert Nx3 RGB means (0..255 floats) to Nx3 Lab (OpenCV: L,a,b in 0..255).
    Robust to NaN/inf.
    """
    means = np.array(means_rgb_0_255, dtype=np.float32)
    means = np.nan_to_num(means, nan=0.0, posinf=255.0, neginf=0.0)
    means_u8 = np.clip(means, 0, 255).astype(np.uint8)

    img = means_u8.reshape(1, -1, 3)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return lab


def compute_metrics(
    merged: pd.DataFrame,
    include_std: bool,
    std_weight: float,
    do_align: bool,
    cov_thr: float,
) -> pd.DataFrame:
    """
    Adds:
      delta_rgb, delta_std, delta_rgb_std,
      mean_L_*, mean_a_*, mean_b_*_lab,
      delta_lab_all, delta_lab_ab, delta_lab_ab_aligned
    """
    # RGB deltas
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
    merged["delta_rgb_std"] = delta_rgb + (std_weight * delta_std if include_std else 0.0)

    # Lab deltas from mean RGB
    ref_means = merged[["mean_r_ref", "mean_g_ref", "mean_b_ref"]].to_numpy(dtype=np.float32)
    src_means = merged[["mean_r_src", "mean_g_src", "mean_b_src"]].to_numpy(dtype=np.float32)

    ref_lab = rgb_means_to_lab(ref_means)
    src_lab = rgb_means_to_lab(src_means)

    merged["mean_L_ref"] = ref_lab[:, 0]
    merged["mean_a_ref"] = ref_lab[:, 1]
    merged["mean_b_ref_lab"] = ref_lab[:, 2]

    merged["mean_L_src"] = src_lab[:, 0]
    merged["mean_a_src"] = src_lab[:, 1]
    merged["mean_b_src_lab"] = src_lab[:, 2]

    dL = merged["mean_L_ref"] - merged["mean_L_src"]
    da = merged["mean_a_ref"] - merged["mean_a_src"]
    db = merged["mean_b_ref_lab"] - merged["mean_b_src_lab"]

    merged["delta_lab_all"] = np.sqrt(dL ** 2 + da ** 2 + db ** 2)
    merged["delta_lab_ab"] = np.sqrt(da ** 2 + db ** 2)

    # Aligned AB (median shift on valid superpixels)
    merged["delta_lab_ab_aligned"] = merged["delta_lab_ab"]
    if do_align:
        valid = merged["coverage"] >= cov_thr
        if valid.any():
            shift_a = np.median((merged.loc[valid, "mean_a_ref"] - merged.loc[valid, "mean_a_src"]).to_numpy())
            shift_b = np.median((merged.loc[valid, "mean_b_ref_lab"] - merged.loc[valid, "mean_b_src_lab"]).to_numpy())

            a_src_al = merged["mean_a_src"] + shift_a
            b_src_al = merged["mean_b_src_lab"] + shift_b

            da_al = merged["mean_a_ref"] - a_src_al
            db_al = merged["mean_b_ref_lab"] - b_src_al

            merged["delta_lab_ab_aligned"] = np.sqrt(da_al ** 2 + db_al ** 2)

    return merged


# -----------------------------
# Heatmaps
# -----------------------------
def normalize_values(values: np.ndarray, pct_low: float, pct_high: float) -> Tuple[np.ndarray, float, float]:
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.zeros_like(values), 0.0, 0.0

    v = values[valid_mask]
    vmin = float(np.percentile(v, pct_low))
    vmax = float(np.percentile(v, pct_high))
    if vmax - vmin < 1e-8:
        norm = np.zeros_like(values)
    else:
        norm = (values - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    norm[~valid_mask] = 0.0
    return norm, vmin, vmax


def build_heatmap_image(
    labels: np.ndarray,
    ref_image: np.ndarray,
    value_map: Dict[int, float],
    pct_low: float,
    pct_high: float,
    alpha_heat: float,
    cmap_name: str = "inferno",
) -> np.ndarray:
    heat_values = np.full(labels.shape, np.nan, dtype=np.float32)
    for lbl, v in value_map.items():
        heat_values[labels == int(lbl)] = float(v)

    norm, _, _ = normalize_values(heat_values, pct_low=pct_low, pct_high=pct_high)
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(norm)[..., :3]  # (H,W,3) float in [0,1]

    ref_float = ref_image.astype(np.float32) / 255.0
    out = ref_float.copy()

    valid = ~np.isnan(heat_values)
    out[valid] = (1.0 - alpha_heat) * ref_float[valid] + alpha_heat * heat_rgb[valid]

    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def save_histogram(values: pd.Series, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    v = values.dropna().to_numpy()
    plt.hist(v, bins=40)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Gallery
# -----------------------------
def bbox_from_labels(labels: np.ndarray, lbl: int) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(labels == lbl)
    if ys.size == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return x1, y1, x2, y2


def resize_and_pad(img: np.ndarray, tile: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((tile, tile, 3), dtype=np.uint8)

    scale = min(tile / h, tile / w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    out = np.zeros((tile, tile, 3), dtype=np.uint8)
    y0 = (tile - nh) // 2
    x0 = (tile - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def build_top_gallery(
    labels: np.ndarray,
    ref_img: np.ndarray,
    src_warp_img: np.ndarray,
    top_df: pd.DataFrame,
    out_path: Path,
    tile: int,
    pad: int,
    k: int,
    gap: int,
    margin: int,
    metric_name: str,
) -> None:
    df = top_df.head(k).copy()
    rows: List[np.ndarray] = []

    H_img, W_img = labels.shape[:2]

    for _, r in df.iterrows():
        lbl = int(r["ref_label_id"])
        delta = float(r["delta_main"])

        bb = bbox_from_labels(labels, lbl)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W_img, x2 + pad)
        y2 = min(H_img, y2 + pad)

        crop_ref = ref_img[y1:y2, x1:x2]
        crop_src = src_warp_img[y1:y2, x1:x2]

        t_ref = resize_and_pad(crop_ref, tile)
        t_src = resize_and_pad(crop_src, tile)

        pair = np.zeros((tile, tile * 2 + gap, 3), dtype=np.uint8)
        pair[:, 0:tile] = t_ref
        pair[:, tile + gap: tile + gap + tile] = t_src

        # annotate
        cv2.putText(pair, f"id={lbl}  d={delta:.2f}", (5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pair, "ref", (5, tile - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pair, "warped", (tile + gap + 5, tile - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        rows.append(pair)

    if not rows:
        return

    # canvas (stack rows)
    title_h = 34
    total_h = title_h + 2 * margin + len(rows) * tile + (len(rows) - 1) * gap
    total_w = 2 * margin + (tile * 2 + gap)

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    title = f"Top-{len(rows)} changes ({metric_name})"
    cv2.putText(canvas, title, (margin, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    y = title_h + margin
    for row_img in rows:
        canvas[y:y + tile, margin:margin + (tile * 2 + gap)] = row_img
        y += tile + gap

    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    ref_df = pd.read_parquet(args.ref_features)
    src_df = pd.read_parquet(args.src_features)

    # Merge on ref_label_id
    merged = ref_df.merge(src_df, on="ref_label_id", suffixes=("_ref", "_src"))

    # coverage from src
    if "coverage_src" not in merged.columns:
        raise KeyError("Expected column `coverage_src` in merged df (from src features).")
    merged["coverage"] = merged["coverage_src"].astype(float)

    # Compute metrics
    merged = compute_metrics(
        merged,
        include_std=args.include_std,
        std_weight=args.std_weight,
        do_align=args.global_color_align,
        cov_thr=args.coverage_threshold,
    )

    # Choose main metric
    if args.metric == "rgb":
        merged["delta_main"] = merged["delta_rgb"]
    elif args.metric == "rgb_std":
        merged["delta_main"] = merged["delta_rgb_std"]
    elif args.metric == "lab_all":
        merged["delta_main"] = merged["delta_lab_all"]
    elif args.metric == "lab_ab":
        merged["delta_main"] = merged["delta_lab_ab"]
    elif args.metric == "lab_ab_aligned":
        merged["delta_main"] = merged["delta_lab_ab_aligned"]
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    # Coverage filter -> NaN out for main metric
    ok = merged["coverage"] >= args.coverage_threshold
    merged.loc[~ok, "delta_main"] = np.nan

    # Save tables
    keep_cols = [
        "ref_label_id", "coverage",
        "delta_main",
        "delta_rgb", "delta_std", "delta_rgb_std",
        "delta_lab_all", "delta_lab_ab", "delta_lab_ab_aligned",
        "mean_r_ref", "mean_g_ref", "mean_b_ref",
        "mean_r_src", "mean_g_src", "mean_b_src",
        "mean_L_ref", "mean_a_ref", "mean_b_ref_lab",
        "mean_L_src", "mean_a_src", "mean_b_src_lab",
    ]
    merged[keep_cols].to_parquet(out_dir / "delta_full.parquet", index=False)

    filtered = merged.dropna(subset=["delta_main"]).copy()
    filtered.to_parquet(out_dir / "delta.parquet", index=False)

    top = filtered.sort_values("delta_main", ascending=False).head(100)
    top[["ref_label_id", "coverage", "delta_main", "delta_rgb", "delta_lab_ab", "delta_lab_ab_aligned"]].to_csv(
        out_dir / "top_changed.csv", index=False
    )

    # Load labels + images for heatmaps/gallery
    labels = load_labels_npz(args.ref_labels)
    ref_image = load_image_rgb(args.ref_image)

    # Heatmaps
    # Coverage heatmap (coverage already in [0,1], normalize over full range)
    cov_map = dict(zip(merged["ref_label_id"].to_numpy(), merged["coverage"].to_numpy()))
    cov_img = build_heatmap_image(
        labels=labels,
        ref_image=ref_image,
        value_map=cov_map,
        pct_low=0.0,
        pct_high=100.0,
        alpha_heat=0.55,
        cmap_name="viridis",
    )
    cv2.imwrite(str(out_dir / "coverage_heatmap.png"), cv2.cvtColor(cov_img, cv2.COLOR_RGB2BGR))

    # Delta heatmap (main metric)
    delta_map = dict(zip(merged["ref_label_id"].to_numpy(), merged["delta_main"].to_numpy()))
    delta_img = build_heatmap_image(
        labels=labels,
        ref_image=ref_image,
        value_map=delta_map,
        pct_low=args.norm_pct_low,
        pct_high=args.norm_pct_high,
        alpha_heat=args.heat_alpha,
        cmap_name="inferno",
    )
    # Save metric-specific name + backward-compatible name
    cv2.imwrite(str(out_dir / f"delta_{args.metric}.png"), cv2.cvtColor(delta_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "delta_heatmap.png"), cv2.cvtColor(delta_img, cv2.COLOR_RGB2BGR))

    # Histograms
    save_histogram(merged["coverage"], out_dir / "coverage_hist.png", "Coverage distribution")
    save_histogram(merged["delta_rgb"].where(ok), out_dir / "delta_rgb_hist.png", "Delta RGB (coverage-filtered)")
    save_histogram(merged["delta_lab_ab"].where(ok), out_dir / "delta_lab_ab_hist.png", "Delta Lab AB (coverage-filtered)")
    if args.global_color_align:
        save_histogram(
            merged["delta_lab_ab_aligned"].where(ok),
            out_dir / "delta_lab_ab_aligned_hist.png",
            "Delta Lab AB aligned (coverage-filtered)",
        )
    save_histogram(merged["delta_main"], out_dir / "delta_hist.png", f"Delta main ({args.metric})")

    # Optional gallery
    if args.src_warp_image is not None:
        src_warp = load_image_rgb(args.src_warp_image)
        if src_warp.shape[:2] != ref_image.shape[:2]:
            raise ValueError(
                f"src_warp_image size {src_warp.shape[:2]} != ref_image size {ref_image.shape[:2]}. "
                "They must match (both in ref frame)."
            )

        gallery_path = out_dir / f"gallery_top{args.gallery_k}_{args.metric}.png"
        build_top_gallery(
            labels=labels,
            ref_img=ref_image,
            src_warp_img=src_warp,
            top_df=top,
            out_path=gallery_path,
            tile=args.gallery_tile,
            pad=args.gallery_pad,
            k=args.gallery_k,
            gap=args.gallery_gap,
            margin=args.gallery_margin,
            metric_name=args.metric,
        )


if __name__ == "__main__":
    main()
