import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage import img_as_float
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float as sk_img_as_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute reference superpixel features for a facade.")
    parser.add_argument("--temporal-manifest", required=True, type=Path, help="CSV with facade_id, year, image_path")
    parser.add_argument("--geom-json", required=True, type=Path, help="JSON with homography and quality metadata")
    parser.add_argument("--facade-id", required=True, type=str, help="Facade identifier")
    parser.add_argument("--ref-year", required=True, type=int, help="Reference year for SLIC labels")
    parser.add_argument("--src-year", required=True, type=int, help="Source year to warp into reference frame")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory root")
    parser.add_argument("--n-segments", type=int, default=400, help="Number of superpixels for SLIC")
    parser.add_argument("--compactness", type=float, default=10.0, help="Compactness for SLIC")
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_manifest_image(manifest: Path, facade_id: str, year: int) -> Tuple[Path, np.ndarray]:
    df = pd.read_csv(manifest)
    subset = df[(df["facade_id"] == facade_id) & (df["year"] == year)]
    if subset.empty:
        raise ValueError(f"No entry for facade_id={facade_id}, year={year} in manifest {manifest}")

    # Accept either image_path or mask_path (some users keep photos under mask_path by accident)
    if "image_path" in subset.columns:
        col = "image_path"
    elif "mask_path" in subset.columns:
        col = "mask_path"
    else:
        raise ValueError("Manifest must contain `image_path` (preferred) or `mask_path` column")

    image_path = Path(subset.iloc[0][col])
    return image_path, load_image(image_path)


def load_geom(geom_json: Path) -> Tuple[str, Optional[np.ndarray]]:
    with geom_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    quality = data.get("quality") or data.get("status_quality") or ""
    H = data.get("H")
    if H is not None:
        H = np.array(H, dtype=float)
        if H.shape != (3, 3):
            H = None
    else:
        H = None
    return quality, H


def compute_slic_labels(image: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    image_float = img_as_float(image)
    labels = slic(
        image_float,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
    )
    return labels.astype(np.int32)


def warp_image_and_mask(src: np.ndarray, H: np.ndarray, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    height, width = target_shape
    warped = cv2.warpPerspective(src, H, (width, height), flags=cv2.INTER_LINEAR)
    mask = np.ones(src.shape[:2], dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, H, (width, height), flags=cv2.INTER_NEAREST)
    valid_mask = warped_mask > 0
    return warped, valid_mask


def compute_features(image: np.ndarray, labels: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    flat_labels = np.unique(labels)
    rows = []
    for lbl in flat_labels:
        mask = labels == lbl
        area_px = int(mask.sum())
        if valid_mask is None:
            mask_valid = mask
        else:
            mask_valid = mask & valid_mask
        valid_count = int(mask_valid.sum())
        coverage = valid_count / area_px if area_px > 0 else 0.0

        if valid_count == 0:
            mean_rgb = (np.nan, np.nan, np.nan)
            std_rgb = (np.nan, np.nan, np.nan)
        else:
            pixels = image[mask_valid]
            mean_rgb = tuple(pixels.mean(axis=0).astype(float))
            std_rgb = tuple(pixels.std(axis=0).astype(float))

        rows.append(
            {
                "ref_label_id": int(lbl),
                "area_px": area_px,
                "coverage": float(coverage),
                "mean_r": float(mean_rgb[0]),
                "mean_g": float(mean_rgb[1]),
                "mean_b": float(mean_rgb[2]),
                "std_r": float(std_rgb[0]),
                "std_g": float(std_rgb[1]),
                "std_b": float(std_rgb[2]),
            }
        )
    return pd.DataFrame(rows)


def save_overlay(image: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    # mark_boundaries expects float in [0,1]
    image_f = sk_img_as_float(image)
    overlay = mark_boundaries(image_f, labels, color=(1, 0, 0), mode="thick")
    overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()

    # Load images
    ref_img_path, ref_image = load_manifest_image(args.temporal_manifest, args.facade_id, args.ref_year)
    src_img_path, src_image = load_manifest_image(args.temporal_manifest, args.facade_id, args.src_year)

    # Load geometry
    quality, H = load_geom(args.geom_json)
    if H is None or quality not in {"strong", "weak"}:
        raise SystemExit("no valid H for this pair")

    # Prepare output dir EARLY (needed for raw warp dumps)
    out_base = args.out_dir / "facades" / str(args.facade_id) / "ref_spx" / str(args.ref_year)
    out_base.mkdir(parents=True, exist_ok=True)

    # Compute reference labels
    ref_labels = compute_slic_labels(ref_image, n_segments=args.n_segments, compactness=args.compactness)
    np.savez_compressed(out_base / "ref_labels.npz", labels=ref_labels)

    # Warp src into ref frame + valid mask
    src_warp, valid_mask = warp_image_and_mask(src_image, H, ref_image.shape[:2])

    # --- Save RAW warp + valid mask (for gallery/debug) ---
    raw_warp_path = out_base / f"src_warp_{args.src_year}_to_{args.ref_year}.png"
    cv2.imwrite(str(raw_warp_path), cv2.cvtColor(src_warp, cv2.COLOR_RGB2BGR))

    valid_mask_path = out_base / f"valid_mask_{args.src_year}_to_{args.ref_year}.png"
    cv2.imwrite(str(valid_mask_path), (valid_mask.astype(np.uint8) * 255))

    # Compute features on the SAME ref_labels
    ref_features = compute_features(ref_image, ref_labels, valid_mask=None)
    src_features = compute_features(src_warp, ref_labels, valid_mask=valid_mask)

    # Save features
    ref_features.to_parquet(out_base / f"features_{args.ref_year}.parquet", index=False)
    src_features.to_parquet(out_base / f"features_{args.src_year}_warped.parquet", index=False)

    # Visual overlays
    save_overlay(ref_image, ref_labels, out_base / "ref_overlay.png")
    save_overlay(src_warp, ref_labels, out_base / "warped_overlay.png")

    # Helpful print (optional)
    print(f"[OK] Saved raw warp: {raw_warp_path}")
    print(f"[OK] Saved valid mask: {valid_mask_path}")
    print(f"[OK] Saved ref labels/features to: {out_base}")


if __name__ == "__main__":
    main()
