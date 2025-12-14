import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import color, io
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import regionprops_table


DEFAULT_SEGMENTS = 800
SMALL_IMAGE_SEGMENTS = 500
SMALL_IMAGE_MIN_SIDE = 512
COMPACTNESS = 10
SIGMA = 1


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build superpixel objects from a temporal manifest.")
    parser.add_argument(
        "--temporal-manifest",
        required=True,
        type=Path,
        help="Path to CSV manifest with facade_id, year, image_path/full_path columns.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--limit-facades",
        type=int,
        default=None,
        help="Optional limit on number of facades to process (in manifest order).",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    required_columns = {"facade_id", "year"}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(sorted(required_columns - set(df.columns)))
        raise ValueError(f"Manifest missing required columns: {missing}")

    if "image_path" in df.columns:
        path_column = "image_path"
    elif "full_path" in df.columns:
        path_column = "full_path"
    else:
        raise ValueError("Manifest must include either 'image_path' or 'full_path' column")

    df = df.rename(columns={path_column: "image_path"})
    return df


def ensure_directories(base_dir: Path):
    (base_dir / "spx").mkdir(parents=True, exist_ok=True)
    (base_dir / "objs").mkdir(parents=True, exist_ok=True)
    (base_dir / "viz").mkdir(parents=True, exist_ok=True)


def choose_segment_count(image: np.ndarray) -> int:
    min_side = min(image.shape[0], image.shape[1])
    if min_side <= SMALL_IMAGE_MIN_SIDE:
        return SMALL_IMAGE_SEGMENTS
    return DEFAULT_SEGMENTS


def compute_superpixels(image: np.ndarray, n_segments: int) -> np.ndarray:
    return slic(
        image,
        n_segments=n_segments,
        compactness=COMPACTNESS,
        sigma=SIGMA,
        start_label=0,
        channel_axis=-1,
    )


def build_objects(labels: np.ndarray, facade_id: str, year: int, gray_image: np.ndarray) -> pd.DataFrame:
    props = regionprops_table(
        labels,
        intensity_image=gray_image,
        properties=(
            "label",
            "area",
            "centroid",
            "bbox",
            "intensity_mean",
            "intensity_std",
        ),
    )
    df = pd.DataFrame(props)
    df = df.rename(
        columns={
            "label": "label_id",
            "area": "area_px",
            "centroid-1": "cx",
            "centroid-0": "cy",
            "bbox-1": "bbox_x1",
            "bbox-0": "bbox_y1",
            "bbox-3": "bbox_x2",
            "bbox-2": "bbox_y2",
            "intensity_mean": "mean_intensity",
            "intensity_std": "std_intensity",
        }
    )
    df.insert(0, "obj_id", np.arange(len(df), dtype=np.int32))
    df.insert(1, "facade_id", facade_id)
    df.insert(2, "year", year)
    df = df[
        [
            "obj_id",
            "facade_id",
            "year",
            "area_px",
            "cx",
            "cy",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "mean_intensity",
            "std_intensity",
        ]
    ]
    return df


def save_outputs(base_dir: Path, year: int, labels: np.ndarray, objects: pd.DataFrame, overlay: np.ndarray):
    np.savez_compressed(base_dir / "spx" / f"{year}_labels.npz", labels=labels.astype(np.int32))
    objects.to_parquet(base_dir / "objs" / f"{year}_spx.parquet", index=False)
    io.imsave(base_dir / "viz" / f"{year}_spx_overlay.png", overlay)


def log_quality(labels: np.ndarray, areas: pd.Series):
    k = len(np.unique(labels))
    min_area = areas.min()
    max_area = areas.max()
    LOGGER.info("Superpixels: K=%d, min area=%s, max area=%s", k, min_area, max_area)
    if k < 200 or k > 3000:
        LOGGER.warning("Superpixel count (%d) outside expected range [200, 3000]", k)


def process_row(row: pd.Series, out_dir: Path):
    image_path = Path(row["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = io.imread(image_path)
    if image.ndim == 2:  # grayscale to RGB
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:  # drop alpha if present
        image = image[..., :3]

    n_segments = choose_segment_count(image)
    labels = compute_superpixels(image, n_segments)

    gray_image = color.rgb2gray(image)
    objects = build_objects(labels, row["facade_id"], row["year"], gray_image)

    overlay = mark_boundaries(image, labels, color=(1, 0, 0))
    log_quality(labels, objects["area_px"])

    facade_dir = out_dir / "facades" / str(row["facade_id"]) / "spx"
    ensure_directories(facade_dir)
    save_outputs(facade_dir, row["year"], labels, objects, overlay)


def main():
    args = parse_args()
    manifest = load_manifest(args.temporal_manifest)
    if args.limit_facades is not None:
        unique_facades = manifest["facade_id"].drop_duplicates().head(args.limit_facades)
        manifest = manifest[manifest["facade_id"].isin(unique_facades)]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in manifest.iterrows():
        LOGGER.info("Processing facade %s, year %s (row %d)", row["facade_id"], row["year"], idx)
        process_row(row, args.out_dir)


if __name__ == "__main__":
    main()
