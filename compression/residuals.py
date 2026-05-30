from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .io import RGBImage, ensure_dir, load_rgb_image, read_csv_rows, save_rgb_image, write_csv_rows


def load_valid_mask(path: Path, expected_size: Tuple[int, int], threshold: int = 0) -> bytes:
    """Load a one-byte-per-pixel validity map as binary 0/1 values."""
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for valid mask I/O") from exc
    image = Image.open(path).convert("L")
    if image.size != expected_size:
        raise ValueError("Valid mask shape mismatch: expected={}, got={} for {}".format(expected_size, image.size, path))
    return bytes(1 if value > threshold else 0 for value in image.tobytes())


def compute_rgb_residual(
    curr_rgb: RGBImage,
    prev_aligned_rgb: RGBImage,
    valid_mask: Optional[bytes] = None,
) -> RGBImage:
    """Compute modulo RGB residual; invalid aligned pixels are written as zero residual.

    A full rectangular image is retained for file I/O and visualization, while
    downstream entropy/heatmap evaluation must use ``valid_mask_path`` from the
    residual manifest to exclude invalid symbols from scores.
    """
    curr_w, curr_h, curr_payload = curr_rgb
    prev_w, prev_h, prev_payload = prev_aligned_rgb
    if (curr_w, curr_h) != (prev_w, prev_h):
        raise ValueError(f"Shape mismatch: curr={(curr_w, curr_h)}, prev_aligned={(prev_w, prev_h)}")
    if valid_mask is not None and len(valid_mask) != curr_w * curr_h:
        raise ValueError("Valid mask payload size mismatch: expected={}, got={}".format(curr_w * curr_h, len(valid_mask)))

    if valid_mask is None:
        residual = bytes(((c - p) % 256) for c, p in zip(curr_payload, prev_payload))
    else:
        output = bytearray(len(curr_payload))
        for pixel_index, is_valid in enumerate(valid_mask):
            base = pixel_index * 3
            if is_valid:
                output[base:base + 3] = bytes(
                    (curr_payload[base + channel] - prev_payload[base + channel]) % 256
                    for channel in range(3)
                )
        residual = bytes(output)
    return curr_w, curr_h, residual


def build_residual_dataset(pairs_csv: Path, out_root: Path) -> List[Dict[str, object]]:
    pair_rows = read_csv_rows(pairs_csv)
    ensure_dir(out_root)

    manifest_rows: List[Dict[str, object]] = []
    pair_meta_rows: List[Dict[str, object]] = []

    for row in pair_rows:
        pair_id = row["pair_id"]
        split = row.get("split", "train") or "train"
        facade_id = row.get("facade_id", "unknown")
        prev_path = Path(row.get("prev_aligned_path") or row["prev_image_path"])
        curr_path = Path(row["curr_image_path"])
        valid_path_raw = row.get("valid_mask_path", "")
        valid_path = Path(valid_path_raw) if valid_path_raw else None
        valid_threshold = int(float(row.get("valid_threshold", "0") or 0))

        prev_rgb = load_rgb_image(prev_path)
        curr_rgb = load_rgb_image(curr_path)
        width, height, _ = curr_rgb
        if valid_path is not None:
            if not valid_path.is_file():
                raise FileNotFoundError("Missing valid mask for {}: {}".format(pair_id, valid_path))
            valid_mask = load_valid_mask(valid_path, (width, height), threshold=valid_threshold)
        else:
            valid_mask = bytes([1]) * (width * height)
        residual_rgb = compute_rgb_residual(curr_rgb, prev_rgb, valid_mask=valid_mask)

        out_dir = out_root / split / facade_id
        ensure_dir(out_dir)
        residual_path = out_dir / f"{pair_id}.ppm"
        save_rgb_image(residual_path, residual_rgb)

        valid_pixel_count = sum(valid_mask)
        manifest_rows.append(
            {
                "pair_id": pair_id,
                "facade_id": facade_id,
                "split": split,
                "residual_path": str(residual_path),
                "prev_aligned_path": str(prev_path),
                "curr_image_path": str(curr_path),
                "valid_mask_path": str(valid_path) if valid_path is not None else "",
                "valid_threshold": valid_threshold,
                "valid_pixel_count": valid_pixel_count,
                "valid_ratio": valid_pixel_count / max(width * height, 1),
                "height": height,
                "width": width,
                "channels": 3,
                "invalid_residual_policy": "zero_filled_and_excluded_from_scoring",
            }
        )
        pair_meta_rows.append(
            {
                "pair_id": pair_id,
                "year_prev": row.get("year_prev", ""),
                "year_curr": row.get("year_curr", ""),
                "split": split,
                "valid_pixel_count": valid_pixel_count,
                "valid_ratio": valid_pixel_count / max(width * height, 1),
            }
        )

    manifest_fields = [
        "pair_id",
        "facade_id",
        "split",
        "residual_path",
        "prev_aligned_path",
        "curr_image_path",
        "valid_mask_path",
        "valid_threshold",
        "valid_pixel_count",
        "valid_ratio",
        "height",
        "width",
        "channels",
        "invalid_residual_policy",
    ]
    meta_fields = ["pair_id", "year_prev", "year_curr", "split", "valid_pixel_count", "valid_ratio"]
    write_csv_rows(out_root / "residual_manifest.csv", manifest_fields, manifest_rows)
    write_csv_rows(out_root / "pair_metadata.csv", meta_fields, pair_meta_rows)
    return manifest_rows
