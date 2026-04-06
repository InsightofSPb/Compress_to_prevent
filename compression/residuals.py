from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .io import RGBImage, ensure_dir, load_rgb_image, read_csv_rows, save_rgb_image, write_csv_rows


def compute_rgb_residual(curr_rgb: RGBImage, prev_aligned_rgb: RGBImage) -> RGBImage:
    curr_w, curr_h, curr_payload = curr_rgb
    prev_w, prev_h, prev_payload = prev_aligned_rgb
    if (curr_w, curr_h) != (prev_w, prev_h):
        raise ValueError(f"Shape mismatch: curr={(curr_w, curr_h)}, prev_aligned={(prev_w, prev_h)}")

    residual = bytes(((c - p) % 256) for c, p in zip(curr_payload, prev_payload))
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

        prev_rgb = load_rgb_image(prev_path)
        curr_rgb = load_rgb_image(curr_path)
        residual_rgb = compute_rgb_residual(curr_rgb, prev_rgb)

        out_dir = out_root / split / facade_id
        ensure_dir(out_dir)
        residual_path = out_dir / f"{pair_id}.ppm"
        save_rgb_image(residual_path, residual_rgb)

        width, height, _ = residual_rgb
        manifest_rows.append(
            {
                "pair_id": pair_id,
                "facade_id": facade_id,
                "split": split,
                "residual_path": str(residual_path),
                "prev_aligned_path": str(prev_path),
                "curr_image_path": str(curr_path),
                "height": height,
                "width": width,
                "channels": 3,
            }
        )
        pair_meta_rows.append(
            {
                "pair_id": pair_id,
                "year_prev": row.get("year_prev", ""),
                "year_curr": row.get("year_curr", ""),
                "split": split,
            }
        )

    manifest_fields = [
        "pair_id",
        "facade_id",
        "split",
        "residual_path",
        "prev_aligned_path",
        "curr_image_path",
        "height",
        "width",
        "channels",
    ]
    meta_fields = ["pair_id", "year_prev", "year_curr", "split"]
    write_csv_rows(out_root / "residual_manifest.csv", manifest_fields, manifest_rows)
    write_csv_rows(out_root / "pair_metadata.csv", meta_fields, pair_meta_rows)
    return manifest_rows
