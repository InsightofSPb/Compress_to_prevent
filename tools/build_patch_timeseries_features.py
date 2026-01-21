import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build patch-level time-series features from patch masks/images."
    )
    parser.add_argument("--patch-manifest", required=True, type=Path, help="CSV/Parquet with facade_id, patch_id, year, mask_path")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--class-map", type=Path, default=None, help="Optional JSON mapping class_id->name or name->id")
    parser.add_argument("--codec", default="png", type=str, choices=("png", "webp"))
    parser.add_argument("--min-area", default=25, type=int, help="Minimum patch area in pixels")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_class_map(path: Optional[Path]) -> Dict[int, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("class-map must be a JSON object")
    mapping: Dict[int, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and k.isdigit():
            mapping[int(k)] = str(v)
        elif isinstance(v, (int, float)) and str(k):
            mapping[int(v)] = str(k)
        else:
            raise ValueError("class-map must map class_id->name or name->class_id")
    return mapping


def sanitize_name(name: str) -> str:
    safe = name.strip().lower().replace(" ", "_").replace("/", "_")
    return "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-"})


def load_mask(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(p) as data:
            if "mask" in data:
                mask = data["mask"]
            else:
                mask = data[list(data.keys())[0]]
    elif p.suffix.lower() == ".npy":
        mask = np.load(p)
    else:
        mask = np.array(Image.open(p))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int64)


def encode_image_bytes(image: np.ndarray, codec: str) -> int:
    buffer = BytesIO()
    img = Image.fromarray(image)
    if codec == "png":
        img.save(buffer, format="PNG", compress_level=9, optimize=False)
    else:
        img.save(buffer, format="WEBP", lossless=True, quality=100, method=6)
    return buffer.getbuffer().nbytes


def load_image(path: str) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def compute_entropy(shares: Iterable[float]) -> float:
    vals = np.array([v for v in shares if v > 0], dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(-np.sum(vals * np.log(vals)))


def build_patch_year_features(df: pd.DataFrame, class_map: Dict[int, str], codec: str, min_area: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        facade_id = row["facade_id"]
        patch_id = row["patch_id"]
        year = int(row["year"])
        mask_path = row["mask_path"]
        quality = row.get("quality")

        mask = load_mask(mask_path)
        total_area = int(mask.size)
        if total_area < min_area:
            continue

        counts = np.bincount(mask.flatten())
        class_ids = np.where(counts > 0)[0].tolist()
        per_class: Dict[str, float] = {}
        per_share: Dict[str, float] = {}
        for cid in class_ids:
            name = class_map.get(cid, f"class_{cid}")
            col_name = sanitize_name(name)
            area = int(counts[cid])
            per_class[f"area_px_{col_name}"] = area
            per_share[f"share_{col_name}"] = float(area / total_area)

        shares = list(per_share.values())
        entropy = compute_entropy(shares)
        max_share = float(max(shares)) if shares else float("nan")

        comp_signal = None
        image_path = row.get("image_path")
        if isinstance(image_path, str) and image_path:
            image = load_image(image_path)
            bytes_size = encode_image_bytes(image, codec)
            comp_signal = float((bytes_size * 8) / max(total_area, 1))

        rows.append(
            {
                "facade_id": facade_id,
                "patch_id": patch_id,
                "year": year,
                "quality": quality,
                "area_px_total": total_area,
                "entropy": entropy,
                "max_share": max_share,
                "comp_signal_bpp": comp_signal,
                **per_class,
                **per_share,
            }
        )

    return pd.DataFrame(rows)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if p.size == 0:
        return float("nan")
    p = np.clip(p.astype(float), 0, None)
    q = np.clip(q.astype(float), 0, None)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * (_kl(p, m) + _kl(q, m))


def build_timeseries_features(year_df: pd.DataFrame) -> pd.DataFrame:
    share_cols = sorted([c for c in year_df.columns if c.startswith("share_")])
    area_cols = sorted([c for c in year_df.columns if c.startswith("area_px_") and c != "area_px_total"])

    rows: List[Dict[str, object]] = []
    for (facade_id, patch_id), group in year_df.groupby(["facade_id", "patch_id"]):
        group_sorted = group.sort_values("year")
        years = group_sorted["year"].to_list()
        for idx in range(len(years) - 1):
            prev = group_sorted.iloc[idx]
            nxt = group_sorted.iloc[idx + 1]

            entry: Dict[str, object] = {
                "facade_id": facade_id,
                "patch_id": patch_id,
                "step_idx": idx,
                "year_prev": int(prev["year"]),
                "year_next": int(nxt["year"]),
                "quality": prev.get("quality"),
                "entropy_t": prev.get("entropy"),
                "entropy_t1": nxt.get("entropy"),
                "max_share_t": prev.get("max_share"),
                "max_share_t1": nxt.get("max_share"),
                "comp_signal_t": prev.get("comp_signal_bpp"),
                "comp_signal_t1": nxt.get("comp_signal_bpp"),
            }

            delta_shares: List[float] = []
            for col in share_cols:
                share_prev = float(prev.get(col, 0.0) or 0.0)
                share_next = float(nxt.get(col, 0.0) or 0.0)
                delta = share_next - share_prev
                entry[f"{col}_t"] = share_prev
                entry[f"{col}_t1"] = share_next
                entry[f"delta_{col}"] = delta
                delta_shares.append(delta)

            delta_areas: List[float] = []
            for col in area_cols:
                area_prev = float(prev.get(col, 0.0) or 0.0)
                area_next = float(nxt.get(col, 0.0) or 0.0)
                delta = area_next - area_prev
                entry[f"{col}_t"] = area_prev
                entry[f"{col}_t1"] = area_next
                entry[f"delta_{col}"] = delta
                delta_areas.append(delta)

            entry["l1_share_change"] = float(np.sum(np.abs(delta_shares))) if delta_shares else float("nan")
            entry["max_delta_share"] = float(np.max(np.abs(delta_shares))) if delta_shares else float("nan")

            p = np.array([float(prev.get(c, 0.0) or 0.0) for c in share_cols])
            q = np.array([float(nxt.get(c, 0.0) or 0.0) for c in share_cols])
            entry["js_divergence"] = js_divergence(p, q) if share_cols else float("nan")

            comp_prev = entry["comp_signal_t"]
            comp_next = entry["comp_signal_t1"]
            if comp_prev is not None and comp_next is not None:
                entry["delta_comp_signal"] = float(comp_next - comp_prev)
                entry["abs_delta_comp_signal"] = abs(entry["delta_comp_signal"])
            else:
                entry["delta_comp_signal"] = float("nan")
                entry["abs_delta_comp_signal"] = float("nan")

            rows.append(entry)

    return pd.DataFrame(rows)


def build_index(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(
            columns=["facade_id", "patch_id", "n_steps", "min_year", "max_year", "usable_for_temporal_model"]
        )

    def summarize_group(df: pd.DataFrame) -> Dict[str, object]:
        n_steps = len(df)
        min_year = int(df["year_prev"].min())
        max_year = int(df["year_next"].max())
        usable = n_steps >= 2
        return {
            "n_steps": n_steps,
            "min_year": min_year,
            "max_year": max_year,
            "usable_for_temporal_model": usable,
        }

    grouped = (
        features.groupby(["facade_id", "patch_id"], as_index=False)
        .apply(lambda g: pd.Series(summarize_group(g)))
        .reset_index(drop=True)
    )
    return grouped


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_table(args.patch_manifest)
    required = {"facade_id", "patch_id", "year", "mask_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"patch-manifest missing required columns: {sorted(missing)}")

    class_map = load_class_map(args.class_map)
    patch_year = build_patch_year_features(manifest, class_map, args.codec, args.min_area)
    if patch_year.empty:
        raise ValueError("No patch-year features generated; check inputs")

    patch_year_path = args.out_dir / "patch_year_features.parquet"
    patch_year.to_parquet(patch_year_path, index=False)

    timeseries = build_timeseries_features(patch_year)
    if timeseries.empty:
        raise ValueError("No patch time-series features generated (need >=2 years per patch)")

    features_path = args.out_dir / "timeseries_features_patch.parquet"
    timeseries.to_parquet(features_path, index=False)

    index_df = build_index(timeseries)
    index_path = args.out_dir / "timeseries_index_patch.csv"
    index_df.to_csv(index_path, index=False)

    print(f"[OK] Saved patch-year features to {patch_year_path}")
    print(f"[OK] Saved patch time-series features to {features_path}")
    print(f"[OK] Saved patch time-series index to {index_path}")


if __name__ == "__main__":
    main()
