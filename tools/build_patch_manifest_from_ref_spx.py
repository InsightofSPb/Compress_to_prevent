import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Edge:
    src_year: int
    dst_year: int
    H: np.ndarray
    quality: str
    valid_mask_path: Optional[Path]
    mask_coords: str  # "src" or "dst"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build patch manifest in reference-year coordinates from ref_spx outputs."
    )
    parser.add_argument("--temporal-manifest", required=True, type=Path, help="CSV with facade_id, year, image_path, mask_path")
    parser.add_argument("--ref-spx-out", required=True, type=Path, help="Directory with ref_spx batch outputs")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for patches and manifest")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--coverage-threshold", type=float, default=0.85)
    parser.add_argument("--interp", type=str, default="linear", choices=("linear", "cubic"))
    parser.add_argument("--mask-interp", type=str, default="nearest", choices=("nearest", "linear"))
    return parser.parse_args()


def _is_valid(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value)) and str(value).strip() != ""


def _detect_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


def _detect_pair_column(columns: Iterable[str], candidates: Iterable[str], suffix: str) -> Optional[str]:
    for name in candidates:
        col = f"{name}_{suffix}"
        if col in columns:
            return col
    return None


def _normalize_year(value: object) -> Optional[int]:
    if not _is_valid(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_temporal_manifest(
    manifest_path: Path,
) -> Tuple[
    Dict[Tuple[str, int], str],
    Dict[Tuple[str, int], str],
    Dict[Tuple[str, int], str],
]:
    df = pd.read_csv(manifest_path)
    columns = set(df.columns)
    if "facade_id" not in columns:
        raise ValueError("temporal manifest must include facade_id column")

    image_candidates = ["image_path", "img_path", "image", "path_image"]
    mask_candidates = ["mask_path", "seg_mask_path", "mask", "path_mask"]
    quality_candidates = ["quality"]

    year_col = "year" if "year" in columns else None
    year_prev_col = "year_prev" if "year_prev" in columns else None
    year_next_col = "year_next" if "year_next" in columns else None

    if year_col is None and not (year_prev_col and year_next_col):
        raise ValueError("temporal manifest must include year or year_prev/year_next columns")

    if year_col is not None:
        image_col = _detect_column(columns, image_candidates)
        mask_col = _detect_column(columns, mask_candidates)
        quality_col = _detect_column(columns, quality_candidates)
        if image_col is None or mask_col is None:
            raise ValueError("temporal manifest must include image_path and mask_path columns")
        img_by_year: Dict[Tuple[str, int], str] = {}
        mask_by_year: Dict[Tuple[str, int], str] = {}
        quality_by_year: Dict[Tuple[str, int], str] = {}
        for _, row in df.iterrows():
            facade_id = str(row["facade_id"])
            year = _normalize_year(row[year_col])
            if year is None:
                continue
            img_val = row.get(image_col)
            mask_val = row.get(mask_col)
            quality_val = row.get(quality_col) if quality_col else None
            if _is_valid(img_val):
                img_by_year[(facade_id, year)] = str(img_val)
            if _is_valid(mask_val):
                mask_by_year[(facade_id, year)] = str(mask_val)
            if _is_valid(quality_val):
                quality_by_year[(facade_id, year)] = str(quality_val)
        return img_by_year, mask_by_year, quality_by_year

    image_prev_col = _detect_pair_column(columns, image_candidates, "prev") or _detect_column(columns, image_candidates)
    image_next_col = _detect_pair_column(columns, image_candidates, "next") or _detect_column(columns, image_candidates)
    mask_prev_col = _detect_pair_column(columns, mask_candidates, "prev") or _detect_column(columns, mask_candidates)
    mask_next_col = _detect_pair_column(columns, mask_candidates, "next") or _detect_column(columns, mask_candidates)
    quality_prev_col = _detect_pair_column(columns, quality_candidates, "prev") or _detect_column(columns, quality_candidates)
    quality_next_col = _detect_pair_column(columns, quality_candidates, "next") or _detect_column(columns, quality_candidates)

    if image_prev_col is None or image_next_col is None or mask_prev_col is None or mask_next_col is None:
        raise ValueError("temporal manifest missing image/mask columns for paired years")

    img_by_year = {}
    mask_by_year = {}
    quality_by_year = {}
    for _, row in df.iterrows():
        facade_id = str(row["facade_id"])
        year_prev = _normalize_year(row[year_prev_col])
        year_next = _normalize_year(row[year_next_col])
        if year_prev is not None:
            img_val = row.get(image_prev_col)
            mask_val = row.get(mask_prev_col)
            quality_val = row.get(quality_prev_col) if quality_prev_col else None
            if _is_valid(img_val):
                img_by_year[(facade_id, year_prev)] = str(img_val)
            if _is_valid(mask_val):
                mask_by_year[(facade_id, year_prev)] = str(mask_val)
            if _is_valid(quality_val):
                quality_by_year[(facade_id, year_prev)] = str(quality_val)
        if year_next is not None:
            img_val = row.get(image_next_col)
            mask_val = row.get(mask_next_col)
            quality_val = row.get(quality_next_col) if quality_next_col else None
            if _is_valid(img_val):
                img_by_year[(facade_id, year_next)] = str(img_val)
            if _is_valid(mask_val):
                mask_by_year[(facade_id, year_next)] = str(mask_val)
            if _is_valid(quality_val):
                quality_by_year[(facade_id, year_next)] = str(quality_val)
    return img_by_year, mask_by_year, quality_by_year


def index_ref_spx_outputs(ref_spx_out: Path) -> Tuple[
    Dict[Tuple[str, int, int], np.ndarray],
    Dict[Tuple[str, int, int], str],
    Dict[Tuple[str, int, int], Optional[Path]],
]:
    H_map: Dict[Tuple[str, int, int], np.ndarray] = {}
    quality_map: Dict[Tuple[str, int, int], str] = {}
    valid_mask_map: Dict[Tuple[str, int, int], Optional[Path]] = {}
    pair_root = ref_spx_out / "pairs"
    if not pair_root.exists():
        raise FileNotFoundError(f"Missing pairs directory under {ref_spx_out}")

    for json_path in pair_root.rglob("*.json"):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        facade_id = data.get("facade_id")
        year_a = data.get("year_a")
        year_b = data.get("year_b")
        H = data.get("H")
        if facade_id is None or year_a is None or year_b is None or H is None:
            continue
        H_arr = np.array(H, dtype=float)
        if H_arr.shape != (3, 3):
            continue
        key = (str(facade_id), int(year_a), int(year_b))
        quality = data.get("status_quality") or data.get("quality") or ""
        H_map[key] = H_arr
        quality_map[key] = str(quality)
        valid_mask = json_path.parent / f"valid_mask_{year_a}_to_{year_b}.png"
        valid_mask_map[key] = valid_mask if valid_mask.exists() else None
    return H_map, quality_map, valid_mask_map


def build_graph(
    H_map: Dict[Tuple[str, int, int], np.ndarray],
    quality_map: Dict[Tuple[str, int, int], str],
    valid_mask_map: Dict[Tuple[str, int, int], Optional[Path]],
) -> Dict[str, Dict[int, List[Edge]]]:
    graph: Dict[str, Dict[int, List[Edge]]] = defaultdict(lambda: defaultdict(list))
    for (facade_id, year_a, year_b), H in H_map.items():
        quality = quality_map.get((facade_id, year_a, year_b), "")
        valid_mask = valid_mask_map.get((facade_id, year_a, year_b))
        graph[facade_id][year_a].append(
            Edge(year_a, year_b, H, quality, valid_mask, "dst")
        )
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue
        graph[facade_id][year_b].append(
            Edge(year_b, year_a, H_inv, quality, valid_mask, "src")
        )
    return graph


def get_H_to_ref(
    graph: Dict[str, Dict[int, List[Edge]]],
    facade_id: str,
    year: int,
    ref_year: int,
) -> Optional[Tuple[np.ndarray, List[Edge], List[int]]]:
    if year == ref_year:
        return np.eye(3, dtype=float), [], [year]

    visited = set()
    parent: Dict[int, Tuple[int, Edge]] = {}
    queue = deque([year])
    visited.add(year)

    while queue:
        current = queue.popleft()
        if current == ref_year:
            break
        for edge in graph.get(facade_id, {}).get(current, []):
            if edge.dst_year in visited:
                continue
            visited.add(edge.dst_year)
            parent[edge.dst_year] = (current, edge)
            queue.append(edge.dst_year)

    if ref_year not in parent:
        return None

    edges: List[Edge] = []
    years: List[int] = [ref_year]
    cursor = ref_year
    while cursor != year:
        prev, edge = parent[cursor]
        edges.append(edge)
        years.append(prev)
        cursor = prev
    edges.reverse()
    years.reverse()

    H_total = np.eye(3, dtype=float)
    for edge in edges:
        H_total = edge.H @ H_total
    return H_total, edges, years


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image {path}")
    return img


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def warp_image(image: np.ndarray, H: np.ndarray, out_shape: Tuple[int, int], interp: int) -> np.ndarray:
    height, width = out_shape
    return cv2.warpPerspective(image, H, (width, height), flags=interp)


def warp_mask(mask: np.ndarray, H: np.ndarray, out_shape: Tuple[int, int], interp: int) -> np.ndarray:
    height, width = out_shape
    return cv2.warpPerspective(mask, H, (width, height), flags=interp)


def quality_from_edges(edges: List[Edge]) -> str:
    qualities = [edge.quality for edge in edges if edge.quality]
    if not qualities:
        return ""
    if any(q == "weak" for q in qualities):
        return "weak"
    if any(q == "strong" for q in qualities):
        return "strong"
    return qualities[0]


def draw_grid(image: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    grid = image.copy()
    height, width = grid.shape[:2]
    color = (0, 255, 0)
    for y in range(0, height, stride):
        cv2.line(grid, (0, y), (width, y), color, 1)
    for x in range(0, width, stride):
        cv2.line(grid, (x, 0), (x, height), color, 1)
    return grid


def main() -> int:
    args = parse_args()
    stride = args.stride if args.stride is not None else args.patch_size
    interp_map = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
    mask_interp_map = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}
    interp = interp_map[args.interp]
    mask_interp = mask_interp_map[args.mask_interp]

    img_by_year, mask_by_year, quality_by_year = load_temporal_manifest(args.temporal_manifest)
    H_map, quality_map, valid_mask_map = index_ref_spx_outputs(args.ref_spx_out)
    graph = build_graph(H_map, quality_map, valid_mask_map)

    out_dir = args.out_dir
    patch_root = out_dir / "patches"
    mask_root = out_dir / "masks"
    debug_root = out_dir / "debug"
    patch_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    facades = sorted({facade for facade, _ in img_by_year.keys()} | {facade for facade, _ in mask_by_year.keys()})

    manifest_rows: List[Dict[str, object]] = []
    facade_stats = []
    debug_facades = set()

    for facade_idx, facade_id in enumerate(facades):
        years = sorted({year for (facade, year) in img_by_year.keys() if facade == facade_id}
                       | {year for (facade, year) in mask_by_year.keys() if facade == facade_id})
        if not years:
            continue
        ref_year = max(years)
        ref_img_path_str = img_by_year.get((facade_id, ref_year))
        if not ref_img_path_str:
            print(f"[warn] Missing ref image for {facade_id} {ref_year}", file=sys.stderr)
            continue
        ref_img_path = Path(ref_img_path_str)
        ref_image = read_image(ref_img_path)
        ref_height, ref_width = ref_image.shape[:2]

        H_cache: Dict[int, Tuple[np.ndarray, List[Edge], List[int]]] = {}
        for year in years:
            H_info = get_H_to_ref(graph, facade_id, year, ref_year)
            if H_info is None:
                print(f"[warn] No path from {facade_id} {year} to ref {ref_year}", file=sys.stderr)
                continue
            H_cache[year] = H_info

        if ref_year not in H_cache:
            H_cache[ref_year] = (np.eye(3, dtype=float), [], [ref_year])

        patches_saved = 0
        for year in years:
            if year not in H_cache:
                continue
            img_path_str = img_by_year.get((facade_id, year))
            mask_path_str = mask_by_year.get((facade_id, year))
            if not img_path_str or not mask_path_str:
                print(f"[warn] Missing image/mask for {facade_id} {year}", file=sys.stderr)
                continue

            image = read_image(Path(img_path_str))
            mask = read_mask(Path(mask_path_str))

            H_total, edges, _ = H_cache[year]
            img_ref = warp_image(image, H_total, (ref_height, ref_width), interp=interp)
            mask_ref = warp_mask(mask, H_total, (ref_height, ref_width), interp=mask_interp)

            if year == ref_year:
                valid_ref = np.ones((ref_height, ref_width), dtype=bool)
            else:
                valid_ref = np.ones((ref_height, ref_width), dtype=bool)
                for edge in edges:
                    if edge.valid_mask_path is None or not edge.valid_mask_path.exists():
                        print(
                            f"[warn] Missing valid mask for {facade_id} {edge.src_year}->{edge.dst_year}",
                            file=sys.stderr,
                        )
                        continue
                    try:
                        step_mask = read_mask(edge.valid_mask_path)
                    except FileNotFoundError:
                        print(
                            f"[warn] Failed to read valid mask for {facade_id} {edge.src_year}->{edge.dst_year}",
                            file=sys.stderr,
                        )
                        continue
                    H_dst_info = H_cache.get(edge.dst_year)
                    if H_dst_info is None:
                        print(
                            f"[warn] Missing H cache for {facade_id} {edge.dst_year} (valid mask)",
                            file=sys.stderr,
                        )
                        continue
                    H_dst_to_ref = H_dst_info[0]
                    if edge.mask_coords == "src":
                        H_step = H_dst_to_ref @ edge.H
                    else:
                        H_step = H_dst_to_ref
                    step_ref = warp_mask(step_mask, H_step, (ref_height, ref_width), interp=cv2.INTER_NEAREST)
                    valid_ref &= step_ref > 0

            patch_dir = patch_root / facade_id
            mask_dir = mask_root / facade_id
            patch_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            for y0 in range(0, ref_height - args.patch_size + 1, stride):
                for x0 in range(0, ref_width - args.patch_size + 1, stride):
                    y1 = y0 + args.patch_size
                    x1 = x0 + args.patch_size
                    coverage = float(np.mean(valid_ref[y0:y1, x0:x1]))
                    if coverage < args.coverage_threshold:
                        continue
                    patch_id = f"r{y0//stride:03d}_c{x0//stride:03d}"

                    patch_subdir = patch_dir / patch_id
                    mask_subdir = mask_dir / patch_id
                    patch_subdir.mkdir(parents=True, exist_ok=True)
                    mask_subdir.mkdir(parents=True, exist_ok=True)

                    patch_img_path = patch_subdir / f"{year}.png"
                    patch_mask_path = mask_subdir / f"{year}.png"
                    cv2.imwrite(str(patch_img_path), img_ref[y0:y1, x0:x1])
                    cv2.imwrite(str(patch_mask_path), mask_ref[y0:y1, x0:x1])

                    quality = quality_by_year.get((facade_id, year))
                    if not quality:
                        quality = quality_from_edges(edges)

                    manifest_rows.append(
                        {
                            "facade_id": facade_id,
                            "patch_id": patch_id,
                            "year": year,
                            "image_path": str(patch_img_path),
                            "mask_path": str(patch_mask_path),
                            "quality": quality,
                            "ref_year": ref_year,
                            "patch_x0": x0,
                            "patch_y0": y0,
                            "patch_x1": x1,
                            "patch_y1": y1,
                            "coverage": coverage,
                        }
                    )
                    patches_saved += 1

            if len(debug_facades) < 2 and facade_id not in debug_facades:
                debug_facades.add(facade_id)

            if facade_id in debug_facades:
                valid_path = debug_root / f"{facade_id}_{year}_valid_ref.png"
                cv2.imwrite(str(valid_path), (valid_ref * 255).astype(np.uint8))

        if facade_id in debug_facades:
            grid_path = debug_root / f"{facade_id}_ref_grid.png"
            cv2.imwrite(str(grid_path), draw_grid(ref_image, args.patch_size, stride))

        facade_stats.append((facade_id, len(years), patches_saved))

    manifest_path = out_dir / "patch_manifest.csv"
    fieldnames = [
        "facade_id",
        "patch_id",
        "year",
        "image_path",
        "mask_path",
        "quality",
        "ref_year",
        "patch_x0",
        "patch_y0",
        "patch_x1",
        "patch_y1",
        "coverage",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    try:
        pd.DataFrame(manifest_rows).to_parquet(out_dir / "patch_manifest.parquet")
    except Exception as exc:
        print(f"[warn] Failed to write parquet manifest: {exc}", file=sys.stderr)

    total_patches = sum(stat[2] for stat in facade_stats)
    print(f"Processed {len(facade_stats)} facades, saved {total_patches} patches.")
    for facade_id, n_years, n_patches in facade_stats:
        print(f"- {facade_id}: years={n_years}, patches={n_patches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
