import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tools.build_manifest_from_masks import parse_mask


@dataclass
class PairRow:
    pair_id: str
    facade_id: str
    year_a: int
    year_b: int
    image_a: Path
    image_b: Path


@dataclass
class MaskEntry:
    facade_id: str
    year: int
    mask_path: Path


@dataclass
class PairResult:
    pair_id: str
    facade_id: str
    year_a: int
    year_b: int
    status: str
    source_mask_path: str
    target_mask_path: str
    warped_mask_path: str
    diff_map_path: str
    overlay_path: str
    iou: Optional[float]
    precision: Optional[float]
    recall: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warp segmentation masks between years using previously estimated pair homography."
    )
    parser.add_argument("--pairs", type=Path, required=True, help="pairs_consecutive.csv from build_pairs_from_masks.py")
    parser.add_argument("--geom-dir", type=Path, required=True, help="Directory with geometry JSONs from debug_mask_rectification.py")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for warped masks, overlays and report")
    parser.add_argument("--mask-manifest", type=Path, default=None, help="CSV with columns facade_id, year, mask_path")
    parser.add_argument("--masks-dir", type=Path, default=None, help="Directory with segmentation masks; filenames should contain year")
    parser.add_argument("--mask-ext", type=str, default=".png,.jpg,.jpeg,.tif,.tiff", help="Extensions for --masks-dir scan")
    parser.add_argument("--facade-id", type=str, default=None, help="Optional facade_id filter")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of pair rows")
    return parser.parse_args()


def read_pairs(path: Path, facade_id: Optional[str], limit: Optional[int]) -> List[PairRow]:
    rows: List[PairRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            current_facade = str(row.get("facade_id", ""))
            if facade_id and current_facade != facade_id:
                continue
            try:
                year_a = int(row["year_a"])
                year_b = int(row["year_b"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid years in row: {row}") from exc
            rows.append(
                PairRow(
                    pair_id=str(row.get("pair_id", f"{current_facade}_{year_a}_{year_b}")),
                    facade_id=current_facade,
                    year_a=year_a,
                    year_b=year_b,
                    image_a=Path(str(row.get("mask_a", ""))),
                    image_b=Path(str(row.get("mask_b", ""))),
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_mask_manifest(path: Path) -> Dict[Tuple[str, int], Path]:
    by_key: Dict[Tuple[str, int], Path] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            facade_id = str(row.get("facade_id", "")).strip()
            year = _safe_int(row.get("year"))
            mask_path = Path(str(row.get("mask_path", "")).strip())
            if not facade_id or year is None or not str(mask_path):
                continue
            if not mask_path.exists():
                continue
            key = (facade_id, year)
            prev = by_key.get(key)
            if prev is None or mask_path.stat().st_size > prev.stat().st_size:
                by_key[key] = mask_path
    return by_key


def scan_masks_dir(masks_dir: Path, ext_arg: str) -> Dict[Tuple[str, int], Path]:
    exts = {e.strip().lower() for e in ext_arg.split(",") if e.strip()}
    by_key: Dict[Tuple[str, int], Path] = {}
    for path in sorted(masks_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        facade_id, year, _, _, _ = parse_mask(path.stem)
        if year is None:
            continue
        key = (facade_id, year)
        prev = by_key.get(key)
        if prev is None or path.stat().st_size > prev.stat().st_size:
            by_key[key] = path
    return by_key


def load_homography(geom_dir: Path, facade_id: str, year_a: int, year_b: int) -> Optional[np.ndarray]:
    geom_candidates = [
        geom_dir / facade_id / f"{year_a}_{year_b}.json",
        geom_dir / f"{facade_id}_{year_a}_{year_b}.json",
    ]
    geom_path = next((p for p in geom_candidates if p.exists()), None)
    if geom_path is None:
        return None

    with geom_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    H = data.get("H")
    if H is None:
        return None
    Hn = np.array(H, dtype=np.float64)
    if Hn.shape != (3, 3):
        return None
    return Hn


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def load_image(path: Path, fallback_shape: Tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        h, w = fallback_shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    return image


def compute_binary_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    a_bin = a > 0
    b_bin = b > 0
    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    a_sum = a_bin.sum()
    b_sum = b_bin.sum()
    iou = 0.0 if union == 0 else float(inter) / float(union)
    precision = 0.0 if a_sum == 0 else float(inter) / float(a_sum)
    recall = 0.0 if b_sum == 0 else float(inter) / float(b_sum)
    return iou, precision, recall


def build_diff_map(warped: np.ndarray, target: np.ndarray) -> np.ndarray:
    warped_bin = warped > 0
    target_bin = target > 0

    only_src = np.logical_and(warped_bin, np.logical_not(target_bin))
    only_tgt = np.logical_and(target_bin, np.logical_not(warped_bin))
    overlap = np.logical_and(warped_bin, target_bin)

    out = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    out[overlap] = (255, 255, 255)
    out[only_src] = (0, 0, 255)
    out[only_tgt] = (0, 255, 0)
    return out


def build_overlay(image_b: np.ndarray, warped: np.ndarray, target: Optional[np.ndarray]) -> np.ndarray:
    overlay = image_b.copy()
    warped_bin = warped > 0

    magenta = np.zeros_like(overlay)
    magenta[:, :, 0] = 255
    magenta[:, :, 2] = 255
    overlay[warped_bin] = cv2.addWeighted(overlay, 0.45, magenta, 0.55, 0)[warped_bin]

    if target is not None:
        target_bin = target > 0
        green = np.zeros_like(overlay)
        green[:, :, 1] = 255
        overlay[target_bin] = cv2.addWeighted(overlay, 0.45, green, 0.55, 0)[target_bin]

    return overlay


def save_pair_outputs(
    out_dir: Path,
    pair: PairRow,
    warped: np.ndarray,
    diff_map: Optional[np.ndarray],
    overlay: np.ndarray,
) -> Tuple[str, str, str]:
    facade_dir = out_dir / "facades" / pair.facade_id
    warped_dir = facade_dir / "warped_masks"
    diff_dir = facade_dir / "diff_maps"
    overlay_dir = facade_dir / "overlays"
    warped_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{pair.year_a}_{pair.year_b}"
    warped_path = warped_dir / f"{stem}_warped.png"
    overlay_path = overlay_dir / f"{stem}_overlay.png"
    diff_path = diff_dir / f"{stem}_diff.png"

    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(overlay_path), overlay)
    if diff_map is not None:
        cv2.imwrite(str(diff_path), diff_map)
    else:
        diff_path = Path("")

    return str(warped_path), str(diff_path), str(overlay_path)


def write_report(out_path: Path, rows: Sequence[PairResult]) -> None:
    fieldnames = [
        "pair_id",
        "facade_id",
        "year_a",
        "year_b",
        "status",
        "source_mask_path",
        "target_mask_path",
        "warped_mask_path",
        "diff_map_path",
        "overlay_path",
        "iou",
        "precision",
        "recall",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "pair_id": row.pair_id,
                    "facade_id": row.facade_id,
                    "year_a": row.year_a,
                    "year_b": row.year_b,
                    "status": row.status,
                    "source_mask_path": row.source_mask_path,
                    "target_mask_path": row.target_mask_path,
                    "warped_mask_path": row.warped_mask_path,
                    "diff_map_path": row.diff_map_path,
                    "overlay_path": row.overlay_path,
                    "iou": "" if row.iou is None else f"{row.iou:.6f}",
                    "precision": "" if row.precision is None else f"{row.precision:.6f}",
                    "recall": "" if row.recall is None else f"{row.recall:.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    if args.mask_manifest is None and args.masks_dir is None:
        raise ValueError("Provide either --mask-manifest or --masks-dir")

    pairs = read_pairs(args.pairs, args.facade_id, args.limit)
    if not pairs:
        print("No pairs to process.")
        return

    if args.mask_manifest is not None:
        mask_index = read_mask_manifest(args.mask_manifest)
    else:
        mask_index = scan_masks_dir(args.masks_dir, args.mask_ext)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: List[PairResult] = []

    for pair in pairs:
        key_a = (pair.facade_id, pair.year_a)
        key_b = (pair.facade_id, pair.year_b)
        src_mask_path = mask_index.get(key_a)
        tgt_mask_path = mask_index.get(key_b)

        if src_mask_path is None:
            results.append(
                PairResult(
                    pair_id=pair.pair_id,
                    facade_id=pair.facade_id,
                    year_a=pair.year_a,
                    year_b=pair.year_b,
                    status="missing_source_mask",
                    source_mask_path="",
                    target_mask_path="" if tgt_mask_path is None else str(tgt_mask_path),
                    warped_mask_path="",
                    diff_map_path="",
                    overlay_path="",
                    iou=None,
                    precision=None,
                    recall=None,
                )
            )
            continue

        H = load_homography(args.geom_dir, pair.facade_id, pair.year_a, pair.year_b)
        if H is None:
            results.append(
                PairResult(
                    pair_id=pair.pair_id,
                    facade_id=pair.facade_id,
                    year_a=pair.year_a,
                    year_b=pair.year_b,
                    status="missing_homography",
                    source_mask_path=str(src_mask_path),
                    target_mask_path="" if tgt_mask_path is None else str(tgt_mask_path),
                    warped_mask_path="",
                    diff_map_path="",
                    overlay_path="",
                    iou=None,
                    precision=None,
                    recall=None,
                )
            )
            continue

        src_mask = load_mask(src_mask_path)
        target_mask = load_mask(tgt_mask_path) if tgt_mask_path is not None and tgt_mask_path.exists() else None
        if target_mask is not None:
            out_h, out_w = target_mask.shape[:2]
        else:
            target_img = cv2.imread(str(pair.image_b), cv2.IMREAD_COLOR)
            if target_img is None:
                results.append(
                    PairResult(
                        pair_id=pair.pair_id,
                        facade_id=pair.facade_id,
                        year_a=pair.year_a,
                        year_b=pair.year_b,
                        status="missing_target_size",
                        source_mask_path=str(src_mask_path),
                        target_mask_path="" if tgt_mask_path is None else str(tgt_mask_path),
                        warped_mask_path="",
                        diff_map_path="",
                        overlay_path="",
                        iou=None,
                        precision=None,
                        recall=None,
                    )
                )
                continue
            out_h, out_w = target_img.shape[:2]

        warped = cv2.warpPerspective(src_mask, H, (out_w, out_h), flags=cv2.INTER_NEAREST)
        diff_map = build_diff_map(warped, target_mask) if target_mask is not None else None
        overlay = build_overlay(
            image_b=load_image(pair.image_b, fallback_shape=(out_h, out_w)),
            warped=warped,
            target=target_mask,
        )

        iou = precision = recall = None
        if target_mask is not None:
            iou, precision, recall = compute_binary_metrics(warped, target_mask)

        warped_path, diff_path, overlay_path = save_pair_outputs(args.out_dir, pair, warped, diff_map, overlay)
        status = "ok_with_target" if target_mask is not None else "ok_without_target"
        results.append(
            PairResult(
                pair_id=pair.pair_id,
                facade_id=pair.facade_id,
                year_a=pair.year_a,
                year_b=pair.year_b,
                status=status,
                source_mask_path=str(src_mask_path),
                target_mask_path="" if tgt_mask_path is None else str(tgt_mask_path),
                warped_mask_path=warped_path,
                diff_map_path=diff_path,
                overlay_path=overlay_path,
                iou=iou,
                precision=precision,
                recall=recall,
            )
        )

    report_path = args.out_dir / "mask_adaptation_report.csv"
    write_report(report_path, results)

    n_ok = sum(1 for r in results if r.status.startswith("ok_"))
    print(f"Processed pairs: {len(results)}")
    print(f"Successful: {n_ok}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
