import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class PairEntry:
    pair_id: str
    facade_id: str
    year_a: str
    year_b: str
    mask_a: Path
    mask_b: Path


@dataclass
class DebugResult:
    pair_id: str
    facade_id: str
    year_a: str
    year_b: str
    status: str
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    iou_fg: float
    iou_edge: float
    overlay_path: str


STATUS_FAIL_MATCHES = "fail_not_enough_matches"
STATUS_FAIL_H = "fail_homography"
STATUS_SUCCESS = "success"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug mask rectification with feature matching")
    parser.add_argument("--pairs", type=Path, required=True, help="Path to pairs_consecutive.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--facade-id", type=str, default=None, help="Optional facade filter")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs")
    parser.add_argument("--method", type=str, choices=["orb", "sift"], default="orb", help="Feature detector method")
    return parser.parse_args()


def read_pairs(path: Path, facade_id: Optional[str], limit: Optional[int]) -> List[PairEntry]:
    entries: List[PairEntry] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if facade_id and row.get("facade_id") != facade_id:
                continue
            entry = PairEntry(
                pair_id=row.get("pair_id", ""),
                facade_id=row.get("facade_id", ""),
                year_a=row.get("year_a", ""),
                year_b=row.get("year_b", ""),
                mask_a=Path(row.get("mask_a", "")),
                mask_b=Path(row.get("mask_b", "")),
            )
            entries.append(entry)
            if limit is not None and len(entries) >= limit:
                break
    return entries


def to_gray_uint8(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Failed to load image")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = normalize_to_uint8(image)
    return image


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    img_float = image.astype(np.float32)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    if max_val - min_val < 1e-6:
        return np.zeros_like(img_float, dtype=np.uint8)
    norm = (img_float - min_val) / (max_val - min_val)
    return (norm * 255).clip(0, 255).astype(np.uint8)


def compute_edges(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 50, 150)


def create_detector(method: str):
    if method == "sift":
        return cv2.SIFT_create()
    return cv2.ORB_create(nfeatures=5000)


def match_features(des_a: np.ndarray, des_b: np.ndarray, method: str) -> List[cv2.DMatch]:
    if des_a is None or des_b is None:
        return []
    norm = cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
    knn_matches = matcher.knnMatch(des_a, des_b, k=2)
    good: List[cv2.DMatch] = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def find_homography(kp_a, kp_b, matches: Sequence[cv2.DMatch]) -> Tuple[Optional[np.ndarray], int]:
    if not matches:
        return None, 0
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransacReprojThreshold=4.0)
    num_inliers = int(mask.sum()) if mask is not None else 0
    return H, num_inliers


def foreground_mask(img: np.ndarray) -> np.ndarray:
    values, counts = np.unique(img, return_counts=True)
    bg_value = values[np.argmax(counts)] if len(values) > 0 else 0
    return img != bg_value


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def make_overlay(mask_a: np.ndarray, mask_b: np.ndarray, warp_a: Optional[np.ndarray], warp_edge_a: Optional[np.ndarray], metrics_text: str) -> np.ndarray:
    h = max(mask_a.shape[0], mask_b.shape[0])
    w = mask_a.shape[1] + mask_b.shape[1] + mask_b.shape[1]
    canvas = np.zeros((h, w), dtype=np.uint8)

    # First panel: mask_a
    canvas[: mask_a.shape[0], : mask_a.shape[1]] = mask_a

    # Second panel: mask_b
    start_b = mask_a.shape[1]
    canvas[: mask_b.shape[0], start_b : start_b + mask_b.shape[1]] = mask_b

    # Third panel: overlay
    start_overlay = start_b + mask_b.shape[1]
    overlay_gray = np.zeros_like(mask_b)
    if warp_a is not None:
        overlay_gray = cv2.addWeighted(overlay_gray, 1.0, warp_a, 1.0, 0)
    overlay_color = cv2.cvtColor(mask_b, cv2.COLOR_GRAY2BGR)
    if warp_edge_a is not None:
        edges_colored = np.zeros_like(overlay_color)
        edges_colored[:, :, 2] = warp_edge_a
        overlay_color = cv2.addWeighted(overlay_color, 0.7, edges_colored, 0.3, 0)
    overlay_color[: overlay_gray.shape[0], : overlay_gray.shape[1], 1] = cv2.bitwise_or(
        overlay_color[: overlay_gray.shape[0], : overlay_gray.shape[1], 1], overlay_gray
    )
    overlay_panel = cv2.cvtColor(overlay_color, cv2.COLOR_BGR2GRAY)

    canvas[: overlay_panel.shape[0], start_overlay : start_overlay + overlay_panel.shape[1]] = overlay_panel

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    y0 = 20
    for idx, line in enumerate(metrics_text.split("\n")):
        cv2.putText(canvas_bgr, line, (10, y0 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas_bgr


def process_pair(entry: PairEntry, out_dir: Path, detector_method: str) -> DebugResult:
    try:
        mask_a_img = cv2.imread(str(entry.mask_a), cv2.IMREAD_UNCHANGED)
        mask_b_img = cv2.imread(str(entry.mask_b), cv2.IMREAD_UNCHANGED)
        mask_a_gray = to_gray_uint8(mask_a_img)
        mask_b_gray = to_gray_uint8(mask_b_img)
    except Exception:
        return DebugResult(
            pair_id=entry.pair_id,
            facade_id=entry.facade_id,
            year_a=entry.year_a,
            year_b=entry.year_b,
            status="fail_load",
            num_matches=0,
            num_inliers=0,
            inlier_ratio=0.0,
            iou_fg=0.0,
            iou_edge=0.0,
            overlay_path="",
        )

    edge_a = compute_edges(mask_a_gray)
    edge_b = compute_edges(mask_b_gray)

    detector = create_detector(detector_method)
    kp_a, des_a = detector.detectAndCompute(edge_a, None)
    kp_b, des_b = detector.detectAndCompute(edge_b, None)

    matches = match_features(des_a, des_b, detector_method)
    num_matches = len(matches)

    if num_matches < 12:
        metrics_text = f"matches:{num_matches}\nstatus:{STATUS_FAIL_MATCHES}"
        overlay = make_overlay(mask_a_gray, mask_b_gray, None, None, metrics_text)
        overlay_path = save_overlay(out_dir, entry.facade_id, entry.year_a, entry.year_b, overlay)
        return DebugResult(
            pair_id=entry.pair_id,
            facade_id=entry.facade_id,
            year_a=entry.year_a,
            year_b=entry.year_b,
            status=STATUS_FAIL_MATCHES,
            num_matches=num_matches,
            num_inliers=0,
            inlier_ratio=0.0,
            iou_fg=0.0,
            iou_edge=0.0,
            overlay_path=overlay_path,
        )

    H, num_inliers = find_homography(kp_a, kp_b, matches)
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0

    if H is None or num_inliers < 20:
        metrics_text = f"matches:{num_matches}\ninliers:{num_inliers}\nstatus:{STATUS_FAIL_H}"
        overlay = make_overlay(mask_a_gray, mask_b_gray, None, None, metrics_text)
        overlay_path = save_overlay(out_dir, entry.facade_id, entry.year_a, entry.year_b, overlay)
        return DebugResult(
            pair_id=entry.pair_id,
            facade_id=entry.facade_id,
            year_a=entry.year_a,
            year_b=entry.year_b,
            status=STATUS_FAIL_H,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            iou_fg=0.0,
            iou_edge=0.0,
            overlay_path=overlay_path,
        )

    Hb, Wb = mask_b_gray.shape
    warp_a = cv2.warpPerspective(mask_a_gray, H, (Wb, Hb))
    warp_edge_a = cv2.warpPerspective(edge_a, H, (Wb, Hb))

    fg_a = foreground_mask(warp_a)
    fg_b = foreground_mask(mask_b_gray)
    iou_fg = compute_iou(fg_a, fg_b)
    iou_edge = compute_iou(warp_edge_a > 0, edge_b > 0)

    metrics_text = (
        f"matches:{num_matches}\n"
        f"inliers:{num_inliers}\n"
        f"inlier_ratio:{inlier_ratio:.3f}\n"
        f"iou_fg:{iou_fg:.3f}\n"
        f"iou_edge:{iou_edge:.3f}\n"
        f"status:{STATUS_SUCCESS}"
    )
    overlay = make_overlay(mask_a_gray, mask_b_gray, warp_a, warp_edge_a, metrics_text)
    overlay_path = save_overlay(out_dir, entry.facade_id, entry.year_a, entry.year_b, overlay)

    return DebugResult(
        pair_id=entry.pair_id,
        facade_id=entry.facade_id,
        year_a=entry.year_a,
        year_b=entry.year_b,
        status=STATUS_SUCCESS,
        num_matches=num_matches,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        iou_fg=iou_fg,
        iou_edge=iou_edge,
        overlay_path=overlay_path,
    )


def save_overlay(out_dir: Path, facade_id: str, year_a: str, year_b: str, overlay: np.ndarray) -> str:
    facade_dir = out_dir / facade_id
    facade_dir.mkdir(parents=True, exist_ok=True)
    filename = f"pair_{year_a}_{year_b}_overlay.png"
    out_path = facade_dir / filename
    cv2.imwrite(str(out_path), overlay)
    return str(out_path)


def write_report(out_path: Path, results: Sequence[DebugResult]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "facade_id",
        "year_a",
        "year_b",
        "status",
        "num_matches",
        "num_inliers",
        "inlier_ratio",
        "iou_fg",
        "iou_edge",
        "overlay_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "pair_id": res.pair_id,
                    "facade_id": res.facade_id,
                    "year_a": res.year_a,
                    "year_b": res.year_b,
                    "status": res.status,
                    "num_matches": res.num_matches,
                    "num_inliers": res.num_inliers,
                    "inlier_ratio": f"{res.inlier_ratio:.6f}",
                    "iou_fg": f"{res.iou_fg:.6f}",
                    "iou_edge": f"{res.iou_edge:.6f}",
                    "overlay_path": res.overlay_path,
                }
            )


def main():
    args = parse_args()
    pairs = read_pairs(args.pairs, args.facade_id, args.limit)
    results: List[DebugResult] = []
    for entry in pairs:
        result = process_pair(entry, args.out_dir, args.method)
        results.append(result)
    write_report(args.out_dir / "report.csv", results)


if __name__ == "__main__":
    main()
