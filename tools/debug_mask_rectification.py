import argparse
import csv
import json
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
    status_quality: str
    H: Optional[np.ndarray]
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    iou_fg: float
    iou_edge: float
    overlay_path: str
    geom_path: str

STATUS_FAIL_MATCHES = "fail_not_enough_matches"
STATUS_FAIL_H = "fail_homography"
STATUS_SUCCESS = "success"
QUALITY_STRONG = "strong"
QUALITY_WEAK = "weak"
QUALITY_NONE = "none"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate rectification (homography) for facade pairs")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--facade-id", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--method", choices=["orb", "sift", "akaze", "loftr"], default="sift")
    p.add_argument("--max-side", type=int, default=1200, help="Resize longest side for matching (<=0 disables)")
    p.add_argument("--crop-nonzero", action="store_true", help="Crop away black padding before matching")
    p.add_argument("--clahe", action="store_true", help="Apply CLAHE to grayscale before matching")
    p.add_argument("--device", type=str, default="cuda", help="Device for LoFTR: cuda or cpu")
    return p.parse_args()

def read_pairs(path: Path, facade_id: Optional[str], limit: Optional[int]) -> List[PairEntry]:
    out: List[PairEntry] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if facade_id and row.get("facade_id") != facade_id:
                continue
            out.append(
                PairEntry(
                    pair_id=row.get("pair_id", ""),
                    facade_id=row.get("facade_id", ""),
                    year_a=row.get("year_a", ""),
                    year_b=row.get("year_b", ""),
                    mask_a=Path(row.get("mask_a", "")),
                    mask_b=Path(row.get("mask_b", "")),
                )
            )
            if limit is not None and len(out) >= limit:
                break
    return out

def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn)
    return (x * 255).clip(0, 255).astype(np.uint8)

def to_gray_uint8(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Failed to load image")
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = normalize_to_uint8(image)
    return image

def crop_nonzero(gray: np.ndarray, thr: int = 5, pad: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    mask = gray > thr
    if mask.mean() < 0.01:
        return gray, (0, 0)
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(gray.shape[1] - 1, x1 + pad); y1 = min(gray.shape[0] - 1, y1 + pad)
    return gray[y0:y1+1, x0:x1+1], (x0, y0)

def resize_max_side(gray: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    if max_side <= 0:
        return gray, 1.0
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray, 1.0
    s = max_side / float(m)
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA), s

def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def compute_edges(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 50, 150)

def build_T_offset(x0: int, y0: int) -> np.ndarray:
    return np.array([[1.0, 0.0, float(x0)],
                     [0.0, 1.0, float(y0)],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def build_S(scale: float) -> np.ndarray:
    return np.array([[scale, 0.0, 0.0],
                     [0.0, scale, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def create_detector(method: str):
    method = method.lower()
    if method == "sift":
        return cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01, edgeThreshold=10)
    if method == "akaze":
        return cv2.AKAZE_create()
    return cv2.ORB_create(nfeatures=12000, fastThreshold=5)

def match_features_opencv(des_a: np.ndarray, des_b: np.ndarray, method: str) -> List[cv2.DMatch]:
    if des_a is None or des_b is None:
        return []
    method = method.lower()
    norm = cv2.NORM_L2 if method in ("sift",) else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    ratio = 0.75 if method == "sift" else 0.8
    good: List[cv2.DMatch] = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def estimate_homography(pts_a: np.ndarray, pts_b: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    if pts_a.shape[0] < 8:
        return None, 0
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, mask = cv2.findHomography(
        pts_a, pts_b, method,
        ransacReprojThreshold=3.0,
        confidence=0.999,
        maxIters=10000,
    )
    num_inliers = int(mask.sum()) if mask is not None else 0
    return H, num_inliers

def loftr_correspondences(gray_a: np.ndarray, gray_b: np.ndarray, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    try:
        import torch
        from kornia.feature import LoFTR
    except Exception as e:
        raise RuntimeError("LoFTR requires torch + kornia. Install: pip install kornia") from e

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    matcher = LoFTR(pretrained="outdoor").to(dev).eval()

    t0 = torch.from_numpy(gray_a).float()[None, None] / 255.0
    t1 = torch.from_numpy(gray_b).float()[None, None] / 255.0
    with torch.inference_mode():
        out = matcher({"image0": t0.to(dev), "image1": t1.to(dev)})

    mk0 = out["keypoints0"].detach().cpu().numpy()
    mk1 = out["keypoints1"].detach().cpu().numpy()
    conf = out.get("confidence", None)
    if conf is not None:
        conf = conf.detach().cpu().numpy()
        keep = conf >= 0.4
        mk0 = mk0[keep]
        mk1 = mk1[keep]
    return mk0.reshape(-1, 1, 2).astype(np.float32), mk1.reshape(-1, 1, 2).astype(np.float32)

def foreground_mask_from_black(gray: np.ndarray, thr: int = 5) -> np.ndarray:
    return gray > thr

def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    uni = np.logical_or(mask_a, mask_b).sum()
    return 0.0 if uni == 0 else float(inter) / float(uni)

def make_overlay(mask_a: np.ndarray, mask_b: np.ndarray,
                warp_a: Optional[np.ndarray], warp_edge_a: Optional[np.ndarray],
                metrics_text: str) -> np.ndarray:
    h = max(mask_a.shape[0], mask_b.shape[0])
    w = mask_a.shape[1] + mask_b.shape[1] + mask_b.shape[1]
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[: mask_a.shape[0], : mask_a.shape[1]] = mask_a
    start_b = mask_a.shape[1]
    canvas[: mask_b.shape[0], start_b : start_b + mask_b.shape[1]] = mask_b

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

def save_overlay(out_dir: Path, facade_id: str, year_a: str, year_b: str, overlay: np.ndarray) -> str:
    d = out_dir / facade_id
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / f"pair_{year_a}_{year_b}_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    return str(out_path)

def save_geom(out_dir: Path, facade_id: str, year_a: str, year_b: str,
              H: Optional[np.ndarray], num_matches: int, num_inliers: int,
              inlier_ratio: float, iou_fg: Optional[float], iou_edge: Optional[float],
              status_quality: str) -> str:
    d = out_dir / "geom" / facade_id
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / f"{year_a}_{year_b}.json"

    def _maybe_int(v: str):
        try:
            return int(v)
        except ValueError:
            return v

    payload = {
        "facade_id": facade_id,
        "year_a": _maybe_int(year_a),
        "year_b": _maybe_int(year_b),
        "H": H.tolist() if H is not None else None,
        "num_matches": int(num_matches),
        "num_inliers": int(num_inliers),
        "inlier_ratio": float(inlier_ratio),
        "iou_fg": None if iou_fg is None else float(iou_fg),
        "iou_edge": None if iou_edge is None else float(iou_edge),
        "status_quality": status_quality,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return str(out_path)

def write_report(out_path: Path, results: Sequence[DebugResult]):
    fieldnames = [
        "pair_id","facade_id","year_a","year_b","status","status_quality",
        "num_matches","num_inliers","inlier_ratio","iou_fg","iou_edge","overlay_path","geom_path"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "pair_id": r.pair_id, "facade_id": r.facade_id, "year_a": r.year_a, "year_b": r.year_b,
                "status": r.status, "status_quality": r.status_quality,
                "num_matches": r.num_matches, "num_inliers": r.num_inliers,
                "inlier_ratio": f"{r.inlier_ratio:.6f}",
                "iou_fg": f"{r.iou_fg:.6f}", "iou_edge": f"{r.iou_edge:.6f}",
                "overlay_path": r.overlay_path, "geom_path": r.geom_path
            })

def quality_from_inliers(num_matches: int, num_inliers: int) -> str:
    ratio = (num_inliers / float(num_matches)) if num_matches > 0 else 0.0
    if num_inliers >= 40 or (num_inliers >= 25 and ratio >= 0.20):
        return QUALITY_STRONG
    if num_inliers >= 15 and ratio >= 0.12:
        return QUALITY_WEAK
    return QUALITY_NONE

def process_pair(entry: PairEntry, out_dir: Path, method: str, max_side: int,
                crop_nz: bool, use_clahe: bool, device: str) -> DebugResult:
    try:
        img_a = cv2.imread(str(entry.mask_a), cv2.IMREAD_UNCHANGED)
        img_b = cv2.imread(str(entry.mask_b), cv2.IMREAD_UNCHANGED)
        gray_a0 = to_gray_uint8(img_a)
        gray_b0 = to_gray_uint8(img_b)
    except Exception:
        return DebugResult(entry.pair_id, entry.facade_id, entry.year_a, entry.year_b,
                          "fail_load", QUALITY_NONE, None, 0, 0, 0.0, 0.0, 0.0, "", "")

    off_a = (0, 0); off_b = (0, 0)
    gray_a = gray_a0; gray_b = gray_b0
    if crop_nz:
        gray_a, off_a = crop_nonzero(gray_a0)
        gray_b, off_b = crop_nonzero(gray_b0)

    gray_a_s, s_a = resize_max_side(gray_a, max_side)
    gray_b_s, s_b = resize_max_side(gray_b, max_side)
    if use_clahe:
        gray_a_s = apply_clahe(gray_a_s)
        gray_b_s = apply_clahe(gray_b_s)

    T_a = build_T_offset(off_a[0], off_a[1])
    T_b = build_T_offset(off_b[0], off_b[1])
    S_a = build_S(s_a)
    S_b = build_S(s_b)

    num_matches = 0
    num_inliers = 0
    H_full = None

    if method == "loftr":
        pts_a, pts_b = loftr_correspondences(gray_a_s, gray_b_s, device=device)
        num_matches = int(pts_a.shape[0])
        if num_matches >= 8:
            H_res, num_inliers = estimate_homography(pts_a, pts_b)
            if H_res is not None:
                H_full = T_b @ np.linalg.inv(S_b) @ H_res @ S_a @ np.linalg.inv(T_a)
    else:
        det = create_detector(method)
        kp_a, des_a = det.detectAndCompute(gray_a_s, None)
        kp_b, des_b = det.detectAndCompute(gray_b_s, None)
        matches = match_features_opencv(des_a, des_b, method)
        num_matches = len(matches)
        if num_matches >= 8:
            pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H_res, num_inliers = estimate_homography(pts_a, pts_b)
            if H_res is not None:
                H_full = T_b @ np.linalg.inv(S_b) @ H_res @ S_a @ np.linalg.inv(T_a)

    inlier_ratio = (num_inliers / float(num_matches)) if num_matches > 0 else 0.0
    quality = QUALITY_NONE if H_full is None else quality_from_inliers(num_matches, num_inliers)

    edge_a = compute_edges(gray_a0)
    edge_b = compute_edges(gray_b0)

    warp_a = None
    warp_edge_a = None
    iou_fg = 0.0
    iou_edge = 0.0
    if H_full is not None:
        Hb, Wb = gray_b0.shape
        warp_a = cv2.warpPerspective(gray_a0, H_full, (Wb, Hb))
        warp_edge_a = cv2.warpPerspective(edge_a, H_full, (Wb, Hb))
        if quality in {QUALITY_STRONG, QUALITY_WEAK}:
            fg_a = foreground_mask_from_black(warp_a)
            fg_b = foreground_mask_from_black(gray_b0)
            iou_fg = compute_iou(fg_a, fg_b)
            iou_edge = compute_iou(warp_edge_a > 0, edge_b > 0)

    status = STATUS_FAIL_MATCHES if num_matches < 8 else (STATUS_SUCCESS if quality in {QUALITY_STRONG, QUALITY_WEAK} else STATUS_FAIL_H)

    metrics_text = (
        f"method:{method}\n"
        f"matches:{num_matches}\n"
        f"inliers:{num_inliers}\n"
        f"inlier_ratio:{inlier_ratio:.3f}\n"
        f"iou_fg:{iou_fg:.3f}\n"
        f"iou_edge:{iou_edge:.3f}\n"
        f"quality:{quality}"
    )
    overlay = make_overlay(gray_a0, gray_b0, warp_a, warp_edge_a, metrics_text)
    overlay_path = save_overlay(out_dir, entry.facade_id, entry.year_a, entry.year_b, overlay)
    geom_path = save_geom(out_dir, entry.facade_id, entry.year_a, entry.year_b, H_full,
                          num_matches, num_inliers, inlier_ratio,
                          iou_fg if H_full is not None else None,
                          iou_edge if H_full is not None else None,
                          quality)

    return DebugResult(entry.pair_id, entry.facade_id, entry.year_a, entry.year_b, status, quality,
                       H_full, num_matches, num_inliers, inlier_ratio, iou_fg, iou_edge, overlay_path, geom_path)

def main():
    args = parse_args()
    pairs = read_pairs(args.pairs, args.facade_id, args.limit)
    results: List[DebugResult] = []
    for e in pairs:
        results.append(process_pair(e, args.out_dir, args.method, args.max_side, args.crop_nonzero, args.clahe, args.device))
    write_report(args.out_dir / "report.csv", results)

if __name__ == "__main__":
    main()
