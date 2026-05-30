from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .io import load_rgb_image, read_csv_rows, write_csv_rows
from .residuals import load_valid_mask

DEEP_METHODS = {"dinov2_patch_cosine", "lpips_change"}


def _np():
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Baselines require numpy. Install eval deps with: pip install -r requirements-eval.txt") from exc
    return np


def _image_to_np(path: Path):
    np = _np()
    w, h, payload = load_rgb_image(path)
    return np.frombuffer(payload, dtype=np.uint8).reshape((h, w, 3)).astype(np.float32)


def _valid_to_np(row: Dict[str, str], height: int, width: int):
    np = _np()
    path_value = row.get("valid_mask_path", "")
    if not path_value:
        return np.ones((height, width), dtype=bool)
    threshold = int(float(row.get("valid_threshold", "0") or 0))
    payload = load_valid_mask(Path(path_value), (width, height), threshold=threshold)
    return np.frombuffer(payload, dtype=np.uint8).reshape((height, width)).astype(bool)


def _tile_grid(h: int, w: int, tile_size: int) -> Iterable[Tuple[int, int, slice, slice]]:
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            yield x0 // tile_size, y0 // tile_size, slice(y0, min(y0 + tile_size, h)), slice(x0, min(x0 + tile_size, w))


def _score_type(method: str) -> str:
    if method == "lpips_change":
        return "perceptual_change_score"
    if method == "dinov2_patch_cosine":
        return "feature_change_score"
    return "baseline_change_score"


def _ssim_map(prev, curr):
    np = _np()
    try:
        from skimage.metrics import structural_similarity
    except Exception as exc:
        raise RuntimeError("Method 'ssim_change' requires scikit-image. Install eval deps with: pip install -r requirements-eval.txt") from exc
    _, full = structural_similarity(prev / 255.0, curr / 255.0, channel_axis=2, data_range=1.0, full=True)
    return 1.0 - full.astype(np.float32)


def _load_dinov2(device: str, model_name: str, cache_dir: Optional[Path], weights_path: Optional[Path], repo_dir: Optional[Path]):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Method 'dinov2_patch_cosine' requires torch/torchvision. Install eval deps with: pip install -r requirements-eval.txt") from exc
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))

    source = "local" if repo_dir else "github"
    repo_or_dir = str(repo_dir) if repo_dir else "facebookresearch/dinov2"
    try:
        model = torch.hub.load(repo_or_dir, model_name, source=source)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DINOv2 '{model_name}'. For offline mode pass --dinov2-repo-dir pointing to local dinov2 clone (and optional --dinov2-weights-path). Original error: {exc}"
        ) from exc

    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return torch, model


def _dino_features(torch, model, img, device: str, cache_path: Optional[Path]):
    np = _np()
    if cache_path and cache_path.exists():
        dat = np.load(cache_path, allow_pickle=True).item()
        return dat["features"], int(dat["grid_h"]), int(dat["grid_w"])

    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        feats = model.forward_features(x)
    patch = feats.get("x_norm_patchtokens")
    if patch is None:
        raise RuntimeError("DINOv2 forward_features output missing 'x_norm_patchtokens'.")
    n = int(patch.shape[1])
    h_p = img.shape[0] // 14
    w_p = img.shape[1] // 14
    if h_p * w_p != n:
        h_p = 1
        while h_p <= n and n % h_p != 0:
            h_p += 1
        w_p = max(1, n // h_p)
    arr = patch[0].detach().cpu().numpy().reshape(h_p, w_p, -1)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, {"features": arr, "grid_h": h_p, "grid_w": w_p}, allow_pickle=True)
    return arr, h_p, w_p


def _pad_image_tile(tile, tile_size: int):
    np = _np()
    height, width = tile.shape[:2]
    if height == tile_size and width == tile_size:
        return tile
    pad_h = tile_size - height
    pad_w = tile_size - width
    return np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _score_lpips_tiles(torch, model, prev_tiles, curr_tiles, device: str, batch_size: int):
    np = _np()
    scores: List[float] = []
    for start in range(0, len(prev_tiles), batch_size):
        p = np.stack(prev_tiles[start:start + batch_size]).transpose(0, 3, 1, 2)
        c = np.stack(curr_tiles[start:start + batch_size]).transpose(0, 3, 1, 2)
        tprev = torch.from_numpy(p).float().to(device) / 127.5 - 1.0
        tcurr = torch.from_numpy(c).float().to(device) / 127.5 - 1.0
        with torch.no_grad():
            values = model(tprev, tcurr).reshape(-1).detach().cpu().numpy()
        scores.extend(float(value) for value in values.tolist())
    return scores


def compute_baseline_tile_scores(
    residual_manifest_csv: Path,
    out_scores_csv: Path,
    methods: Sequence[str],
    tile_size: int = 32,
    device: str = "cpu",
    feature_cache_dir: Optional[Path] = None,
    dinov2_model_name: str = "dinov2_vitb14",
    dinov2_cache_dir: Optional[Path] = None,
    dinov2_weights_path: Optional[Path] = None,
    dinov2_repo_dir: Optional[Path] = None,
    lpips_net: str = "alex",
    temporal_features_csv: Optional[Path] = None,
    artifact_index_csv: Optional[Path] = None,
    skip_deep_baselines: bool = False,
    min_valid_ratio: float = 0.50,
    include_splits: Optional[Sequence[str]] = None,
    deep_batch_size: int = 128,
    show_progress: bool = False,
) -> List[Dict[str, object]]:
    """Compute tile scores on valid aligned pixels only.

    Invalid pixels are neutralised by setting ``curr=prev`` before feature
    extraction. Per-pixel baselines are averaged over valid pixels only. Tiles
    below ``min_valid_ratio`` are excluded. LPIPS is evaluated in tile batches;
    DINOv2 is evaluated as whole-image patch features and pooled to the common
    tile grid.
    """
    np = _np()
    _ = (temporal_features_csv, artifact_index_csv)
    methods = [m.strip() for m in methods if m.strip()]
    if not methods:
        raise ValueError("No methods provided")
    if not 0.0 <= min_valid_ratio <= 1.0:
        raise ValueError("min_valid_ratio must be in [0, 1]")
    if deep_batch_size <= 0:
        raise ValueError("deep_batch_size must be positive")

    torch = dino_model = lpips_model = None
    if "dinov2_patch_cosine" in methods and not skip_deep_baselines:
        torch, dino_model = _load_dinov2(device, dinov2_model_name, dinov2_cache_dir, dinov2_weights_path, dinov2_repo_dir)

    if "lpips_change" in methods and not skip_deep_baselines:
        try:
            import lpips
            import torch as torch_lp
        except Exception as exc:
            raise RuntimeError("Method 'lpips_change' requires lpips+torch. Install eval deps with: pip install -r requirements-eval.txt") from exc
        lpips_model = lpips.LPIPS(net=lpips_net).to(device).eval()
        if torch is None:
            torch = torch_lp

    rows = read_csv_rows(residual_manifest_csv)
    if include_splits:
        permitted = set(include_splits)
        rows = [row for row in rows if row.get("split", "") in permitted]
    out_rows: List[Dict[str, object]] = []
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(rows, desc="Scoring temporal pairs", unit="pair", disable=not show_progress)
    except Exception:
        iterator = rows

    for row in iterator:
        pair_id = row["pair_id"]
        facade_id = row.get("facade_id", "")
        split = row.get("split", "")
        prev = _image_to_np(Path(row["prev_aligned_path"]))
        curr = _image_to_np(Path(row["curr_image_path"]))
        h, w = prev.shape[:2]
        if curr.shape[:2] != (h, w):
            raise ValueError(f"Image shape mismatch for pair {pair_id}: prev={prev.shape}, curr={curr.shape}")
        valid = _valid_to_np(row, h, w)
        curr_neutral = curr.copy()
        curr_neutral[~valid] = prev[~valid]

        abs_diff = np.abs(curr_neutral - prev)
        gray_prev = 0.299 * prev[:, :, 0] + 0.587 * prev[:, :, 1] + 0.114 * prev[:, :, 2]
        gray_curr = 0.299 * curr_neutral[:, :, 0] + 0.587 * curr_neutral[:, :, 1] + 0.114 * curr_neutral[:, :, 2]
        gray_diff = np.abs(gray_curr - gray_prev)
        ssim_change_map = _ssim_map(prev, curr_neutral) if "ssim_change" in methods else None

        dino_dist = dino_h = dino_w = None
        if "dinov2_patch_cosine" in methods and not skip_deep_baselines:
            key_prev = hashlib.sha1(f"{pair_id}|prev".encode()).hexdigest()
            key_curr = hashlib.sha1(f"{pair_id}|curr_valid_neutral".encode()).hexdigest()
            prev_cache = feature_cache_dir / f"{key_prev}.npy" if feature_cache_dir else None
            curr_cache = feature_cache_dir / f"{key_curr}.npy" if feature_cache_dir else None
            pfeat, ph, pw = _dino_features(torch, dino_model, prev, device, prev_cache)
            cfeat, ch, cw = _dino_features(torch, dino_model, curr_neutral, device, curr_cache)
            hmin, wmin = min(ph, ch), min(pw, cw)
            p = pfeat[:hmin, :wmin, :]
            c = cfeat[:hmin, :wmin, :]
            p = p / (np.linalg.norm(p, axis=2, keepdims=True) + 1e-8)
            c = c / (np.linalg.norm(c, axis=2, keepdims=True) + 1e-8)
            dino_dist = 1.0 - np.sum(p * c, axis=2)
            dino_h, dino_w = hmin, wmin

        eligible_tiles = []
        lpips_prev_tiles = []
        lpips_curr_tiles = []
        for tx, ty, ys, xs in _tile_grid(h, w, tile_size):
            valid_patch = valid[ys, xs]
            valid_count = int(valid_patch.sum())
            tile_pixels = int(valid_patch.size)
            valid_ratio = valid_count / max(tile_pixels, 1)
            if valid_ratio < min_valid_ratio or valid_count == 0:
                continue
            common = {
                "pair_id": pair_id,
                "facade_id": facade_id,
                "split": split,
                "tile_x": tx,
                "tile_y": ty,
                "tile_size": tile_size,
                "valid_pixel_count": valid_count,
                "valid_ratio": valid_ratio,
            }
            eligible_tiles.append((common, ys, xs))
            for method in methods:
                if skip_deep_baselines and method in DEEP_METHODS:
                    continue
                if method == "absdiff_l1":
                    score = float(abs_diff[ys, xs, :][valid_patch].mean())
                elif method == "absdiff_l2":
                    squared = (curr_neutral[ys, xs, :] - prev[ys, xs, :]) ** 2
                    score = float(np.sqrt(np.mean(squared[valid_patch])))
                elif method == "grayscale_absdiff":
                    score = float(gray_diff[ys, xs][valid_patch].mean())
                elif method == "ssim_change":
                    score = float(ssim_change_map[ys, xs][valid_patch].mean())
                elif method == "dinov2_patch_cosine":
                    if dino_dist is None:
                        continue
                    py0 = int(ys.start / h * dino_h)
                    py1 = max(py0 + 1, int(ys.stop / h * dino_h))
                    px0 = int(xs.start / w * dino_w)
                    px1 = max(px0 + 1, int(xs.stop / w * dino_w))
                    score = float(dino_dist[py0:py1, px0:px1].mean())
                elif method == "lpips_change":
                    continue
                else:
                    continue
                out_rows.append({**common, "method": method, "score_type": _score_type(method), "tile_score": score})
            if "lpips_change" in methods and not skip_deep_baselines:
                lpips_prev_tiles.append(_pad_image_tile(prev[ys, xs, :], tile_size))
                lpips_curr_tiles.append(_pad_image_tile(curr_neutral[ys, xs, :], tile_size))

        if lpips_prev_tiles:
            lpips_scores = _score_lpips_tiles(torch, lpips_model, lpips_prev_tiles, lpips_curr_tiles, device, deep_batch_size)
            for (common, _, _), score in zip(eligible_tiles, lpips_scores):
                out_rows.append({**common, "method": "lpips_change", "score_type": _score_type("lpips_change"), "tile_score": score})
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(split=split, scores=len(out_rows))

    fields = ["pair_id", "facade_id", "split", "method", "score_type", "tile_x", "tile_y", "tile_score", "tile_size", "valid_pixel_count", "valid_ratio"]
    write_csv_rows(out_scores_csv, fields, out_rows)
    return out_rows
