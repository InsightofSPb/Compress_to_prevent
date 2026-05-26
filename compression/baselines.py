from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .io import load_rgb_image, read_csv_rows, write_csv_rows

PIXEL_METHODS = {"absdiff_l1", "absdiff_l2", "grayscale_absdiff", "ssim_change"}
DEEP_METHODS = {"dinov2_patch_cosine", "lpips_change"}
OPTIONAL_METHODS = {"temporal_semantic_feature_distance", "orb_feature_density"}
REQUIRED_FULL_METHODS = {"absdiff_l1", "absdiff_l2", "grayscale_absdiff", "ssim_change", "dinov2_patch_cosine", "lpips_change"}


def _np():
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Baselines require numpy. Install eval deps with: pip install -r requirements-eval.txt") from exc
    return np


def _image_to_np(path: Path):
    np = _np()
    w, h, payload = load_rgb_image(path)
    arr = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, 3)).astype(np.float32)
    return arr


def _tile_grid(h: int, w: int, tile_size: int) -> Iterable[Tuple[int, int, slice, slice]]:
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            yield x0 // tile_size, y0 // tile_size, slice(y0, min(y0 + tile_size, h)), slice(x0, min(x0 + tile_size, w))


def _score_type(method: str) -> str:
    if method == "lpips_change":
        return "perceptual_change_score"
    if method in {"dinov2_patch_cosine", "temporal_semantic_feature_distance", "orb_feature_density"}:
        return "feature_change_score"
    return "baseline_change_score"


def _require_skimage() -> None:
    try:
        import skimage.metrics  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Method 'ssim_change' requires scikit-image. Install eval deps with: pip install -r requirements-eval.txt") from exc


def _ssim_map(prev, curr):
    from skimage.metrics import structural_similarity

    sim, full = structural_similarity(prev / 255.0, curr / 255.0, channel_axis=2, data_range=1.0, full=True)
    _ = sim
    return 1.0 - full.astype(np.float32)


def _load_dinov2(device: str, model_name: str, cache_dir: Optional[Path], weights_path: Optional[Path]):
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("Method 'dinov2_patch_cosine' requires torch/torchvision. Install eval deps with: pip install -r requirements-eval.txt") from exc

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))

    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DINOv2 model '{model_name}'. Ensure internet access or provide --dinov2-weights-path / --dinov2-cache-dir. Original error: {exc}"
        ) from exc

    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    model = model.to(device)
    model.eval()
    return torch, F, model


def _dino_features(torch, model, img, cache_path: Optional[Path]):
    if cache_path and cache_path.exists():
        return np.load(cache_path)

    x = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    x = x.unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        feats = model.forward_features(x)
    patch = feats.get("x_norm_patchtokens")
    if patch is None:
        raise RuntimeError("DINOv2 forward_features output missing patch tokens.")
    patch_np = patch[0].detach().cpu().numpy()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, patch_np)
    return patch_np


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
    lpips_net: str = "alex",
    temporal_features_csv: Optional[Path] = None,
    artifact_index_csv: Optional[Path] = None,
    skip_deep_baselines: bool = False,
) -> List[Dict[str, object]]:
    np = _np()
    methods = [m.strip() for m in methods if m.strip()]
    if not methods:
        raise ValueError("No methods provided")

    if not skip_deep_baselines:
        for req in ["dinov2_patch_cosine", "lpips_change"]:
            if req in methods:
                pass

    if "ssim_change" in methods:
        _require_skimage()

    torch = F = dino_model = lpips_model = None
    if "dinov2_patch_cosine" in methods and not skip_deep_baselines:
        torch, F, dino_model = _load_dinov2(device=device, model_name=dinov2_model_name, cache_dir=dinov2_cache_dir, weights_path=dinov2_weights_path)

    if "lpips_change" in methods and not skip_deep_baselines:
        try:
            import lpips
            import torch as torch_lp
        except Exception as exc:
            raise RuntimeError("Method 'lpips_change' requires lpips+torch. Install eval deps with: pip install -r requirements-eval.txt") from exc
        lpips_model = lpips.LPIPS(net=lpips_net).to(device)
        lpips_model.eval()
        if torch is None:
            torch = torch_lp

    rows = read_csv_rows(residual_manifest_csv)
    out_rows: List[Dict[str, object]] = []

    for row in rows:
        pair_id = row["pair_id"]
        facade_id = row.get("facade_id", "")
        split = row.get("split", "")
        prev = _image_to_np(Path(row["prev_aligned_path"]))
        curr = _image_to_np(Path(row["curr_image_path"]))
        h, w = prev.shape[:2]
        if curr.shape[:2] != (h, w):
            raise ValueError(f"Image shape mismatch for pair {pair_id}: prev={prev.shape}, curr={curr.shape}")

        abs_diff = np.abs(curr - prev)
        gray_diff = np.abs((0.299 * curr[:, :, 0] + 0.587 * curr[:, :, 1] + 0.114 * curr[:, :, 2]) - (0.299 * prev[:, :, 0] + 0.587 * prev[:, :, 1] + 0.114 * prev[:, :, 2]))
        ssim_change_map = _ssim_map(prev, curr) if "ssim_change" in methods else None

        dino_dist = None
        if "dinov2_patch_cosine" in methods:
            if skip_deep_baselines:
                continue
            key_prev = hashlib.sha1(f"{pair_id}|prev|{tile_size}".encode()).hexdigest()
            key_curr = hashlib.sha1(f"{pair_id}|curr|{tile_size}".encode()).hexdigest()
            prev_cache = feature_cache_dir / f"{key_prev}.npy" if feature_cache_dir else None
            curr_cache = feature_cache_dir / f"{key_curr}.npy" if feature_cache_dir else None
            pfeat = _dino_features(torch, dino_model, prev, prev_cache)
            cfeat = _dino_features(torch, dino_model, curr, curr_cache)
            p = pfeat / (np.linalg.norm(pfeat, axis=1, keepdims=True) + 1e-8)
            c = cfeat / (np.linalg.norm(cfeat, axis=1, keepdims=True) + 1e-8)
            dino_dist = 1.0 - np.sum(p * c, axis=1)
            patch_side = int(round(math.sqrt(len(dino_dist))))

        for tx, ty, ys, xs in _tile_grid(h, w, tile_size):
            for method in methods:
                if skip_deep_baselines and method in DEEP_METHODS:
                    continue
                if method == "absdiff_l1":
                    score = float(abs_diff[ys, xs, :].mean())
                elif method == "absdiff_l2":
                    score = float(np.sqrt(np.mean((curr[ys, xs, :] - prev[ys, xs, :]) ** 2)))
                elif method == "grayscale_absdiff":
                    score = float(gray_diff[ys, xs].mean())
                elif method == "ssim_change":
                    score = float(ssim_change_map[ys, xs, :].mean() if ssim_change_map.ndim == 3 else ssim_change_map[ys, xs].mean())
                elif method == "lpips_change":
                    patch_prev = prev[ys, xs, :]
                    patch_curr = curr[ys, xs, :]
                    tens_prev = torch.from_numpy(patch_prev.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                    tens_curr = torch.from_numpy(patch_curr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                    tens_prev = (tens_prev / 127.5) - 1.0
                    tens_curr = (tens_curr / 127.5) - 1.0
                    with torch.no_grad():
                        score = float(lpips_model(tens_prev, tens_curr).item())
                elif method == "dinov2_patch_cosine":
                    px = min(tx, patch_side - 1)
                    py = min(ty, patch_side - 1)
                    score = float(dino_dist[py * patch_side + px])
                else:
                    continue

                out_rows.append({
                    "pair_id": pair_id,
                    "facade_id": facade_id,
                    "split": split,
                    "method": method,
                    "score_type": _score_type(method),
                    "tile_x": tx,
                    "tile_y": ty,
                    "tile_score": score,
                    "tile_size": tile_size,
                })

    fields = ["pair_id", "facade_id", "split", "method", "score_type", "tile_x", "tile_y", "tile_score", "tile_size"]
    write_csv_rows(out_scores_csv, fields, out_rows)
    return out_rows
