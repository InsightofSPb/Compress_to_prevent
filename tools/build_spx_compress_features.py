import argparse
import json
import logging
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)

DEFAULT_DAMAGE_CLASSES = (
    "CRACK",
    "SPALLING",
    "DELAMINATION",
    "MISSING_ELEMENT",
    "WATER_STAIN",
    "EFFLORESCENCE",
    "CORROSION",
)


@dataclass
class LpossConfig:
    config_path: Path
    checkpoint_path: Path
    dataset_config: Path
    device: str
    damage_classes: Tuple[str, ...]
    mean_classes: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compressibility and delta-bpp features for superpixels."
    )
    parser.add_argument("--temporal-manifest", required=True, type=Path)
    parser.add_argument("--spx-cache", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--geom-dir", required=True, type=Path)
    parser.add_argument("--match-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--codec", default="png", type=str)
    parser.add_argument("--mode", default="mask", choices=["mask", "bbox"])
    parser.add_argument("--min-area", default=50, type=int)
    parser.add_argument(
        "--min-coverage",
        default=0.10,
        type=float,
        help="Minimum fraction of valid warped pixels inside the (year_b) superpixel mask to keep delta_bpp.",
    )
    parser.add_argument("--limit", default=None, type=int)

    parser.add_argument("--lposs-config", type=Path, default=None)
    parser.add_argument("--lposs-checkpoint", type=Path, default=None)
    parser.add_argument("--lposs-dataset-config", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--damage-classes",
        type=str,
        default=",".join(DEFAULT_DAMAGE_CLASSES),
        help="Comma-separated class names to aggregate as damage.",
    )
    parser.add_argument(
        "--mean-classes",
        type=str,
        default="",
        help="Comma-separated class names to report mean probabilities for.",
    )
    parser.add_argument(
        "--export-masks-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory to export per-image prediction masks and overlays. "
            "If set, the script saves original images, predicted masks, and overlays."
        ),
    )
    parser.add_argument(
        "--gt-mask-col",
        type=str,
        default="mask_path",
        help="Manifest column name that stores ground-truth mask paths for overlay export.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Alpha blending factor for overlay visualizations.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    required = {"facade_id", "year"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Manifest missing required columns: {missing}")

    if "image_path" in df.columns:
        path_col = "image_path"
    elif "full_path" in df.columns:
        path_col = "full_path"
    elif "mask_path" in df.columns:
        path_col = "mask_path"
    else:
        raise ValueError("Manifest must include image_path, mask_path, or full_path column")

    if path_col != "image_path":
        df = df.rename(columns={path_col: "image_path"})
    return df


def read_image(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def load_labels(spx_cache: Path, facade_id: str, year: int) -> np.ndarray:
    label_path = spx_cache / "facades" / str(facade_id) / "spx" / "spx" / f"{year}_labels.npz"
    if not label_path.exists():
        raise FileNotFoundError(f"Labels not found: {label_path}")
    with np.load(label_path) as data:
        return data["labels"].astype(np.int32)


def load_objects(spx_cache: Path, facade_id: str, year: int) -> pd.DataFrame:
    obj_path = spx_cache / "facades" / str(facade_id) / "spx" / "objs" / f"{year}_spx.parquet"
    if not obj_path.exists():
        raise FileNotFoundError(f"Objects parquet not found: {obj_path}")
    return pd.read_parquet(obj_path)


def load_geom(geom_dir: Path, facade_id: str, year_a: int, year_b: int) -> Tuple[str, Optional[np.ndarray]]:
    candidates = [
        geom_dir / str(facade_id) / f"{year_a}_{year_b}.json",
        geom_dir / f"{facade_id}_{year_a}_{year_b}.json",
    ]
    geom_path = next((p for p in candidates if p.exists()), None)
    if geom_path is None:
        LOGGER.warning("Geometry file not found for %s %s_%s", facade_id, year_a, year_b)
        return "none", None
    with geom_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    status_quality = data.get("status_quality", "none") or "none"
    H = data.get("H")
    if H is not None:
        H = np.array(H, dtype=float)
        if H.shape != (3, 3):
            LOGGER.warning("Invalid H shape in %s: %s", geom_path, H.shape)
            H = None
    else:
        H = None
    return status_quality, H


def load_matches(match_dir: Path, facade_id: str, year_a: int, year_b: int) -> Optional[Dict[str, object]]:
    """Load superpixel match file if present.

    NOTE: This is *not* the same as SIFT/ORB feature matches used for homography.
    Here we expect obj_id correspondences between superpixels: a=obj_id_a, b=obj_id_b.

    If the file is missing, we return None and the caller may fallback to computing
    delta_bpp for every superpixel in year_b using only the homography.
    """
    match_path = match_dir / f"{facade_id}" / f"{year_a}_{year_b}_match.json"
    if not match_path.exists():
        LOGGER.warning("Match file not found (will fallback to per-B-superpixel delta): %s", match_path)
        return None
    with match_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def encode_crop(crop: np.ndarray, codec: str) -> int:
    """Encode a crop in-memory and return encoded size in bytes.

    We fix codec parameters for determinism across runs / Pillow versions.
    """
    buffer = BytesIO()
    image = Image.fromarray(crop)

    codec_l = codec.lower()
    save_kwargs: Dict[str, object] = {}
    if codec_l == "png":
        save_kwargs.update({"compress_level": 9, "optimize": False})
    elif codec_l == "webp":
        save_kwargs.update({"lossless": True, "quality": 100, "method": 6})

    image.save(buffer, format=codec.upper(), **save_kwargs)
    return buffer.getbuffer().nbytes


def compute_bpp_excess(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    area_px: float,
    codec: str,
    mode: str,
    base_cache: Dict[Tuple[int, int, str], int],
) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2].copy()
    if mode == "mask":
        mask_crop = mask[y1:y2, x1:x2]
        crop[~mask_crop] = 0

    bytes_real = encode_crop(crop, codec)
    h, w = crop.shape[:2]
    key = (h, w, codec.lower())
    if key not in base_cache:
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base_cache[key] = encode_crop(base, codec)
    bytes_base = base_cache[key]

    bytes_excess_signed = float(bytes_real - bytes_base)
    bytes_excess = max(bytes_real - bytes_base, 0)

    area_px = max(float(area_px), 1.0)
    bbox_area = max((x2 - x1) * (y2 - y1), 1)

    denom_area = area_px if mode == "mask" else float(bbox_area)
    denom_area = max(float(denom_area), 1.0)

    bpp_excess = 8.0 * float(bytes_excess) / denom_area
    bpp_excess_signed = 8.0 * float(bytes_excess_signed) / denom_area
    bpp_bbox = 8.0 * float(bytes_real) / float(bbox_area)

    return {
        "bytes_real": float(bytes_real),
        "bytes_base": float(bytes_base),
        "bytes_excess": float(bytes_excess),
        "bytes_excess_signed": float(bytes_excess_signed),
        "bpp_excess": float(bpp_excess),
        "bpp_excess_signed": float(bpp_excess_signed),
        "bpp_excess_denom": float(denom_area),
        "bpp_bbox": float(bpp_bbox),
    }


def resolve_lposs_config(args: argparse.Namespace) -> Optional[LpossConfig]:
    if args.lposs_config is None or args.lposs_checkpoint is None:
        return None
    if args.lposs_dataset_config is None:
        raise ValueError("LPOSS enabled: --lposs-dataset-config is required")
    damage = tuple(x.strip() for x in args.damage_classes.split(",") if x.strip())
    mean_classes = tuple(x.strip() for x in args.mean_classes.split(",") if x.strip())
    return LpossConfig(
        config_path=args.lposs_config,
        checkpoint_path=args.lposs_checkpoint,
        dataset_config=args.lposs_dataset_config,
        device=args.device,
        damage_classes=damage,
        mean_classes=mean_classes,
    )


def extract_class_names(dataset_cfg) -> List[str]:
    """Best-effort extraction of class names from an mmcv Config dataset config."""
    candidates = []

    candidates.append(dataset_cfg.get("classes", None))
    metainfo = dataset_cfg.get("metainfo", None) or {}
    if isinstance(metainfo, dict):
        candidates.append(metainfo.get("classes", None))

    for path in [
        ("data", "train", "dataset", "classes"),
        ("data", "train", "dataset", "metainfo", "classes"),
        ("train_dataloader", "dataset", "classes"),
        ("train_dataloader", "dataset", "metainfo", "classes"),
        ("val_dataloader", "dataset", "classes"),
        ("val_dataloader", "dataset", "metainfo", "classes"),
    ]:
        node = dataset_cfg
        ok = True
        for key in path:
            if not hasattr(node, "get"):
                ok = False
                break
            node = node.get(key, None)
            if node is None:
                ok = False
                break
        if ok:
            candidates.append(node)

    for cand in candidates:
        if cand is None:
            continue
        if isinstance(cand, (list, tuple)) and cand and isinstance(cand[0], str):
            return list(cand)

    raise ValueError(
        "Could not infer `classes` from LPOSS dataset config. "
        "Please ensure it defines `classes = [...]` or `metainfo = dict(classes=[...])`."
    )


def extract_palette(dataset_cfg, num_classes: int) -> List[List[int]]:
    candidates = []
    candidates.append(dataset_cfg.get("palette", None))
    metainfo = dataset_cfg.get("metainfo", None) or {}
    if isinstance(metainfo, dict):
        candidates.append(metainfo.get("palette", None))

    for path in [
        ("data", "train", "dataset", "palette"),
        ("data", "train", "dataset", "metainfo", "palette"),
        ("train_dataloader", "dataset", "palette"),
        ("train_dataloader", "dataset", "metainfo", "palette"),
        ("val_dataloader", "dataset", "palette"),
        ("val_dataloader", "dataset", "metainfo", "palette"),
    ]:
        node = dataset_cfg
        ok = True
        for key in path:
            if not hasattr(node, "get"):
                ok = False
                break
            node = node.get(key, None)
            if node is None:
                ok = False
                break
        if ok:
            candidates.append(node)

    for cand in candidates:
        if cand is None:
            continue
        if isinstance(cand, (list, tuple)) and cand and isinstance(cand[0], (list, tuple)):
            return [list(map(int, row[:3])) for row in cand]

    rng = np.random.default_rng(17)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette.tolist()


def _colorize_mask(mask: np.ndarray, palette: Sequence[Sequence[int]]) -> np.ndarray:
    h, w = mask.shape
    palette_arr = np.asarray(palette, dtype=np.uint8)
    idx = np.clip(mask.astype(np.int64), 0, len(palette_arr) - 1)
    colored = palette_arr[idx.reshape(-1)].reshape(h, w, 3)
    return colored


def _blend_overlay(image: np.ndarray, mask_rgb: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    img = image.astype(np.float32)
    overlay = mask_rgb.astype(np.float32)
    blended = img * (1.0 - alpha) + overlay * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _load_mask(path: Path) -> np.ndarray:
    mask = io.imread(str(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32)


# ---------------------------
# LPOSS inference (no mmseg / no mmcv.ops)
# ---------------------------

def _ensure_repo_imports() -> None:
    """Make sure repo root is in sys.path so `from models import build_model` works."""
    repo_root = Path(__file__).resolve().parent.parent  # tools/ -> repo root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_hydra_cfg(config_path: Path):
    """Load Hydra config from a YAML file path.

    This matches how your finetune.py typically uses Hydra compose.
    """
    from hydra import compose, initialize_config_dir

    config_dir = str(config_path.parent)
    config_name = config_path.stem
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def _get_nested(obj, keys: Sequence[str], default=None):
    cur = obj
    for k in keys:
        if cur is None:
            return default
        if hasattr(cur, "get"):
            cur = cur.get(k, None)
        elif isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


def build_lposs_model(lposs: LpossConfig):
    """Load LPOSS model for inference without importing mmseg losses/ops.

    Returns: (seg_model, class_names, norm_params)
      norm_params = (mean, std, bgr_to_rgb)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from mmcv import Config

    _ensure_repo_imports()
    from models import build_model  # noqa: E402

    # 1) Load Hydra cfg from YAML (same style as finetune)
    cfg = _load_hydra_cfg(lposs.config_path)

    # 2) Dataset config -> class names
    dataset_cfg = Config.fromfile(str(lposs.dataset_config))
    class_names = extract_class_names(dataset_cfg)
    palette = extract_palette(dataset_cfg, num_classes=len(class_names))

    # 3) Build base model
    base_model = build_model(cfg.model, class_names=class_names)

    class IdentityHead(nn.Module):
        def forward(self, feats: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
            return feats

    class FineTuneWrapper(nn.Module):
        """Inference-time wrapper consistent with finetune training logic.

        - tries backbone(..., return_feat=True) to get (logits, feats)
        - runs decode_head.cls_seg(feats)
        - returns (logits + mixed_logits)/2
        """
        def __init__(self, base_model_: nn.Module) -> None:
            super().__init__()
            self.base_model = base_model_
            self.mixer = IdentityHead()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            backbone = getattr(self.base_model, "clip_backbone", None) or self.base_model

            logits = None
            feats = None

            # Try common signatures
            try:
                out = backbone(x, return_feat=True)
                if isinstance(out, tuple) and len(out) >= 2:
                    logits, feats = out[0], out[1]
                else:
                    logits, feats = out, None
            except TypeError:
                out = backbone(x)
                if isinstance(out, tuple):
                    logits = out[0]
                    feats = out[1] if len(out) > 1 else None
                else:
                    logits, feats = out, None

            if feats is None:
                feats = logits

            feats = self.mixer(feats, logits)

            decode_head = getattr(self.base_model, "decode_head", None)
            if decode_head is None and hasattr(self.base_model, "clip_backbone"):
                decode_head = getattr(self.base_model.clip_backbone, "decode_head", None)
            if decode_head is None:
                raise AttributeError("No decode_head found in base_model")

            mixed_logits = decode_head.cls_seg(feats)

            # Ensure shapes match before averaging
            if mixed_logits.shape[-2:] != logits.shape[-2:]:
                mixed_logits = F.interpolate(
                    mixed_logits, size=logits.shape[-2:], mode="bilinear", align_corners=False
                )

            return (logits + mixed_logits) / 2.0

    seg_model = FineTuneWrapper(base_model)

    # 4) Load checkpoint
    ckpt = torch.load(lposs.checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state" in ckpt:
        state = ckpt["state"]
    else:
        state = ckpt

    # Remove common prefixes if present
    if isinstance(state, dict):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        # if any(k.startswith("base_model.") for k in state.keys()):
        #     state = {k.replace("base_model.", "", 1): v for k, v in state.items()}

    missing, unexpected = seg_model.load_state_dict(state, strict=False)
    if missing or unexpected:
        LOGGER.warning("LPOSS checkpoint load missing=%s unexpected=%s", missing, unexpected)

    device = torch.device(lposs.device)
    seg_model.to(device)
    seg_model.eval()

    # 5) Norm params (defaults are mmseg-ish)
    dp = _get_nested(cfg, ["model", "data_preprocessor"], default=None) or _get_nested(cfg, ["data_preprocessor"], default=None)
    mean = _get_nested(dp, ["mean"], default=[123.675, 116.28, 103.53])
    std = _get_nested(dp, ["std"], default=[58.395, 57.12, 57.375])
    bgr_to_rgb = bool(_get_nested(dp, ["bgr_to_rgb"], default=True))
    mean = [float(x) for x in list(mean)]
    std = [float(x) for x in list(std)]

    return seg_model, class_names, palette, (mean, std, bgr_to_rgb)


def lposs_predict_map(image: np.ndarray, seg_model, norm_params) -> np.ndarray:
    """Run model forward and return per-pixel class probabilities [H,W,C] on original resolution."""
    import torch
    import torch.nn.functional as F

    mean, std, bgr_to_rgb = norm_params

    # skimage.io.imread -> RGB usually. If your pipeline is BGR, set bgr_to_rgb=False in config.
    img = image
    if bgr_to_rgb is False:
        # user explicitly says BGR is expected -> convert RGB->BGR
        img = img[..., ::-1].copy()

    H, W = img.shape[:2]
    x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)  # 1CHW

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t

    # Pad to multiple of 14 (ViT patch size), safer for DINO backbones
    pad_h = (14 - (H % 14)) % 14
    pad_w = (14 - (W % 14)) % 14
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    device = next(seg_model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        logits = seg_model(x)  # 1,C,h,w (maybe padded)
        # unpad logits to original spatial size if it matches padded size
        logits = logits[:, :, :H, :W]
        # if logits are not at full resolution, upsample
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)

    probs = probs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return probs


def compute_lposs_stats(
    probs: np.ndarray,
    mask: np.ndarray,
    class_names: Sequence[str],
    damage_classes: Sequence[str],
    mean_classes: Sequence[str],
) -> Dict[str, float]:
    idxs = np.where(mask)
    if idxs[0].size == 0:
        out = {
            "p_damage": float("nan"),
            "p_top1": float("nan"),
            "cls_top1": float("nan"),
            "cls_top1_name": "",
            "entropy_mean": float("nan"),
            "entropy_norm_mean": float("nan"),
            "margin_mean": float("nan"),
        }
        for c in mean_classes:
            out[f"mean_p_{c}"] = float("nan")
        return out

    probs_masked = probs[idxs[0], idxs[1], :]
    top1 = np.max(probs_masked, axis=1)
    top1_idx = np.argmax(probs_masked, axis=1)
    p_top1 = float(np.mean(top1))
    cls_top1_idx = int(np.bincount(top1_idx, minlength=len(class_names)).argmax())
    cls_top1 = float(cls_top1_idx)
    cls_top1_name = class_names[cls_top1_idx] if 0 <= cls_top1_idx < len(class_names) else ""

    entropy = -np.sum(probs_masked * np.log(np.clip(probs_masked, 1e-8, None)), axis=1)
    entropy_mean = float(np.mean(entropy))
    entropy_norm_mean = float(entropy_mean / np.log(max(len(class_names), 2)))

    sorted_probs = np.sort(probs_masked, axis=1)
    margin_mean = float(np.mean(sorted_probs[:, -1] - sorted_probs[:, -2])) if sorted_probs.shape[1] > 1 else 0.0

    name_to_idx = {name: i for i, name in enumerate(class_names)}
    dmg_idxs = [name_to_idx[c] for c in damage_classes if c in name_to_idx]
    p_damage = float(np.mean(np.sum(probs_masked[:, dmg_idxs], axis=1))) if dmg_idxs else 0.0

    mean_class_probs: Dict[str, float] = {}
    for c in mean_classes:
        if c in name_to_idx:
            mean_class_probs[f"mean_p_{c}"] = float(np.mean(probs_masked[:, name_to_idx[c]]))
        else:
            mean_class_probs[f"mean_p_{c}"] = float("nan")

    return {
        "p_damage": p_damage,
        "p_top1": p_top1,
        "cls_top1": cls_top1,
        "cls_top1_name": cls_top1_name,
        "entropy_mean": entropy_mean,
        "entropy_norm_mean": entropy_norm_mean,
        "margin_mean": margin_mean,
        **mean_class_probs,
    }


def compute_year_features(
    manifest: pd.DataFrame,
    spx_cache: Path,
    out_dir: Path,
    codec: str,
    mode: str,
    min_area: int,
    limit: Optional[int],
    lposs_cfg: Optional[LpossConfig],
    export_masks_dir: Optional[Path],
    gt_mask_col: str,
    overlay_alpha: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base_cache: Dict[Tuple[int, int, str], int] = {}

    lposs_inferencer = None
    lposs_classes: Optional[List[str]] = None
    lposs_palette: Optional[List[List[int]]] = None
    lposs_norm = None
    if lposs_cfg is not None:
        lposs_inferencer, lposs_classes, lposs_palette, lposs_norm = build_lposs_model(lposs_cfg)

    if export_masks_dir is not None and lposs_cfg is None:
        raise ValueError("--export-masks-dir requires --lposs-config and --lposs-checkpoint")

    manifest_sorted = manifest.sort_values(["facade_id", "year"]).reset_index(drop=True)

    for i, r in manifest_sorted.iterrows():
        if limit is not None and i >= limit:
            break
        facade_id = str(r["facade_id"])
        year = int(r["year"])
        image_path = str(r["image_path"])

        image = read_image(image_path)
        labels = load_labels(spx_cache, facade_id, year)
        objs = load_objects(spx_cache, facade_id, year)

        probs = None
        if lposs_cfg is not None and lposs_inferencer is not None and lposs_classes is not None and lposs_norm is not None:
            probs = lposs_predict_map(image, lposs_inferencer, lposs_norm)

        if export_masks_dir is not None and probs is not None and lposs_classes is not None and lposs_palette is not None:
            stem = Path(image_path).stem
            file_tag = f"{facade_id}_{year}_{stem}"
            pred_labels = np.argmax(probs, axis=-1).astype(np.uint8)

            img_path = export_masks_dir / "images" / f"{file_tag}.png"
            pred_mask_path = export_masks_dir / "pred_masks" / f"{file_tag}.png"
            pred_overlay_path = export_masks_dir / "overlay_pred" / f"{file_tag}.png"
            _save_image(img_path, image)
            _save_image(pred_mask_path, pred_labels)

            pred_rgb = _colorize_mask(pred_labels, lposs_palette)
            pred_overlay = _blend_overlay(image, pred_rgb, overlay_alpha)
            _save_image(pred_overlay_path, pred_overlay)

            if gt_mask_col in r and pd.notna(r[gt_mask_col]):
                gt_path = Path(str(r[gt_mask_col]))
                if gt_path.exists():
                    gt_mask = _load_mask(gt_path)
                    gt_rgb = _colorize_mask(gt_mask, lposs_palette)
                    gt_overlay = _blend_overlay(image, gt_rgb, overlay_alpha)
                    gt_overlay_path = export_masks_dir / "overlay_gt" / f"{file_tag}.png"
                    _save_image(gt_overlay_path, gt_overlay)
                else:
                    LOGGER.warning("GT mask not found: %s", gt_path)
            else:
                LOGGER.warning(
                    "GT mask column '%s' missing or empty for %s %s; skipping GT overlay.",
                    gt_mask_col,
                    facade_id,
                    year,
                )

        for _, obj in objs.iterrows():
            obj_id = int(obj["obj_id"])
            area_px = float(obj["area_px"])
            if area_px < float(min_area):
                continue

            x1 = int(obj["bbox_x1"])
            y1 = int(obj["bbox_y1"])
            x2 = int(obj["bbox_x2"])
            y2 = int(obj["bbox_y2"])

            label_id = int(obj.get("label_id", obj_id))
            mask = labels == label_id

            stats = compute_bpp_excess(
                image=image,
                mask=mask,
                bbox=(x1, y1, x2, y2),
                area_px=area_px,
                codec=codec,
                mode=mode,
                base_cache=base_cache,
            )

            row_out: Dict[str, object] = {
                "facade_id": facade_id,
                "year": year,
                "obj_id": obj_id,
                "area_px": float(area_px),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "cx": float(obj.get("cx", float("nan"))),
                "cy": float(obj.get("cy", float("nan"))),
                **stats,
            }

            if probs is not None and lposs_classes is not None and lposs_cfg is not None:
                lposs_stats = compute_lposs_stats(
                    probs=probs,
                    mask=mask,
                    class_names=lposs_classes,
                    damage_classes=lposs_cfg.damage_classes,
                    mean_classes=lposs_cfg.mean_classes,
                )
                row_out.update(lposs_stats)

            rows.append(row_out)

        LOGGER.info("Processed per-year features for %s %s", facade_id, year)

    df = pd.DataFrame(rows)
    out_path = out_dir / "spx_features.parquet"
    df.to_parquet(out_path, index=False)
    LOGGER.info("Saved per-year features: %s", out_path)
    return df


def compute_pair_features(
    pairs_df: pd.DataFrame,
    manifest: pd.DataFrame,
    spx_cache: Path,
    geom_dir: Path,
    match_dir: Path,
    codec: str,
    mode: str,
    min_area: int,
    min_coverage: float,
    limit: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    rows: List[Dict[str, object]] = []
    base_cache: Dict[Tuple[int, int, str], int] = {}
    quality_stats: Dict[str, Dict[str, int]] = {}

    import cv2

    manifest_index = {
        (str(r.facade_id), int(r.year)): str(r.image_path) for r in manifest.itertuples(index=False)
    }

    def _emit_row(
        *,
        facade_id: str,
        year_a: int,
        year_b: int,
        obj_id_a: int,
        obj_id_b: int,
        status_quality: str,
        pair_mode: str,
        match_score: float,
        match_d_pos: float,
        coverage_match: float,
        mask_sup: np.ndarray,
        support_area: int,
        support_ratio: float,
        image_b: np.ndarray,
        image_a_warp: np.ndarray,
    ) -> None:
        ys, xs = np.where(mask_sup)
        if ys.size == 0 or xs.size == 0:
            return
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1

        stats_b = compute_bpp_excess(
            image=image_b,
            mask=mask_sup,
            bbox=(x1, y1, x2, y2),
            area_px=float(support_area),
            codec=codec,
            mode=mode,
            base_cache=base_cache,
        )
        stats_a = compute_bpp_excess(
            image=image_a_warp,
            mask=mask_sup,
            bbox=(x1, y1, x2, y2),
            area_px=float(support_area),
            codec=codec,
            mode=mode,
            base_cache=base_cache,
        )

        bpp_excess_b = float(stats_b["bpp_excess"])
        bpp_excess_a = float(stats_a["bpp_excess"])
        delta_bpp = bpp_excess_b - bpp_excess_a
        delta_bpp_rel = delta_bpp / (abs(bpp_excess_a) + 1e-6)

        rows.append(
            {
                "facade_id": facade_id,
                "year_a": int(year_a),
                "year_b": int(year_b),
                "obj_id_a": int(obj_id_a),
                "obj_id_b": int(obj_id_b),
                "status_quality": status_quality,
                "pair_mode": pair_mode,
                "match_score": float(match_score),
                "match_d_pos": float(match_d_pos),
                "coverage_match": float(coverage_match),
                "support_area_px": int(support_area),
                "support_area_ratio": float(support_ratio),
                "bpp_excess_b": bpp_excess_b,
                "bpp_excess_a_warp": bpp_excess_a,
                "delta_bpp": float(delta_bpp),
                "delta_bpp_rel": float(delta_bpp_rel),
            }
        )
        quality_stats[status_quality]["kept"] += 1

    for i, row in pairs_df.iterrows():
        if limit is not None and i >= limit:
            break

        facade_id = str(row["facade_id"])
        year_a = int(row["year_a"])
        year_b = int(row["year_b"])

        img_a_path = manifest_index.get((facade_id, year_a))
        img_b_path = manifest_index.get((facade_id, year_b))
        if img_a_path is None or img_b_path is None:
            LOGGER.warning("Missing image paths in manifest for %s %s_%s", facade_id, year_a, year_b)
            continue

        image_a = read_image(img_a_path)
        image_b = read_image(img_b_path)

        geom_quality, H = load_geom(geom_dir, facade_id, year_a, year_b)
        match_data = load_matches(match_dir, facade_id, year_a, year_b)

        if match_data is None:
            matches: Optional[List[Dict[str, object]]] = None
            status_quality = geom_quality
        else:
            matches = list(match_data.get("matches", []))
            status_quality = str(match_data.get("status_quality", geom_quality) or geom_quality)

        labels_a = load_labels(spx_cache, facade_id, year_a)
        labels_b = load_labels(spx_cache, facade_id, year_b)
        objs_a = load_objects(spx_cache, facade_id, year_a)
        objs_b = load_objects(spx_cache, facade_id, year_b)

        area_b_map = {int(r["obj_id"]): float(r["area_px"]) for _, r in objs_b.iterrows()}
        label_map_a = {int(r["obj_id"]): int(r.get("label_id", r["obj_id"])) for _, r in objs_a.iterrows()}
        label_map_b = {int(r["obj_id"]): int(r.get("label_id", r["obj_id"])) for _, r in objs_b.iterrows()}

        if H is not None:
            h_b, w_b = image_b.shape[:2]
            image_a_warp = cv2.warpPerspective(image_a, H, (w_b, h_b))
            ones = np.ones(image_a.shape[:2], dtype=np.uint8)
            valid_mask = cv2.warpPerspective(ones, H, (w_b, h_b), flags=cv2.INTER_NEAREST).astype(bool)
        else:
            image_a_warp = image_a
            if image_a.shape[:2] != image_b.shape[:2]:
                image_a_warp = cv2.resize(image_a, (image_b.shape[1], image_b.shape[0]))
            valid_mask = np.ones(image_b.shape[:2], dtype=bool)

        quality_stats.setdefault(status_quality, {"total": 0, "kept": 0})

        def _valid_support(mask_b: np.ndarray, obj_id_b: int) -> Tuple[Optional[np.ndarray], int, float]:
            mask_sup = mask_b & valid_mask
            support_area = int(mask_sup.sum())
            if support_area < int(min_area):
                return None, 0, float("nan")
            area_b = area_b_map.get(int(obj_id_b), float("nan"))
            support_ratio = float(support_area / area_b) if area_b and np.isfinite(area_b) else float("nan")
            if np.isfinite(support_ratio) and support_ratio < float(min_coverage):
                return None, 0, support_ratio
            return mask_sup, support_area, support_ratio

        if matches is not None and len(matches) > 0:
            for match in matches:
                quality_stats[status_quality]["total"] += 1
                obj_id_a = int(match.get("a"))
                obj_id_b = int(match.get("b"))

                label_b = label_map_b.get(obj_id_b, obj_id_b)
                mask_b = labels_b == label_b

                mask_sup, support_area, support_ratio = _valid_support(mask_b, obj_id_b)
                if mask_sup is None:
                    continue

                coverage_match = float("nan")
                if H is not None:
                    label_a = label_map_a.get(obj_id_a, obj_id_a)
                    mask_a = labels_a == label_a
                    mask_a_warp = cv2.warpPerspective(
                        mask_a.astype(np.uint8),
                        H,
                        (mask_b.shape[1], mask_b.shape[0]),
                        flags=cv2.INTER_NEAREST,
                    ).astype(bool)
                    denom = float(mask_b.sum()) if mask_b.sum() else 1.0
                    coverage_match = float((mask_b & mask_a_warp).sum() / denom)

                _emit_row(
                    facade_id=facade_id,
                    year_a=year_a,
                    year_b=year_b,
                    obj_id_a=obj_id_a,
                    obj_id_b=obj_id_b,
                    status_quality=status_quality,
                    pair_mode="matched",
                    match_score=float(match.get("score", float("nan"))),
                    match_d_pos=float(match.get("d_pos", float("nan"))),
                    coverage_match=coverage_match,
                    mask_sup=mask_sup,
                    support_area=support_area,
                    support_ratio=support_ratio,
                    image_b=image_b,
                    image_a_warp=image_a_warp,
                )
        else:
            for _, item_b in objs_b.iterrows():
                quality_stats[status_quality]["total"] += 1
                obj_id_b = int(item_b["obj_id"])
                label_b = label_map_b.get(obj_id_b, obj_id_b)
                mask_b = labels_b == label_b

                mask_sup, support_area, support_ratio = _valid_support(mask_b, obj_id_b)
                if mask_sup is None:
                    continue

                _emit_row(
                    facade_id=facade_id,
                    year_a=year_a,
                    year_b=year_b,
                    obj_id_a=-1,
                    obj_id_b=obj_id_b,
                    status_quality=status_quality,
                    pair_mode="fallback",
                    match_score=float("nan"),
                    match_d_pos=float("nan"),
                    coverage_match=float("nan"),
                    mask_sup=mask_sup,
                    support_area=support_area,
                    support_ratio=support_ratio,
                    image_b=image_b,
                    image_a_warp=image_a_warp,
                )

        LOGGER.info("Processed pair %s %s_%s", facade_id, year_a, year_b)

    quality_summary = {
        status: {
            "total": int(counts["total"]),
            "kept": int(counts["kept"]),
            "missing_frac": float(1.0 - counts["kept"] / counts["total"] if counts["total"] else 0.0),
        }
        for status, counts in quality_stats.items()
    }
    return pd.DataFrame(rows), quality_summary


def compute_qc_summary(
    spx_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    quality_summary: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    bpp = spx_df["bpp_excess"].to_numpy()
    bpp_clean = bpp[np.isfinite(bpp)]
    summary = {
        "bpp_excess_min": float(np.min(bpp_clean)) if bpp_clean.size else float("nan"),
        "bpp_excess_median": float(np.median(bpp_clean)) if bpp_clean.size else float("nan"),
        "bpp_excess_p95": float(np.percentile(bpp_clean, 95)) if bpp_clean.size else float("nan"),
    }

    area = spx_df["area_px"].to_numpy()
    mask = np.isfinite(bpp) & np.isfinite(area)
    if np.sum(mask) > 2:
        corr = np.corrcoef(bpp[mask], area[mask])[0, 1]
        summary["corr_bpp_excess_area_px"] = float(corr)
    else:
        summary["corr_bpp_excess_area_px"] = float("nan")

    summary["delta_missing_by_quality"] = quality_summary
    summary["pair_rows"] = int(len(pair_df))
    summary["spx_rows"] = int(len(spx_df))
    return summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.temporal_manifest)
    pairs_df = pd.read_csv(args.pairs)
    required_pairs = {"facade_id", "year_a", "year_b"}
    if not required_pairs.issubset(pairs_df.columns):
        missing = ", ".join(sorted(required_pairs - set(pairs_df.columns)))
        raise ValueError(f"Pairs CSV missing required columns: {missing}")

    lposs_cfg = resolve_lposs_config(args)

    spx_df = compute_year_features(
        manifest=manifest,
        spx_cache=args.spx_cache,
        out_dir=args.out_dir,
        codec=args.codec,
        mode=args.mode,
        min_area=args.min_area,
        limit=args.limit,
        lposs_cfg=lposs_cfg,
        export_masks_dir=args.export_masks_dir,
        gt_mask_col=args.gt_mask_col,
        overlay_alpha=args.overlay_alpha,
    )

    pair_features, quality_summary = compute_pair_features(
        pairs_df,
        manifest,
        args.spx_cache,
        args.geom_dir,
        args.match_dir,
        args.codec,
        args.mode,
        args.min_area,
        args.min_coverage,
        args.limit,
    )
    pair_out = args.out_dir / "pair_features.parquet"
    pair_features.to_parquet(pair_out, index=False)
    LOGGER.info("Saved per-pair features: %s", pair_out)

    qc = compute_qc_summary(spx_df, pair_features, quality_summary)
    qc_path = args.out_dir / "qc_summary.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    LOGGER.info("Saved QC summary: %s", qc_path)


if __name__ == "__main__":
    main()
