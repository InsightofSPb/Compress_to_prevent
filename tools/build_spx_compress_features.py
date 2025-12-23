import argparse
import json
import logging
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
    parser.add_argument("--limit", default=None, type=int)

    parser.add_argument("--lposs-config", type=Path, default=None)
    parser.add_argument("--lposs-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--lposs-dataset-config",
        type=Path,
        default=Path("segmentation/configs/_base_/datasets/facades_test.py"),
        help="Dataset config providing class names for LPOSS stats.",
    )
    parser.add_argument("--device", default="cuda:0", type=str)
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
    elif "mask_path" in df.columns:
        path_col = "mask_path"
    elif "full_path" in df.columns:
        path_col = "full_path"
    else:
        raise ValueError("Manifest must include image_path, mask_path, or full_path column")

    df = df.rename(columns={path_col: "image_path"})
    return df


def read_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = io.imread(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


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


def load_matches(match_dir: Path, facade_id: str, year_a: int, year_b: int) -> Dict[str, object]:
    match_path = match_dir / f"{facade_id}" / f"{year_a}_{year_b}_match.json"
    if not match_path.exists():
        raise FileNotFoundError(f"Match file not found: {match_path}")
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

    key = (crop.shape[0], crop.shape[1], codec)
    if key not in base_cache:
        base_crop = np.zeros_like(crop)
        base_cache[key] = encode_crop(base_crop, codec)
    bytes_base = base_cache[key]

    bytes_excess = max(bytes_real - bytes_base, 0)
    area_px = max(float(area_px), 1.0)
    bbox_area = max((x2 - x1) * (y2 - y1), 1)
    denom_area = area_px if mode == "mask" else float(bbox_area)
    denom_area = max(float(denom_area), 1.0)
    bpp_excess = 8.0 * bytes_excess / denom_area
    bpp_bbox = 8.0 * bytes_real / float(bbox_area)

    return {
        "bytes_real": float(bytes_real),
        "bytes_base": float(bytes_base),
        "bytes_excess": float(bytes_excess),
        "bpp_excess": float(bpp_excess),
        "bpp_excess_denom": float(denom_area),
        "bpp_bbox": float(bpp_bbox),
    }


def resolve_lposs_config(args: argparse.Namespace) -> Optional[LpossConfig]:
    if args.lposs_config is None or args.lposs_checkpoint is None:
        return None
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
    """Best-effort extraction of class names from an mmcv/mmengine dataset config."""
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

def build_lposs_model(lposs: LpossConfig):
    """Load LPOSS segmentation model and build a reusable inferencer."""
    import torch
    from mmcv import Config
    from models import build_model
    from segmentation.evaluation.lposs_eval import LPOSS_Infrencer

    cfg = Config.fromfile(str(lposs.config_path))
    dataset_cfg = Config.fromfile(str(lposs.dataset_config))
    class_names = extract_class_names(dataset_cfg)

    model = build_model(cfg.model, class_names=class_names)

    checkpoint = torch.load(lposs.checkpoint_path, map_location="cpu")
    state = checkpoint.get("state") if isinstance(checkpoint, dict) and "state" in checkpoint else checkpoint
    if isinstance(state, dict) and "base_model." in next(iter(state.keys()), ""):
        state = {k.replace("base_model.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        LOGGER.warning("LPOSS checkpoint load missing=%s unexpected=%s", missing, unexpected)

    device = torch.device(lposs.device)
    model.to(device)
    model.eval()
    seg_model = LPOSS_Infrencer(model, cfg, num_classes=len(class_names), test_cfg={"mode": "whole"})
    seg_model.to(device)
    seg_model.eval()
    return seg_model, class_names


def lposs_predict_map(
    image: np.ndarray,
    seg_model,
) -> np.ndarray:
    """Run LPOSS inferencer once and return per-pixel class probabilities [H,W,C]."""
    import torch

    h, w = image.shape[:2]
    input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    if input_tensor.dtype != torch.uint8:
        input_tensor = input_tensor.to(torch.uint8)

    device = next(seg_model.parameters()).device
    input_tensor = input_tensor.to(device)

    metas = [
        {
            "ori_shape": (h, w, 3),
            "img_shape": (h, w, 3),
            "pad_shape": (h, w, 3),
            "flip": False,
            "flip_direction": None,
        }
    ]

    with torch.no_grad():
        logits = seg_model.whole_inference(input_tensor, metas, rescale=True)
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
    mask_idx = np.where(mask)
    if mask_idx[0].size == 0:
        return {
            "p_damage": float("nan"),
            "p_top1": float("nan"),
            "cls_top1": float("nan"),
            "entropy_mean": float("nan"),
            "margin_mean": float("nan"),
            **{f"mean_p_{name}": float("nan") for name in mean_classes},
        }

    probs_masked = probs[mask_idx]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    damage_indices = [class_to_idx[name] for name in damage_classes if name in class_to_idx]
    if damage_indices:
        p_damage = float(np.mean(np.sum(probs_masked[:, damage_indices], axis=1)))
    else:
        p_damage = float("nan")

    top1_idx = np.argmax(probs_masked, axis=1)
    p_top1 = float(np.mean(np.take_along_axis(probs_masked, top1_idx[:, None], axis=1)))
    cls_top1_idx = int(np.bincount(top1_idx, minlength=len(class_names)).argmax())
    cls_top1 = float(cls_top1_idx)
    cls_top1_name = class_names[cls_top1_idx] if 0 <= cls_top1_idx < len(class_names) else ""

    entropy = -np.sum(probs_masked * np.log(np.clip(probs_masked, 1e-8, None)), axis=1)
    entropy_mean = float(np.mean(entropy))
    entropy_norm_mean = float(entropy_mean / np.log(max(len(class_names), 2)))

    sorted_probs = np.sort(probs_masked, axis=1)
    margin_mean = float(np.mean(sorted_probs[:, -1] - sorted_probs[:, -2])) if sorted_probs.shape[1] > 1 else 0.0

    mean_class_probs = {}
    for name in mean_classes:
        idx = class_to_idx.get(name)
        mean_class_probs[f"mean_p_{name}"] = float(np.mean(probs_masked[:, idx])) if idx is not None else float("nan")

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
    codec: str,
    mode: str,
    lposs_cfg: Optional[LpossConfig],
    limit: Optional[int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base_cache: Dict[Tuple[int, int, str], int] = {}
    lposs_inferencer = None
    lposs_classes: Optional[List[str]] = None

    if lposs_cfg is not None:
        lposs_inferencer, lposs_classes = build_lposs_model(lposs_cfg)

    for idx, row in manifest.iterrows():
        if limit is not None and idx >= limit:
            break
        facade_id = row["facade_id"]
        year = int(row["year"])
        image_path = Path(row["image_path"])

        image = read_image(image_path)
        labels = load_labels(spx_cache, facade_id, year)
        objs = load_objects(spx_cache, facade_id, year)

        if labels.shape[:2] != image.shape[:2]:
            raise ValueError(f"Label/image size mismatch for {facade_id} {year}")

        probs = None
        if lposs_cfg is not None and lposs_inferencer is not None and lposs_classes is not None:
            probs = lposs_predict_map(image, lposs_inferencer)

        for _, obj in objs.iterrows():
            obj_id = int(obj["obj_id"])
            label_value = int(obj.get("label_id", obj_id))
            x1, y1, x2, y2 = (int(obj["bbox_x1"]), int(obj["bbox_y1"]), int(obj["bbox_x2"]), int(obj["bbox_y2"]))
            mask = labels == label_value
            stats = compute_bpp_excess(
                image,
                mask,
                (x1, y1, x2, y2),
                float(obj["area_px"]),
                codec,
                mode,
                base_cache,
            )

            row_data: Dict[str, object] = {
                "facade_id": facade_id,
                "year": year,
                "obj_id": obj_id,
                "area_px": float(obj["area_px"]),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "cx": float(obj["cx"]),
                "cy": float(obj["cy"]),
                "codec": codec,
                "mode": mode,
                **stats,
            }

            if probs is not None and lposs_classes is not None:
                lposs_stats = compute_lposs_stats(
                    probs,
                    mask,
                    lposs_classes,
                    lposs_cfg.damage_classes,
                    lposs_cfg.mean_classes,
                )
                row_data.update(lposs_stats)

            rows.append(row_data)

        LOGGER.info("Processed per-year features for %s %s", facade_id, year)

    return pd.DataFrame(rows)


def compute_pair_features(
    pairs_df: pd.DataFrame,
    manifest: pd.DataFrame,
    spx_cache: Path,
    geom_dir: Path,
    match_dir: Path,
    codec: str,
    mode: str,
    min_area: int,
    limit: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    rows: List[Dict[str, object]] = []
    base_cache: Dict[Tuple[int, int, str], int] = {}
    quality_stats: Dict[str, Dict[str, int]] = {}

    import cv2

    manifest_index = {
        (str(item["facade_id"]), int(item["year"])): Path(item["image_path"])
        for _, item in manifest.iterrows()
    }

    for idx, row in pairs_df.iterrows():
        if limit is not None and idx >= limit:
            break
        facade_id = row["facade_id"]
        year_a = int(row["year_a"])
        year_b = int(row["year_b"])

        geom_quality, H = load_geom(geom_dir, facade_id, year_a, year_b)
        match_data = load_matches(match_dir, facade_id, year_a, year_b)
        matches = match_data.get("matches", [])
        status_quality = match_data.get("status_quality", geom_quality) or geom_quality

        labels_a = load_labels(spx_cache, facade_id, year_a)
        labels_b = load_labels(spx_cache, facade_id, year_b)
        objs_a = load_objects(spx_cache, facade_id, year_a)
        objs_b = load_objects(spx_cache, facade_id, year_b)
        area_b_map = {int(item["obj_id"]): float(item["area_px"]) for _, item in objs_b.iterrows()}
        label_map_a = {int(item["obj_id"]): int(item.get("label_id", item["obj_id"])) for _, item in objs_a.iterrows()}
        label_map_b = {int(item["obj_id"]): int(item.get("label_id", item["obj_id"])) for _, item in objs_b.iterrows()}

        image_a_path = manifest_index.get((str(facade_id), year_a))
        image_b_path = manifest_index.get((str(facade_id), year_b))
        if image_a_path is None or image_b_path is None:
            raise ValueError(f"Missing image paths in manifest for {facade_id} {year_a}/{year_b}")
        image_a = read_image(image_a_path)
        image_b = read_image(image_b_path)

        if labels_a.shape[:2] != image_a.shape[:2] or labels_b.shape[:2] != image_b.shape[:2]:
            raise ValueError(f"Label/image size mismatch for {facade_id} {year_a}/{year_b}")

        if H is not None:
            h_b, w_b = image_b.shape[:2]
            image_a_warp = cv2.warpPerspective(image_a, H, (w_b, h_b))
        else:
            image_a_warp = image_a
            if image_a.shape[:2] != image_b.shape[:2]:
                image_a_warp = cv2.resize(image_a, (image_b.shape[1], image_b.shape[0]))

        quality_stats.setdefault(status_quality, {"total": 0, "kept": 0})

        for match in matches:
            quality_stats[status_quality]["total"] += 1
            obj_id_a = int(match["a"])
            obj_id_b = int(match["b"])
            label_a = label_map_a.get(obj_id_a, obj_id_a)
            label_b = label_map_b.get(obj_id_b, obj_id_b)
            mask_b = labels_b == label_b
            mask_a = labels_a == label_a

            if H is not None:
                mask_a_warp = cv2.warpPerspective(
                    mask_a.astype(np.uint8),
                    H,
                    (mask_b.shape[1], mask_b.shape[0]),
                    flags=cv2.INTER_NEAREST,
                ).astype(bool)
                mask_sup = mask_b & mask_a_warp
            else:
                mask_sup = mask_b

            support_area = int(mask_sup.sum())
            if support_area < min_area:
                continue

            ys, xs = np.where(mask_sup)
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1

            area_b = area_b_map.get(obj_id_b, float("nan"))
            support_ratio = float(support_area / area_b) if area_b and not np.isnan(area_b) else float("nan")

            stats_b = compute_bpp_excess(
                image_b,
                mask_sup,
                (x1, y1, x2, y2),
                support_area,
                codec,
                mode,
                base_cache,
            )
            stats_a = compute_bpp_excess(
                image_a_warp,
                mask_sup,
                (x1, y1, x2, y2),
                support_area,
                codec,
                mode,
                base_cache,
            )

            bpp_excess_b = stats_b["bpp_excess"]
            bpp_excess_a = stats_a["bpp_excess"]
            delta_bpp = bpp_excess_b - bpp_excess_a
            delta_bpp_rel = delta_bpp / (abs(bpp_excess_a) + 1e-6)

            rows.append(
                {
                    "facade_id": facade_id,
                    "year_a": year_a,
                    "year_b": year_b,
                    "obj_id_a": obj_id_a,
                    "obj_id_b": obj_id_b,
                    "status_quality": status_quality,
                    "match_score": float(match.get("score", float("nan"))),
                    "match_d_pos": float(match.get("d_pos", float("nan"))),
                    "support_area_px": support_area,
                    "support_area_ratio": support_ratio,
                    "bpp_excess_b": bpp_excess_b,
                    "bpp_excess_a_warp": bpp_excess_a,
                    "delta_bpp": delta_bpp,
                    "delta_bpp_rel": delta_bpp_rel,
                }
            )
            quality_stats[status_quality]["kept"] += 1

        LOGGER.info("Processed pair %s %s_%s", facade_id, year_a, year_b)

    quality_summary = {
        status: {
            "total": counts["total"],
            "kept": counts["kept"],
            "missing_frac": float(
                1.0 - counts["kept"] / counts["total"] if counts["total"] else 0.0
            ),
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
        summary["corr_bpp_excess_area_px"] = float(np.corrcoef(bpp[mask], area[mask])[0, 1])
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

    spx_features = compute_year_features(
        manifest,
        args.spx_cache,
        args.codec,
        args.mode,
        lposs_cfg,
        args.limit,
    )
    spx_out = args.out_dir / "spx_features.parquet"
    spx_features.to_parquet(spx_out, index=False)
    LOGGER.info("Saved per-year features: %s", spx_out)

    pair_features, quality_summary = compute_pair_features(
        pairs_df,
        manifest,
        args.spx_cache,
        args.geom_dir,
        args.match_dir,
        args.codec,
        args.mode,
        args.min_area,
        args.limit,
    )
    pair_out = args.out_dir / "pair_features.parquet"
    pair_features.to_parquet(pair_out, index=False)
    LOGGER.info("Saved per-pair features: %s", pair_out)

    qc_summary = compute_qc_summary(spx_features, pair_features, quality_summary)
    qc_path = args.out_dir / "qc_summary.json"
    with qc_path.open("w", encoding="utf-8") as handle:
        json.dump(qc_summary, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Saved QC summary: %s", qc_path)


if __name__ == "__main__":
    main()
