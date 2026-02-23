import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm
import cv2
import numpy as np
import yaml

from dataset_ops import apply_zoom, build_transforms, generate_tiles, apply_cutmix, apply_mixup

YEAR_SUFFIX_RE = re.compile(r"_(20\d{2})$")
PXL_RE = re.compile(r"^PXL_(20\d{2})\d{4}_")
PHOTO_RE = re.compile(r"^photo_.*?(20\d{2})-\d{2}-\d{2}_")
HASH_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{6,}-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protocol-based dataset split + tiling + augmentation for facade segmentation"
    )
    parser.add_argument("--data-root", type=Path, help="Dataset root with images/ and masks/")
    parser.add_argument("--coco-json", type=Path, default=None, help="Optional COCO JSON index (not implemented yet)")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root")
    parser.add_argument("--protocol", type=str, default="A,B", help="A, B, or A,B")
    parser.add_argument("--test-years", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment-config", type=Path, default=Path("configs/augmentation.yaml"))
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--tile-only", action="store_true", help="Disable augmentations and run tile-only")
    parser.set_defaults(augment=True)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--pad-mode", type=str, choices=["constant", "reflect"], default=None)
    parser.add_argument("--min-content-ratio", type=float, default=None)
    parser.add_argument("--image-exts", nargs="+", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    return parser.parse_args()


def extract_year(file_name: str) -> Optional[int]:
    stem = Path(file_name).stem

    suffix_match = YEAR_SUFFIX_RE.search(stem)
    if suffix_match:
        return int(suffix_match.group(1))

    pxl_match = PXL_RE.match(stem)
    if pxl_match:
        return int(pxl_match.group(1))

    photo_match = PHOTO_RE.match(stem)
    if photo_match:
        return int(photo_match.group(1))

    if stem.startswith("IMG_"):
        return 2025

    return None


def strip_hash_prefix(name: str) -> str:
    return HASH_PREFIX_RE.sub("", name)


def extract_facade_id(file_name: str, year: int) -> str:
    stem = Path(file_name).stem
    stem = strip_hash_prefix(stem)
    suffix = f"_{year}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    facade_id = stem.strip("_-")
    return facade_id


def extract_source_id(file_name: str) -> str:
    stem = Path(file_name).stem
    return strip_hash_prefix(stem)



def load_yaml_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tiling_defaults(config_path: Path) -> Dict[str, object]:
    cfg = load_yaml_config(config_path)
    tiling = cfg.get("tiling", {})
    return {
        "tile_size": int(tiling.get("height", 448)),
        "stride": int(tiling.get("stride_h", 224)),
        "pad_mode": str(tiling.get("pad_mode", "constant")),
        "min_content_ratio": float(tiling.get("min_content_ratio", 0.6)),
        "augmentations_per_image": int(cfg.get("augmentations_per_image", 1)),
        "augmentations": cfg.get("augmentations", {}),
        "seed": int(cfg.get("seed", 42)),
        # чтобы overlay_cfg в run_protocol брался из конфига, а не только дефолты
        "overlay_alpha": float(cfg.get("overlay_alpha", 0.45)),
        "palette": cfg.get("palette", []),
    }

def is_forced_test_row(row: Dict[str, object], test_years_set: set[int]) -> bool:
    """
    Правило "обязательного теста" для временного сплита:
    - обычные исторические фото с year in test_years -> test
    - НО PXL_/IMG_ (phone captures / отдельные разметки без истории) НЕ форсим в test
      и разрешаем им попасть в train/val.
    """
    year = int(row["year"])
    if year not in test_years_set:
        return False

    img_path_obj = row.get("image_path")
    if isinstance(img_path_obj, Path):
        stem = img_path_obj.stem
    else:
        # fallback на rel_image_path / string
        rel_name = str(row.get("rel_image_path", ""))
        stem = Path(rel_name).stem

    return not (stem.startswith("PXL_") or stem.startswith("IMG_"))

def collect_samples_from_root(data_root: Path, image_exts: Sequence[str]) -> List[Dict[str, object]]:
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError("Expected 'images/' and 'masks/' directories under --data-root")

    ext_set = {ext.lower() for ext in image_exts}
    samples: List[Dict[str, object]] = []

    # ---------- Build mask indexes once (fast + robust matching) ----------
    mask_files = [p for p in masks_dir.rglob("*") if p.is_file()]

    # exact relative path under masks/
    mask_by_rel_exact: Dict[str, Path] = {}
    # relative path without extension -> list[Path] (to support .jpg image / .png mask)
    mask_by_rel_stem: Dict[str, List[Path]] = {}
    # exact basename in masks root (or anywhere)
    mask_by_name_exact: Dict[str, List[Path]] = {}
    # basename stem (lower) -> list[Path]
    mask_by_name_stem: Dict[str, List[Path]] = {}

    # which mask extensions are actually present
    present_mask_exts = set()

    for mp in mask_files:
        rel = mp.relative_to(masks_dir)
        rel_key = rel.as_posix().lower()
        rel_stem_key = rel.with_suffix("").as_posix().lower()
        name_key = mp.name.lower()
        stem_key = mp.stem.lower()

        mask_by_rel_exact[rel_key] = mp
        mask_by_rel_stem.setdefault(rel_stem_key, []).append(mp)
        mask_by_name_exact.setdefault(name_key, []).append(mp)
        mask_by_name_stem.setdefault(stem_key, []).append(mp)

        present_mask_exts.add(mp.suffix.lower())

    # Prefer common mask extensions first
    preferred_mask_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    ordered_mask_exts = [e for e in preferred_mask_exts if e in present_mask_exts] + [
        e for e in sorted(present_mask_exts) if e not in preferred_mask_exts
    ]

    def _pick_unique(cands: List[Path]) -> Optional[Path]:
        """Return unique candidate if unambiguous, else None."""
        uniq = []
        seen = set()
        for p in cands:
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen:
                uniq.append(p)
                seen.add(key)
        if len(uniq) == 1:
            return uniq[0]
        return None

    def _stem_variants(stem: str) -> List[str]:
        s = stem.lower()
        # Include hash-stripped version too (helper exists in your script)
        s2 = strip_hash_prefix(s)
        base = [s]
        if s2 and s2 != s:
            base.append(s2)

        variants = []
        suffixes = ["", "_mask", "-mask", "_seg", "-seg"]
        for b in base:
            for suf in suffixes:
                variants.append(f"{b}{suf}")

        # deduplicate preserving order
        out = []
        seen = set()
        for v in variants:
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out

    def resolve_mask_for_image(image_path: Path) -> Tuple[Optional[Path], str]:
        """Try multiple strategies to find a corresponding mask."""
        rel_from_images = image_path.relative_to(images_dir)
        rel_key = rel_from_images.as_posix().lower()
        rel_stem_key = rel_from_images.with_suffix("").as_posix().lower()

        # 1) Exact relative path match (same subdirs + same filename/ext)
        exact = mask_by_rel_exact.get(rel_key)
        if exact is not None:
            return exact, "exact_rel"

        # 2) Same relative stem, different extension (e.g., images/foo.jpg -> masks/foo.png)
        rel_stem_cands = mask_by_rel_stem.get(rel_stem_key, [])
        if rel_stem_cands:
            # choose by preferred extension if possible
            for ext in ordered_mask_exts:
                for c in rel_stem_cands:
                    if c.suffix.lower() == ext:
                        return c, "rel_stem_alt_ext"
            # fallback if only one
            one = _pick_unique(rel_stem_cands)
            if one is not None:
                return one, "rel_stem_unique"

        # 3) Exact basename anywhere in masks tree
        name_exact_cands = mask_by_name_exact.get(image_path.name.lower(), [])
        one = _pick_unique(name_exact_cands)
        if one is not None:
            return one, "exact_name_anywhere"

        # 4) Same basename stem anywhere in masks tree (plus common suffix variants)
        for stem_variant in _stem_variants(image_path.stem):
            cands = mask_by_name_stem.get(stem_variant, [])
            if not cands:
                continue

            # Prefer exact relative parent if there are multiple files with same stem
            img_rel_parent = rel_from_images.parent.as_posix().lower()
            same_parent = []
            for c in cands:
                c_rel_parent = c.relative_to(masks_dir).parent.as_posix().lower()
                if c_rel_parent == img_rel_parent:
                    same_parent.append(c)

            if same_parent:
                for ext in ordered_mask_exts:
                    for c in same_parent:
                        if c.suffix.lower() == ext:
                            return c, "same_stem_same_parent"
                one = _pick_unique(same_parent)
                if one is not None:
                    return one, "same_stem_same_parent_unique"

            # Otherwise pick by preferred extension if unique enough
            for ext in ordered_mask_exts:
                ext_cands = [c for c in cands if c.suffix.lower() == ext]
                one = _pick_unique(ext_cands)
                if one is not None:
                    return one, "same_stem_alt_ext_anywhere"

            one = _pick_unique(cands)
            if one is not None:
                return one, "same_stem_anywhere_unique"

        return None, "not_found"

    skipped_missing_mask = 0
    skipped_ambiguous = 0
    match_stats: Dict[str, int] = {}

    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in ext_set:
            continue

        mask_path, match_mode = resolve_mask_for_image(image_path)
        if mask_path is None:
            print(f"[WARN] Mask not found for image, skipping: {image_path}")
            skipped_missing_mask += 1
            continue

        # Safety: if matching mode could theoretically be ambiguous and resolve returned None, we already skipped.
        match_stats[match_mode] = match_stats.get(match_mode, 0) + 1

        year = extract_year(image_path.name)
        if year is None:
            raise ValueError(
                f"Could not extract year from filename '{image_path.name}'. "
                "Expected suffix _20xx, PXL_YYYYMMDD_..., photo_*_YYYY-MM-DD_..., or IMG_ prefix."
            )

        facade_id = extract_facade_id(image_path.name, year)
        source_id = extract_source_id(image_path.name)
        if not source_id:
            raise ValueError(f"Empty source_id extracted from filename '{image_path.name}'")

        samples.append(
            {
                "image_path": image_path,
                "mask_path": mask_path,
                "rel_image_path": str(image_path.relative_to(data_root)),
                "rel_mask_path": str(mask_path.relative_to(data_root)),
                "year": year,
                "facade_id": facade_id,
                "source_id": source_id,
            }
        )

    if skipped_missing_mask > 0:
        print(f"[INFO] Skipped {skipped_missing_mask} image(s) because mask was not found.")
    if skipped_ambiguous > 0:
        print(f"[INFO] Skipped {skipped_ambiguous} image(s) because mask match was ambiguous.")
    if match_stats:
        print("[INFO] Mask match modes:", ", ".join(f"{k}={v}" for k, v in sorted(match_stats.items())))

    if not samples:
        raise ValueError(f"No input images found in {images_dir}")
    return samples


def split_by_group(rows: List[Dict[str, object]], group_key: str, val_ratio: float, seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    group_ids = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    if not group_ids:
        return [], []

    n_val = int(round(len(group_ids) * val_ratio))
    if len(group_ids) > 1:
        n_val = max(1, min(len(group_ids) - 1, n_val))
    else:
        n_val = 0

    val_groups = set(group_ids[:n_val])
    train_rows, val_rows = [], []
    for gid, items in grouped.items():
        if gid in val_groups:
            val_rows.extend(items)
        else:
            train_rows.extend(items)
    return train_rows, val_rows


def build_protocol_splits(
    protocol: str,
    rows: List[Dict[str, object]],
    test_years: Sequence[int],
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    test_years_set = set(test_years)

    # В test идут только "forced test" строки по новой логике
    test_rows = [row for row in rows if is_forced_test_row(row, test_years_set)]

    if protocol == "A":
        # Всё остальное -> pool train/val (включая PXL_/IMG_ даже если year=2025)
        trainval_rows = [row for row in rows if not is_forced_test_row(row, test_years_set)]
        train_rows, val_rows = split_by_group(trainval_rows, "source_id", val_ratio, seed)

    elif protocol == "B":
        for row in rows:
            facade_id = str(row["facade_id"])
            if not facade_id or facade_id.lower().startswith("unknown"):
                raise RuntimeError(
                    f"Protocol B requires non-empty facade_id; got '{facade_id}' in {row['rel_image_path']}"
                )

        # Тестовые фасады определяем только по forced-test строкам
        test_facades = {str(row["facade_id"]) for row in test_rows}

        # В train/val берём всё, что:
        # 1) не forced-test
        # 2) и не пересекается по facade_id с тестовыми фасадами (анти-лик в B)
        trainval_rows = [
            row
            for row in rows
            if (not is_forced_test_row(row, test_years_set)) and str(row["facade_id"]) not in test_facades
        ]
        train_rows, val_rows = split_by_group(trainval_rows, "facade_id", val_ratio, seed)

    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    return {"train": train_rows, "val": val_rows, "test": test_rows}


def write_csv(rows: List[Dict[str, object]], path: Path, fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_tile(image, mask, out_image: Path, out_mask: Path) -> None:
    cv2.imwrite(str(out_image), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_mask), mask)

def tile_and_prepare(
    split_name: str,
    rows: List[Dict[str, object]],
    protocol_tiles_root: Path,
    protocol_root: Path,
    tiling_cfg: Dict[str, object],
    do_augment: bool,
    augmentations_per_tile: int,
    aug_transform,
    zoom_cfg: Dict[str, object],
    seed: int,
    pbar=None,
    mixup_cfg: Optional[Dict[str, object]] = None,
    cutmix_cfg: Optional[Dict[str, object]] = None,
    overlay_cfg: Optional[Dict[str, object]] = None,
    hires_cfg: Optional[Dict[str, object]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """
    hires_cfg ожидается примерно такой:
      {
        "prefixes": ["PXL_", "IMG_"],
        "year": 2025,
        "max_tiles": 40,
        "non_overlap": True
      }
    """
    import hashlib

    random.seed(seed)
    np.random.seed(seed)

    split_images_dir = protocol_tiles_root / split_name / "images"
    split_masks_dir = protocol_tiles_root / split_name / "masks"
    split_images_dir.mkdir(parents=True, exist_ok=True)
    split_masks_dir.mkdir(parents=True, exist_ok=True)

    mixup_cfg = mixup_cfg or {}
    cutmix_cfg = cutmix_cfg or {}
    overlay_cfg = overlay_cfg or {}
    hires_cfg = hires_cfg or {}

    save_overlays = bool(overlay_cfg.get("enabled", False))
    overlay_alpha = float(overlay_cfg.get("alpha", 0.45))
    overlay_palette = overlay_cfg.get("palette", []) or []

    split_overlays_dir = protocol_tiles_root / split_name / "overlays"
    if save_overlays:
        split_overlays_dir.mkdir(parents=True, exist_ok=True)

    tile_manifest: List[Dict[str, object]] = []
    stats = {
        "base_tiles": 0,
        "augmented_tiles": 0,
        "hires_sources_seen": 0,
        "hires_sources_capped": 0,
        "hires_tiles_dropped": 0,
    }

    def _read_image_mask_pair(image_path: Path, mask_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            return None
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, mask

    def _normalize_palette(palette_obj) -> List[Tuple[int, int, int]]:
        out: List[Tuple[int, int, int]] = []
        if not isinstance(palette_obj, list):
            return out
        for item in palette_obj:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                try:
                    r, g, b = int(item[0]), int(item[1]), int(item[2])
                    out.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
                except Exception:
                    continue
        return out

    def _default_color_for_label(label: int) -> Tuple[int, int, int]:
        # deterministic pseudo-random color (skip 0 = background)
        if label <= 0:
            return (0, 0, 0)
        return (
            int((37 * label + 17) % 256),
            int((97 * label + 53) % 256),
            int((17 * label + 193) % 256),
        )

    def _make_overlay(image_rgb: np.ndarray, mask_lbl: np.ndarray) -> np.ndarray:
        img = image_rgb.copy()
        if mask_lbl.ndim == 3:
            mask_lbl = mask_lbl[..., 0]
        mask_lbl = mask_lbl.astype(np.int32, copy=False)

        h, w = img.shape[:2]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        palette = _normalize_palette(overlay_palette)
        labels = np.unique(mask_lbl)
        for lab in labels:
            if int(lab) == 0:
                continue
            if palette:
                color = palette[(int(lab) - 1) % len(palette)]
            else:
                color = _default_color_for_label(int(lab))
            color_mask[mask_lbl == lab] = color

        # blend only on non-background pixels
        fg = (mask_lbl > 0)
        if np.any(fg):
            blended = img.astype(np.float32)
            cm = color_mask.astype(np.float32)
            blended[fg] = (1.0 - overlay_alpha) * blended[fg] + overlay_alpha * cm[fg]
            img = np.clip(blended, 0, 255).astype(np.uint8)
        return img

    def _save_overlay(image_rgb: np.ndarray, mask_lbl: np.ndarray, out_path: Path) -> None:
        if not save_overlays:
            return
        overlay_rgb = _make_overlay(image_rgb, mask_lbl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    def _sample_partner_row(exclude_source_id: str, max_tries: int = 16) -> Optional[Dict[str, object]]:
        if not rows:
            return None
        if len(rows) == 1:
            return rows[0]

        for _ in range(max_tries):
            candidate = random.choice(rows)
            if str(candidate.get("source_id")) != str(exclude_source_id):
                return candidate

        return random.choice(rows)

    def _sample_partner_tile(
        exclude_source_id: str,
        target_hw: Tuple[int, int],
        max_tries: int = 12,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a random tile (image, mask) from another image in the same split.
        """
        target_h, target_w = target_hw

        for _ in range(max_tries):
            partner_row = _sample_partner_row(exclude_source_id)
            if partner_row is None:
                return None

            partner_image_path = Path(partner_row["image_path"])
            partner_mask_path = Path(partner_row["mask_path"])
            loaded = _read_image_mask_pair(partner_image_path, partner_mask_path)
            if loaded is None:
                continue

            partner_image, partner_mask = loaded
            partner_tiles = list(
                generate_tiles(
                    image=partner_image,
                    mask=partner_mask,
                    tile_h=int(tiling_cfg["tile_size"]),
                    tile_w=int(tiling_cfg["tile_size"]),
                    stride_h=int(tiling_cfg["stride"]),
                    stride_w=int(tiling_cfg["stride"]),
                    pad_mode=str(tiling_cfg["pad_mode"]),
                    min_content_ratio=float(tiling_cfg["min_content_ratio"]),
                )
            )
            if not partner_tiles:
                continue

            _, _, _, p_img_tile, p_mask_tile = random.choice(partner_tiles)

            if p_img_tile.shape[:2] != (target_h, target_w):
                p_img_tile = cv2.resize(p_img_tile, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                p_mask_tile = cv2.resize(p_mask_tile, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            return p_img_tile, p_mask_tile

        return None

    def _safe_apply_mixup(
        img: np.ndarray,
        mask: np.ndarray,
        partner_img: np.ndarray,
        partner_mask: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return apply_mixup(img, mask, partner_img, partner_mask, alpha=alpha)
        except TypeError:
            return apply_mixup(img, mask, partner_img, partner_mask, alpha)

    def _safe_apply_cutmix(
        img: np.ndarray,
        mask: np.ndarray,
        partner_img: np.ndarray,
        partner_mask: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return apply_cutmix(img, mask, partner_img, partner_mask, alpha=alpha)
        except TypeError:
            return apply_cutmix(img, mask, partner_img, partner_mask, alpha)

    # ---------- HI-RES helpers (PXL_/IMG_) ----------
    def _is_hires_row(row_obj: Dict[str, object]) -> bool:
        # ограничиваем только train/val (как договаривались)
        if split_name not in {"train", "val"}:
            return False

        try:
            y = int(row_obj.get("year", -1))
        except Exception:
            return False

        hires_year = int(hires_cfg.get("year", 2025))
        if y != hires_year:
            return False

        prefixes = hires_cfg.get("prefixes", ["PXL_", "IMG_"])
        if not isinstance(prefixes, (list, tuple)) or not prefixes:
            prefixes = ["PXL_", "IMG_"]

        img_path_obj = row_obj.get("image_path")
        if isinstance(img_path_obj, Path):
            stem = img_path_obj.stem
        else:
            stem = Path(str(row_obj.get("rel_image_path", ""))).stem

        return any(stem.startswith(str(p)) for p in prefixes)

    def _mask_content_score(mask_tile: np.ndarray) -> float:
        """
        Score = доля пикселей НЕ фона (label > 0) среди валидных.
        ignore=255 исключаем из знаменателя.
        """
        m = mask_tile
        if m is None:
            return 0.0
        if m.ndim == 3:
            m = m[..., 0]
        m = m.astype(np.int32, copy=False)

        valid = (m != 255)
        if not np.any(valid):
            return 0.0

        fg = (m > 0) & valid
        return float(fg.sum() / valid.sum())

    def _select_hires_tiles_by_content(
        source_id: str,
        tiles_all: List[Tuple[int, int, int, np.ndarray, np.ndarray]],
        max_tiles: int,
    ) -> List[Tuple[int, int, int, np.ndarray, np.ndarray]]:
        """
        Берём top-K тайлов по max-content, tie-break детерминированный от seed+source_id.
        """
        if max_tiles <= 0 or len(tiles_all) <= max_tiles:
            return tiles_all

        # стабильный tie-break на каждый source, чтобы результат не зависел от порядка обхода
        h = hashlib.md5(f"{seed}::{source_id}".encode("utf-8")).hexdigest()
        local_seed = int(h[:8], 16)
        rng = random.Random(local_seed)

        scored = []
        for t in tiles_all:
            tile_idx, x, y, img_t, msk_t = t
            s = _mask_content_score(msk_t)
            scored.append((s, rng.random(), t))  # rng.random() = deterministic tie-break

        scored.sort(key=lambda z: (-z[0], z[1]))
        return [t for _, _, t in scored[:max_tiles]]

    for row in rows:
        try:
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])

            loaded = _read_image_mask_pair(image_path, mask_path)
            if loaded is None:
                print(
                    "[WARN] Failed to read image/mask pair, skipping:\n"
                    f"       image: {image_path} (exists={image_path.exists()})\n"
                    f"       mask : {mask_path} (exists={mask_path.exists()})"
                )
                continue

            image, mask = loaded

            tile_size = int(tiling_cfg["tile_size"])
            base_stride = int(tiling_cfg["stride"])

            is_hires = _is_hires_row(row)
            if is_hires:
                stats["hires_sources_seen"] += 1

            # non-overlap для hires (если включено)
            if is_hires and bool(hires_cfg.get("non_overlap", True)):
                stride_h = tile_size
                stride_w = tile_size
            else:
                stride_h = base_stride
                stride_w = base_stride

            tiles_all = list(
                generate_tiles(
                    image=image,
                    mask=mask,
                    tile_h=tile_size,
                    tile_w=tile_size,
                    stride_h=stride_h,
                    stride_w=stride_w,
                    pad_mode=str(tiling_cfg["pad_mode"]),
                    min_content_ratio=float(tiling_cfg["min_content_ratio"]),
                )
            )

            # cap по max-content для hires
            if is_hires:
                max_tiles = int(hires_cfg.get("max_tiles", 40))
                if max_tiles > 0 and len(tiles_all) > max_tiles:
                    before = len(tiles_all)
                    tiles_all = _select_hires_tiles_by_content(
                        source_id=str(row["source_id"]),
                        tiles_all=tiles_all,
                        max_tiles=max_tiles,
                    )
                    dropped = before - len(tiles_all)
                    if dropped > 0:
                        stats["hires_sources_capped"] += 1
                        stats["hires_tiles_dropped"] += dropped

            for tile_idx, x, y, image_tile, mask_tile in tiles_all:
                # ---------- base tile ----------
                tile_id = f"{row['source_id']}_x{x}_y{y}_tile{tile_idx}"
                file_name = f"{tile_id}.png"
                out_image = split_images_dir / file_name
                out_mask = split_masks_dir / file_name
                save_tile(image_tile, mask_tile, out_image, out_mask)

                if save_overlays:
                    out_overlay = split_overlays_dir / file_name
                    _save_overlay(image_tile, mask_tile, out_overlay)

                base_item = {
                    "rel_image_path": str(out_image.relative_to(protocol_root)),
                    "rel_mask_path": str(out_mask.relative_to(protocol_root)),
                    "year": row["year"],
                    "facade_id": row["facade_id"],
                    "source_id": row["source_id"],
                    "tile_id": tile_id,
                }
                if save_overlays:
                    base_item["rel_overlay_path"] = str((split_overlays_dir / file_name).relative_to(protocol_root))

                tile_manifest.append(base_item)
                stats["base_tiles"] += 1

                if not do_augment:
                    continue

                # ---------- augmented copies ----------
                for aug_idx in range(augmentations_per_tile):
                    transformed = aug_transform(image=image_tile, mask=mask_tile)
                    aug_image = transformed["image"]
                    aug_mask = transformed["mask"]
                    aug_image, aug_mask = apply_zoom(aug_image, aug_mask, zoom_cfg)

                    partner_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
                    target_hw = aug_image.shape[:2]

                    # MixUp
                    if mixup_cfg.get("p", 0) > 0 and random.random() < float(mixup_cfg.get("p", 0)):
                        partner_cache = _sample_partner_tile(
                            exclude_source_id=str(row["source_id"]),
                            target_hw=target_hw,
                        )
                        if partner_cache is not None:
                            p_img_tile, p_mask_tile = partner_cache

                            p_transformed = aug_transform(image=p_img_tile, mask=p_mask_tile)
                            p_aug_image = p_transformed["image"]
                            p_aug_mask = p_transformed["mask"]
                            p_aug_image, p_aug_mask = apply_zoom(p_aug_image, p_aug_mask, zoom_cfg)

                            if p_aug_image.shape[:2] != aug_image.shape[:2]:
                                h, w = aug_image.shape[:2]
                                p_aug_image = cv2.resize(p_aug_image, (w, h), interpolation=cv2.INTER_LINEAR)
                                p_aug_mask = cv2.resize(p_aug_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                            aug_image, aug_mask = _safe_apply_mixup(
                                aug_image,
                                aug_mask,
                                p_aug_image,
                                p_aug_mask,
                                float(mixup_cfg.get("alpha", 0.3)),
                            )

                    # CutMix
                    if cutmix_cfg.get("p", 0) > 0 and random.random() < float(cutmix_cfg.get("p", 0)):
                        if partner_cache is None:
                            partner_cache = _sample_partner_tile(
                                exclude_source_id=str(row["source_id"]),
                                target_hw=target_hw,
                            )

                        if partner_cache is not None:
                            p_img_tile, p_mask_tile = partner_cache

                            p_transformed = aug_transform(image=p_img_tile, mask=p_mask_tile)
                            p_aug_image = p_transformed["image"]
                            p_aug_mask = p_transformed["mask"]
                            p_aug_image, p_aug_mask = apply_zoom(p_aug_image, p_aug_mask, zoom_cfg)

                            if p_aug_image.shape[:2] != aug_image.shape[:2]:
                                h, w = aug_image.shape[:2]
                                p_aug_image = cv2.resize(p_aug_image, (w, h), interpolation=cv2.INTER_LINEAR)
                                p_aug_mask = cv2.resize(p_aug_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                            aug_image, aug_mask = _safe_apply_cutmix(
                                aug_image,
                                aug_mask,
                                p_aug_image,
                                p_aug_mask,
                                float(cutmix_cfg.get("alpha", 0.8)),
                            )

                    aug_name = f"aug_{tile_id}_{aug_idx}.png"
                    aug_image_path = split_images_dir / aug_name
                    aug_mask_path = split_masks_dir / aug_name
                    save_tile(aug_image, aug_mask, aug_image_path, aug_mask_path)

                    if save_overlays:
                        aug_overlay_path = split_overlays_dir / aug_name
                        _save_overlay(aug_image, aug_mask, aug_overlay_path)

                    aug_item = {
                        "rel_image_path": str(aug_image_path.relative_to(protocol_root)),
                        "rel_mask_path": str(aug_mask_path.relative_to(protocol_root)),
                        "year": row["year"],
                        "facade_id": row["facade_id"],
                        "source_id": row["source_id"],
                        "tile_id": f"aug_{tile_id}_{aug_idx}",
                    }
                    if save_overlays:
                        aug_item["rel_overlay_path"] = str((split_overlays_dir / aug_name).relative_to(protocol_root))

                    tile_manifest.append(aug_item)
                    stats["augmented_tiles"] += 1

        finally:
            if pbar is not None:
                pbar.update(1)

    return tile_manifest, stats

def year_distribution(rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(row["year"]) for row in rows).items()))


def run_sanity_checks(protocol: str, splits: Dict[str, List[Dict[str, object]]], test_years: Sequence[int]) -> None:
    test_years_set = set(test_years)
    train_val_rows = splits["train"] + splits["val"]

    # В train/val запрещены только forced-test строки (а не любые year in test_years)
    invalid_rows = [row for row in train_val_rows if is_forced_test_row(row, test_years_set)]
    if invalid_rows:
        raise RuntimeError(
            f"Found {len(invalid_rows)} train/val records that are forced to test by rule."
        )

    if protocol == "B":
        test_facades = {str(row["facade_id"]) for row in splits["test"]}
        train_val_facades = {str(row["facade_id"]) for row in train_val_rows}
        leakage = sorted(test_facades & train_val_facades)
        if leakage:
            raise RuntimeError(f"Protocol B leakage detected for facade_id(s): {leakage}")


def run_protocol(protocol: str, all_rows: List[Dict[str, object]], args: argparse.Namespace, tiling_defaults: Dict[str, object]) -> None:
    protocol_root = args.out_root / f"protocol_{protocol}"
    manifests_dir = protocol_root / "manifests"
    tiles_root = protocol_root / "tiles"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    splits = build_protocol_splits(protocol, all_rows, args.test_years, args.val_ratio, args.seed)
    run_sanity_checks(protocol, splits, args.test_years)

    write_csv(
        splits["train"],
        manifests_dir / "train_sources.csv",
        ["rel_image_path", "rel_mask_path", "year", "facade_id", "source_id"],
    )
    write_csv(
        splits["val"],
        manifests_dir / "val_sources.csv",
        ["rel_image_path", "rel_mask_path", "year", "facade_id", "source_id"],
    )
    write_csv(
        splits["test"],
        manifests_dir / "test_sources.csv",
        ["rel_image_path", "rel_mask_path", "year", "facade_id", "source_id"],
    )

    augment_enabled = bool(args.augment and not args.tile_only)
    aug_transform = None
    zoom_cfg: Dict[str, object] = {}

    tiling_cfg = {
        "tile_size": args.tile_size if args.tile_size is not None else tiling_defaults["tile_size"],
        "stride": args.stride if args.stride is not None else tiling_defaults["stride"],
        "pad_mode": args.pad_mode if args.pad_mode is not None else tiling_defaults["pad_mode"],
        "min_content_ratio": args.min_content_ratio
        if args.min_content_ratio is not None
        else tiling_defaults["min_content_ratio"],
    }

    split_tile_manifests: Dict[str, List[Dict[str, object]]] = {}
    tile_stats: Dict[str, Dict[str, int]] = {}

    # Один progress bar на весь протокол (A/B), считаем по числу исходных изображений
    total_items = sum(len(splits[split_name]) for split_name in ("train", "val", "test"))

    with tqdm(
        total=total_items,
        desc=f"Protocol {protocol}",
        unit="img",
        leave=True,
    ) as protocol_pbar:
        for split_name in ("train", "val", "test"):
            split_do_augment = split_name in {"train"} and augment_enabled
            if split_do_augment and aug_transform is None:
                aug_transform = build_transforms(tiling_defaults["augmentations"])
                zoom_cfg = tiling_defaults["augmentations"].get("zoom", {})

            split_manifest, split_stat = tile_and_prepare(
                split_name=split_name,
                rows=splits[split_name],
                protocol_tiles_root=tiles_root,
                protocol_root=protocol_root,
                tiling_cfg=tiling_cfg,
                do_augment=split_do_augment,
                augmentations_per_tile=int(tiling_defaults["augmentations_per_image"]),
                aug_transform=aug_transform,
                zoom_cfg=zoom_cfg,
                seed=args.seed,
                pbar=protocol_pbar,
                mixup_cfg=tiling_defaults.get("augmentations", {}).get("mixup", {}),
                cutmix_cfg=tiling_defaults.get("augmentations", {}).get("cutmix", {}),
                overlay_cfg={
                    "enabled": True,  # можно потом вынести в config при желании
                    "alpha": float(tiling_defaults.get("overlay_alpha", 0.45)),
                    "palette": tiling_defaults.get("palette", []),
                },
            )
            split_tile_manifests[split_name] = split_manifest
            tile_stats[split_name] = split_stat
            write_csv(
                split_manifest,
                manifests_dir / f"{split_name}.csv",
                ["rel_image_path", "rel_mask_path", "year", "facade_id", "source_id", "tile_id"],
            )

    split_config = {
        "protocol": protocol,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_years": args.test_years,
        "augment": augment_enabled,
        "tile_only": args.tile_only,
        "tiling": tiling_cfg,
        "augment_config": str(args.augment_config),
        "input": {"data_root": str(args.data_root) if args.data_root else None, "coco_json": str(args.coco_json) if args.coco_json else None},
    }

    stats: Dict[str, object] = {
        "sources": {split: len(splits[split]) for split in ("train", "val", "test")},
        "tiles": {
            split: {
                "total": len(split_tile_manifests[split]),
                "base_tiles": tile_stats[split]["base_tiles"],
                "augmented_tiles": tile_stats[split]["augmented_tiles"],
            }
            for split in ("train", "val", "test")
        },
        "source_year_distribution": {split: year_distribution(splits[split]) for split in ("train", "val", "test")},
        "test_augmented_samples": tile_stats["test"]["augmented_tiles"],
    }

    if stats["test_augmented_samples"] != 0:
        raise RuntimeError("test_augmented_samples must be 0")

    train_val_facade_overlap: List[str] = []
    if protocol == "B":
        train_facades = {str(row["facade_id"]) for row in splits["train"]}
        val_facades = {str(row["facade_id"]) for row in splits["val"]}
        test_facades = {str(row["facade_id"]) for row in splits["test"]}
        train_val_facade_overlap = sorted(train_facades & val_facades)
        stats["facade_sets"] = {
            "train": len(train_facades),
            "val": len(val_facades),
            "test": len(test_facades),
        }
        stats["facade_intersections"] = {
            "train_val": train_val_facade_overlap,
            "train_test": sorted(train_facades & test_facades),
            "val_test": sorted(val_facades & test_facades),
        }

    with (manifests_dir / "split_config.json").open("w", encoding="utf-8") as f:
        json.dump(split_config, f, indent=2, ensure_ascii=False)
    with (manifests_dir / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if protocol == "B" and train_val_facade_overlap:
        raise RuntimeError(
            "Protocol B requires disjoint facade_id between train and val. "
            f"Overlap: {train_val_facade_overlap}"
        )

    print(f"[{protocol}] complete. test_augmented_samples={stats['test_augmented_samples']}")


def main() -> None:
    args = parse_args()

    if not args.data_root and not args.coco_json:
        raise ValueError("Provide either --data-root or --coco-json")
    if args.data_root and args.coco_json:
        raise ValueError("Use only one input mode: --data-root or --coco-json")
    if args.coco_json:
        raise NotImplementedError("COCO mode is not implemented yet; use --data-root")

    args.out_root.mkdir(parents=True, exist_ok=True)
    tiling_defaults = load_tiling_defaults(args.augment_config)

    samples = collect_samples_from_root(args.data_root, args.image_exts)

    protocols = [part.strip().upper() for part in args.protocol.split(",") if part.strip()]
    if not protocols:
        raise ValueError("No protocols selected")

    for protocol in protocols:
        if protocol not in {"A", "B"}:
            raise ValueError(f"Unsupported protocol '{protocol}'. Allowed: A, B")
        run_protocol(protocol, samples, args, tiling_defaults)


if __name__ == "__main__":
    main()
