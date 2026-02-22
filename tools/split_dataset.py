import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from dataset_ops import apply_zoom, build_transforms, generate_tiles

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
    }


def collect_samples_from_root(data_root: Path, image_exts: Sequence[str]) -> List[Dict[str, object]]:
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError("Expected 'images/' and 'masks/' directories under --data-root")

    ext_set = {ext.lower() for ext in image_exts}
    samples: List[Dict[str, object]] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in ext_set:
            continue
        rel_from_images = image_path.relative_to(images_dir)
        mask_path = masks_dir / rel_from_images
        if not mask_path.exists():
            fallback = masks_dir / image_path.name
            if fallback.exists():
                mask_path = fallback
            else:
                raise FileNotFoundError(f"Mask not found for image: {image_path}")

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


def build_protocol_splits(protocol: str, rows: List[Dict[str, object]], test_years: Sequence[int], val_ratio: float, seed: int) -> Dict[str, List[Dict[str, object]]]:
    test_years_set = set(test_years)
    test_rows = [row for row in rows if int(row["year"]) in test_years_set]

    if protocol == "A":
        trainval_rows = [row for row in rows if int(row["year"]) not in test_years_set]
        train_rows, val_rows = split_by_group(trainval_rows, "source_id", val_ratio, seed)
    elif protocol == "B":
        for row in rows:
            facade_id = str(row["facade_id"])
            if not facade_id or facade_id.lower().startswith("unknown"):
                raise RuntimeError(
                    f"Protocol B requires non-empty facade_id; got '{facade_id}' in {row['rel_image_path']}"
                )
        test_facades = {str(row["facade_id"]) for row in test_rows}
        trainval_rows = [
            row
            for row in rows
            if int(row["year"]) not in test_years_set and str(row["facade_id"]) not in test_facades
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
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    random.seed(seed)
    np.random.seed(seed)

    split_images_dir = protocol_tiles_root / split_name / "images"
    split_masks_dir = protocol_tiles_root / split_name / "masks"
    split_images_dir.mkdir(parents=True, exist_ok=True)
    split_masks_dir.mkdir(parents=True, exist_ok=True)

    tile_manifest: List[Dict[str, object]] = []
    stats = {"base_tiles": 0, "augmented_tiles": 0}

    for row in rows:
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read pair: {row['image_path']} / {row['mask_path']}")
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for tile_idx, x, y, image_tile, mask_tile in generate_tiles(
            image=image,
            mask=mask,
            tile_h=int(tiling_cfg["tile_size"]),
            tile_w=int(tiling_cfg["tile_size"]),
            stride_h=int(tiling_cfg["stride"]),
            stride_w=int(tiling_cfg["stride"]),
            pad_mode=str(tiling_cfg["pad_mode"]),
            min_content_ratio=float(tiling_cfg["min_content_ratio"]),
        ):
            tile_id = f"{row['source_id']}_x{x}_y{y}_tile{tile_idx}"
            file_name = f"{tile_id}.png"
            out_image = split_images_dir / file_name
            out_mask = split_masks_dir / file_name
            save_tile(image_tile, mask_tile, out_image, out_mask)

            tile_manifest.append(
                {
                    "rel_image_path": str(out_image.relative_to(protocol_root)),
                    "rel_mask_path": str(out_mask.relative_to(protocol_root)),
                    "year": row["year"],
                    "facade_id": row["facade_id"],
                    "source_id": row["source_id"],
                    "tile_id": tile_id,
                }
            )
            stats["base_tiles"] += 1

            if do_augment:
                for aug_idx in range(augmentations_per_tile):
                    transformed = aug_transform(image=image_tile, mask=mask_tile)
                    aug_image = transformed["image"]
                    aug_mask = transformed["mask"]
                    aug_image, aug_mask = apply_zoom(aug_image, aug_mask, zoom_cfg)
                    aug_name = f"aug_{tile_id}_{aug_idx}.png"
                    aug_image_path = split_images_dir / aug_name
                    aug_mask_path = split_masks_dir / aug_name
                    save_tile(aug_image, aug_mask, aug_image_path, aug_mask_path)

                    tile_manifest.append(
                        {
                            "rel_image_path": str(aug_image_path.relative_to(protocol_root)),
                            "rel_mask_path": str(aug_mask_path.relative_to(protocol_root)),
                            "year": row["year"],
                            "facade_id": row["facade_id"],
                            "source_id": row["source_id"],
                            "tile_id": f"aug_{tile_id}_{aug_idx}",
                        }
                    )
                    stats["augmented_tiles"] += 1

    return tile_manifest, stats


def year_distribution(rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(row["year"]) for row in rows).items()))


def run_sanity_checks(protocol: str, splits: Dict[str, List[Dict[str, object]]], test_years: Sequence[int]) -> None:
    test_years_set = set(test_years)
    train_val_rows = splits["train"] + splits["val"]
    invalid_year_rows = [row for row in train_val_rows if int(row["year"]) in test_years_set]
    if invalid_year_rows:
        raise RuntimeError(
            f"Found {len(invalid_year_rows)} train/val records with test years {sorted(test_years_set)}"
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

    for split_name in ("train", "val", "test"):
        split_do_augment = split_name in {"train", "val"} and augment_enabled
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
