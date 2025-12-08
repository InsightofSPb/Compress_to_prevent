import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a dataset with paired images and masks into train/val/test subsets. "
            "Expected input layout: <root>/images and <root>/masks with matching filenames."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the dataset root containing 'images' and 'masks' subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Where to place the split dataset. Defaults to <data-root>/data_prepared "
            "with 'train', 'val', and 'test' subfolders."
        ),
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Portion of samples for the train split."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Portion of samples for the validation split."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Portion of samples for the test split."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling before the split."
    )
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        help="Image extensions to consider when listing samples.",
    )
    return parser.parse_args()


def collect_samples(images_dir: Path, masks_dir: Path, exts: List[str]) -> List[Tuple[Path, Path]]:
    samples: List[Tuple[Path, Path]] = []
    ext_set = {ext.lower() for ext in exts}
    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in ext_set or not image_path.is_file():
            continue
        mask_path = masks_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image: {image_path}")
        samples.append((image_path, mask_path))
    if not samples:
        raise ValueError(f"No samples found in {images_dir} with extensions {exts}")
    return samples


def split_indices(num_samples: int, ratios: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    if any(r < 0 for r in ratios.values()):
        raise ValueError("Split ratios must be non-negative.")
    ratio_sum = sum(ratios.values())
    if not 0.99 <= ratio_sum <= 1.01:
        raise ValueError("Split ratios must sum to 1.0.")

    indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(indices)

    train_end = int(num_samples * ratios["train"])
    val_end = train_end + int(num_samples * ratios["val"])

    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def prepare_output_dirs(base_output: Path) -> Dict[str, Dict[str, Path]]:
    splits = {}
    for split in ("train", "val", "test"):
        split_root = base_output / split
        image_dir = split_root / "images"
        mask_dir = split_root / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        splits[split] = {"images": image_dir, "masks": mask_dir}
    return splits


def copy_sample(image_src: Path, mask_src: Path, dest_dirs: Dict[str, Path]) -> None:
    shutil.copy2(image_src, dest_dirs["images"] / image_src.name)
    shutil.copy2(mask_src, dest_dirs["masks"] / mask_src.name)


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"

    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            "Expected 'images' and 'masks' subdirectories under the data root."
        )

    samples = collect_samples(images_dir, masks_dir, args.image_exts)

    output_dir = args.output_dir or data_root / "data_prepared"
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    split_map = split_indices(len(samples), ratios, args.seed)
    dest_dirs = prepare_output_dirs(output_dir)

    for split, indices in split_map.items():
        for idx in indices:
            image_src, mask_src = samples[idx]
            copy_sample(image_src, mask_src, dest_dirs[split])

    print(
        f"Split {len(samples)} samples into train={len(split_map['train'])}, "
        f"val={len(split_map['val'])}, test={len(split_map['test'])}."
    )
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
