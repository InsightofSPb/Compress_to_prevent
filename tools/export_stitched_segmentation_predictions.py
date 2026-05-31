#!/usr/bin/env python3
"""Export full-resolution stitched semantic prediction masks for temporal figures.

The evaluator used for the facade-disjoint segmentation comparison saves visual
triptychs, but temporal qualitative figures require raw predicted label maps for
both observations of a facade pair. This script reuses exactly the same tiled
stitching and model construction protocol as ``evaluate_segmentation_tiled.py``
and writes one indexed uint8 PNG prediction per source RGB image together with a
CSV manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_ROOT = Path(__file__).resolve().parent
for item in (PROJECT_ROOT, TOOLS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import cv2  # noqa: E402
import mmcv  # type: ignore  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from evaluate_segmentation_tiled import build_eval_dataset, build_wrapper  # noqa: E402
from finetune_tiled import group_tiles, read_tile_manifest, tile_dataset_index  # noqa: E402


FIELDS = [
    "source_id", "image_stem", "source_image", "source_mask", "prediction_path",
    "split", "model_label", "checkpoint", "height", "width",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stitched LPOSS/MaskCLIP prediction masks.")
    parser.add_argument("config", help="Hydra model config, for example lposs")
    parser.add_argument("--eval-dataset-config", required=True)
    parser.add_argument("--tiles-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Fine-tuned checkpoint; omit to export stock predictions.")
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--mix-strategy", choices=("add", "concat", "replace"), default="add")
    parser.add_argument("--use-embedding-mixer", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_palette_preview(out_path: Path, class_names: Sequence[str], palette: Sequence[Sequence[int]]) -> None:
    width, row_height = 520, 34
    canvas = np.zeros((row_height * len(class_names), width, 3), dtype=np.uint8)
    for idx, (name, colour) in enumerate(zip(class_names, palette)):
        y0, y1 = idx * row_height, (idx + 1) * row_height
        rgb = tuple(int(v) for v in colour)
        bgr = (rgb[2], rgb[1], rgb[0])
        canvas[y0 + 4:y1 - 4, 8:52] = bgr
        cv2.putText(canvas, "{}: {}".format(idx, name), (68, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (235, 235, 235), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not args.tiles_manifest.is_file():
        raise FileNotFoundError("Tiles manifest does not exist: {}".format(args.tiles_manifest))
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(args.checkpoint))
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("Output directory is not empty: {}. Pass --overwrite.".format(args.out_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.out_dir / "masks"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    model_label = args.model_label or ("finetuned" if args.checkpoint is not None else "stock")
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name=args.config)
    dataset = build_eval_dataset(args.eval_dataset_config, args.split)
    class_names = list(dataset.CLASSES)
    palette = dataset.PALETTE
    wrapper = build_wrapper(cfg, class_names, args.mix_strategy, args.use_embedding_mixer)
    checkpoint_metadata = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        wrapper.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
        checkpoint_metadata = {
            "path": str(args.checkpoint),
            "epoch": checkpoint.get("epoch"),
            "selection_score_name": checkpoint.get("selection_score_name"),
            "selection_score_value": checkpoint.get("selection_score_value"),
        }
    wrapper.eval()
    device = next(wrapper.parameters()).device

    tile_index = tile_dataset_index(dataset)
    grouped = group_tiles(read_tile_manifest(args.tiles_manifest))
    output_rows: List[Dict[str, object]] = []
    for source_id in tqdm(sorted(grouped), desc="Export stitched predictions", unit="image"):
        tiles = grouped[source_id]
        height, width = int(tiles[0]["original_height"]), int(tiles[0]["original_width"])
        logits_sum = torch.zeros((len(class_names), height, width), dtype=torch.float32)
        coverage = torch.zeros((height, width), dtype=torch.float32)
        for row in tiles:
            tile_name = Path(row["image_path"]).name
            sample = dataset[tile_index[tile_name]]
            image_tensor = sample["img"].data
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            logits, _ = wrapper(image_tensor.to(device))
            logits = F.interpolate(logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)[0].cpu()
            x, y = int(row["x"]), int(row["y"])
            h, w = int(row["content_height"]), int(row["content_width"])
            logits_sum[:, y:y + h, x:x + w] += logits[:, :h, :w]
            coverage[y:y + h, x:x + w] += 1.0
        if float(coverage.min()) < 1.0:
            raise RuntimeError("Uncovered pixels in stitched prediction: {}".format(source_id))
        prediction = (logits_sum / coverage.unsqueeze(0)).argmax(dim=0).numpy().astype(np.uint8)
        out_path = prediction_dir / (Path(tiles[0]["source_image"]).stem + ".png")
        if not cv2.imwrite(str(out_path), prediction):
            raise OSError("Could not write prediction mask: {}".format(out_path))
        output_rows.append({
            "source_id": source_id,
            "image_stem": Path(tiles[0]["source_image"]).stem,
            "source_image": tiles[0]["source_image"],
            "source_mask": tiles[0]["source_mask"],
            "prediction_path": str(out_path.resolve()),
            "split": args.split,
            "model_label": model_label,
            "checkpoint": str(args.checkpoint) if args.checkpoint else "",
            "height": height,
            "width": width,
        })

    manifest_path = args.out_dir / "prediction_manifest.csv"
    write_csv(manifest_path, output_rows)
    save_palette_preview(args.out_dir / "semantic_palette.png", class_names, palette)
    report = {
        "model_label": model_label,
        "model_variant": "finetuned_maskclip_branch" if args.checkpoint else "pretrained_stock_maskclip_branch",
        "split": args.split,
        "checkpoint": checkpoint_metadata,
        "tiles_manifest": str(args.tiles_manifest),
        "n_source_images": len(output_rows),
        "class_names": class_names,
        "palette": palette,
        "prediction_manifest": str(manifest_path),
        "prediction_dir": str(prediction_dir),
        "note": "Predictions are stitched full-resolution indexed class masks exported with the same inference protocol as segmentation evaluation.",
    }
    report_path = args.out_dir / "prediction_export_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Exported stitched prediction masks:", len(output_rows))
    print("Masks:", prediction_dir)
    print("Manifest:", manifest_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
