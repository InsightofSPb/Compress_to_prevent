#!/usr/bin/env python3
"""Evaluate stock or fine-tuned MaskCLIP segmentation branch on stitched tiles.

This entrypoint provides the reviewer-facing comparison between:

* ``stock``: pretrained weights loaded by the LPOSS configuration, evaluated
  zero-shot with the same facade class names and the same tiled stitching path;
* ``finetuned``: a checkpoint produced by ``tools/finetune_tiled.py``.

Model selection must be performed on validation only. Test is intended for a
single final evaluation after choosing the fine-tuned checkpoint by validation
DAMAGE_MACRO_MIOU.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mmcv  # type: ignore  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from mmseg.datasets import build_dataset  # noqa: E402

import finetune as common  # noqa: E402
from finetune_tiled import evaluate_stitched_tiles  # noqa: E402
from models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitched segmentation evaluation for stock/fine-tuned branches.")
    parser.add_argument("config", help="Hydra model config, for example lposs")
    parser.add_argument("--eval-dataset-config", required=True,
                        help="MMSeg tiled evaluation config; its root is reused while split subdirectory is selected here.")
    parser.add_argument("--tiles-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Fine-tuned .pth checkpoint. Omit for stock/pretrained evaluation.")
    parser.add_argument("--model-label", default=None,
                        help="Output model label; defaults to stock or finetuned based on checkpoint presence.")
    parser.add_argument("--mix-strategy", choices=("add", "concat", "replace"), default="add")
    parser.add_argument("--use-embedding-mixer", action="store_true",
                        help="Must match the architecture used to create the supplied checkpoint.")
    parser.add_argument("--loss-mode", choices=("ce_weighted", "ce_dice", "focal_dice"), default="focal_dice")
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--save-visualizations", type=int, default=-1,
                        help="Number of stitched visualizations; -1 saves all source images.")
    return parser.parse_args()


def build_eval_dataset(config_path: str, split: str):
    cfg = mmcv.Config.fromfile(config_path)
    dataset_cfg = cfg.data.val.copy()
    dataset_cfg["img_dir"] = "{}/images".format(split)
    dataset_cfg["ann_dir"] = "{}/masks".format(split)
    return build_dataset(dataset_cfg, dict(test_mode=False))


def build_wrapper(cfg, class_names: List[str], mix_strategy: str, use_embedding_mixer: bool) -> nn.Module:
    base_model = build_model(cfg.model, class_names=class_names).cuda()
    mixers: List[nn.Module] = []
    if use_embedding_mixer:
        decode_head = getattr(base_model, "decode_head", None)
        if decode_head is None and hasattr(base_model, "clip_backbone"):
            decode_head = getattr(base_model.clip_backbone, "decode_head", None)
        channels = getattr(decode_head, "text_channels", 512)
        mixers.append(common.EmbeddingMixer(channels))
    return common.FineTuneWrapper(base_model, mixers=mixers, mix_strategy=mix_strategy).cuda()


def print_metric_summary(metrics: dict) -> None:
    print("mIoU={:.6f} mIoU_no_background={:.6f} DAMAGE_MACRO_MIOU={:.6f} mF1={:.6f}".format(
        metrics.get("mIoU", 0.0),
        metrics.get("mIoU_no_background", 0.0),
        metrics.get("DAMAGE_MACRO_MIOU", 0.0),
        metrics.get("mF1", 0.0),
    ))
    print("Per-class IoU:")
    for name, values in metrics.get("class_metrics", {}).items():
        print("  {:18s} {:.6f}".format(name, values.get("iou", 0.0)))
    print("Grouped IoU:")
    for name, values in metrics.get("group_metrics", {}).items():
        print("  {:18s} {:.6f}".format(name, values.get("iou", 0.0)))


def main() -> None:
    args = parse_args()
    model_label = args.model_label or ("finetuned" if args.checkpoint is not None else "stock")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(args.checkpoint))
    if not args.tiles_manifest.is_file():
        raise FileNotFoundError("Tiles manifest does not exist: {}".format(args.tiles_manifest))

    config_dir = PROJECT_ROOT / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=args.config)

    dataset = build_eval_dataset(args.eval_dataset_config, args.split)
    class_names = list(dataset.CLASSES)
    ignore_index = getattr(dataset, "ignore_index", 255)
    wrapper = build_wrapper(cfg, class_names, args.mix_strategy, args.use_embedding_mixer)

    checkpoint_metadata = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        wrapper.load_state_dict(state, strict=True)
        checkpoint_metadata = {
            "path": str(args.checkpoint),
            "epoch": checkpoint.get("epoch"),
            "selection_score_name": checkpoint.get("selection_score_name"),
            "selection_score_value": checkpoint.get("selection_score_value"),
            "saved_metrics": checkpoint.get("metrics"),
        }

    run_dir = args.output_root / "{}_{}".format(model_label, args.split)
    viz_dir = run_dir / "viz"
    run_dir.mkdir(parents=True, exist_ok=True)
    loss_cfg = {"mode": args.loss_mode, "dice_weight": args.dice_weight, "focal_gamma": args.focal_gamma}
    metrics, auxiliary_loss = evaluate_stitched_tiles(
        wrapper, dataset, args.tiles_manifest, class_names, loss_cfg,
        class_weights=None, ignore_index=ignore_index,
        viz_dir=viz_dir, max_visualizations=args.save_visualizations,
    )

    report = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_label": model_label,
        "model_variant": "pretrained_stock_maskclip_branch" if args.checkpoint is None else "finetuned_maskclip_branch",
        "architecture_note": "Evaluation follows the FineTuneWrapper MaskCLIP segmentation branch; full LPOSS DINO refinement is not invoked.",
        "split": args.split,
        "eval_dataset_config": args.eval_dataset_config,
        "tiles_manifest": str(args.tiles_manifest),
        "n_source_images": len({row["source_id"] for row in __import__("csv").DictReader(args.tiles_manifest.open("r", encoding="utf-8"))}),
        "checkpoint": checkpoint_metadata,
        "class_names": class_names,
        "ignore_index": ignore_index,
        "metrics": metrics,
        "auxiliary_unweighted_loss": float(auxiliary_loss),
        "auxiliary_loss_note": "Metrics are primary; this loss is computed without train-derived class weights for evaluation-only reproducibility.",
        "visualizations_dir": str(viz_dir),
    }
    report_path = run_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Model variant:", report["model_variant"])
    print("Split:", args.split)
    print_metric_summary(metrics)
    print("Report:", report_path)
    print("Visualizations:", viz_dir)


if __name__ == "__main__":
    main()
