import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import sys

import mmcv  # type: ignore
import numpy as np
import torch
from hydra import compose, initialize
from mmseg.apis import single_gpu_test
from mmseg.core.evaluation.metrics import eval_metrics

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.logger import get_logger
from models import build_model
from segmentation.evaluation import (
    build_seg_dataloader,
    build_seg_dataset,
    build_seg_inference,
)

STRUCTURAL_DAMAGE = {"CRACK", "SPALLING", "DELAMINATION", "MISSING_ELEMENT"}
SURFACE_STAIN = {"WATER_STAIN", "EFFLORESCENCE", "CORROSION"}
HUMAN_ACTIVITY = {"TEXT_OR_IMAGES", "REPAIRS"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare base and finetuned models on the facades test set"
    )
    parser.add_argument("config", help="Hydra config name (without extension)")
    parser.add_argument(
        "dataset_config", help="MMSegmentation dataset config for the test split"
    )
    parser.add_argument(
        "finetuned_checkpoint", help="Path to the finetuned model checkpoint"
    )
    parser.add_argument(
        "output_dir", help="Directory to store metrics and visualizations"
    )
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Optional path to a base model checkpoint; if omitted, use defaults",
    )
    parser.add_argument(
        "--num-gradcam",
        type=int,
        default=10,
        help="Number of test images for Grad-CAM generation",
    )
    parser.add_argument(
        "--background-class-id",
        type=int,
        default=None,
        help="Optional background class id to omit in overlays",
    )
    parser.add_argument(
        "--ignore-index", type=int, default=255, help="Ignore index used in masks"
    )
    return parser.parse_args()


def load_model(
    cfg, checkpoint_path: Optional[str], class_names: List[str], device: torch.device
):
    logger = get_logger()
    model = build_model(cfg.model, class_names=class_names)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(
                "Missing keys when loading %s: %s", checkpoint_path, missing
            )
        if unexpected:
            logger.warning(
                "Unexpected keys when loading %s: %s", checkpoint_path, unexpected
            )
    else:
        logger.info(
            "No base checkpoint provided; using pretrained weights from the config"
        )
    model.to(device)
    model.eval()
    return model


def evaluate_predictions(dataset, results: List[np.ndarray]) -> Dict:
    metrics = eval_metrics(
        results,
        dataset.get_gt_seg_maps(),
        num_classes=len(dataset.CLASSES),
        ignore_index=dataset.ignore_index,
        label_map=dict(),
        reduce_zero_label=dataset.reduce_zero_label,
        metrics=["mIoU", "mDice"],
    )
    return metrics


def group_metrics(metrics: Dict, class_names: List[str]) -> Dict[str, Dict[str, float]]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    grouped = {}

    def collect(group: set) -> Dict[str, float]:
        idxs = [class_to_idx[name] for name in class_names if name in group]
        iou = float(np.nanmean(metrics["IoU"][idxs])) if idxs else float("nan")
        dice = float(np.nanmean(metrics["Dice"][idxs])) if idxs else float("nan")
        return {"mIoU": iou, "F1": dice}

    grouped["structural_damage"] = collect(STRUCTURAL_DAMAGE)
    grouped["surface_stain"] = collect(SURFACE_STAIN)
    grouped["human_activity"] = collect(HUMAN_ACTIVITY)
    return grouped


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    palette: List[List[int]],
    ignore_index: int,
    background_id: Optional[int],
):
    overlay = image.copy()
    color_mask = np.zeros_like(image)
    for idx, color in enumerate(palette):
        if idx == background_id:
            continue
        color_mask[mask == idx] = color
    valid = mask != ignore_index
    if background_id is not None:
        valid &= mask != background_id
    overlay[valid] = (0.6 * overlay[valid] + 0.4 * color_mask[valid]).astype(np.uint8)
    return overlay


def save_overlays(
    dataset, predictions: List[np.ndarray], output_root: str, background_id: Optional[int], ignore_index: int
):
    os.makedirs(output_root, exist_ok=True)
    for idx, pred in enumerate(predictions):
        img_info = dataset.img_infos[idx]
        img_path = img_info.get("filename", img_info.get("img_path"))
        if img_path is None:
            img_path = os.path.join(dataset.img_dir, img_info["img_suffix"])
        image = mmcv.imread(img_path)
        gt = dataset.get_gt_seg_map_by_idx(idx).astype(np.int64)

        gt_overlay = overlay_mask(image, gt, dataset.PALETTE, ignore_index, background_id)
        pred_overlay = overlay_mask(image, pred, dataset.PALETTE, ignore_index, background_id)

        base_name = os.path.splitext(os.path.basename(img_info.get("ori_filename", img_path)))[0]
        mmcv.imwrite(gt_overlay, os.path.join(output_root, f"{base_name}_gt_overlay.png"))
        mmcv.imwrite(pred_overlay, os.path.join(output_root, f"{base_name}_pred_overlay.png"))


def compute_gradcam(model, image_tensor: torch.Tensor, target_class: int, device: torch.device):
    model.eval()
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True
    dino_feats, clip_feats, clf = model(image_tensor)
    clip_feats.retain_grad()
    clip_preds = torch.matmul(clip_feats, clf.T)  # H x W x C
    target_map = clip_preds[..., target_class].mean()
    model.zero_grad()
    target_map.backward(retain_graph=True)
    grads = clip_feats.grad
    weights = grads.mean(dim=(0, 1))
    cam = torch.relu((clip_feats * weights).sum(dim=-1))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = torch.nn.functional.interpolate(
        cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
    )
    return cam.squeeze().detach().cpu().numpy()


def save_gradcams(
    model, dataset, output_root: str, num_images: int, device: torch.device
):
    os.makedirs(output_root, exist_ok=True)
    for idx in range(min(num_images, len(dataset))):
        data = dataset[idx]
        image_tensor = data["img"].data  # DataContainer
        image_tensor = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor
        img_np = mmcv.imread(dataset.img_infos[idx]["filename"])[:, :, ::-1]
        for class_idx, class_name in enumerate(dataset.CLASSES):
            cam = compute_gradcam(model, image_tensor, class_idx, device)
            heatmap = mmcv.imresize(cam, (img_np.shape[1], img_np.shape[0]))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-6)
            heatmap = np.uint8(255 * heatmap)
            heatmap = mmcv.apply_color_map(heatmap, colormap="jet")
            blended = (0.6 * img_np + 0.4 * heatmap).astype(np.uint8)
            out_path = os.path.join(output_root, f"sample{idx:03d}_{class_name}.png")
            mmcv.imwrite(blended[:, :, ::-1], out_path)


def run_model(
    label: str,
    cfg,
    dataset,
    device: torch.device,
    checkpoint: Optional[str],
    output_dir: str,
    background_id: Optional[int],
    ignore_index: int,
):
    logger = get_logger()
    logger.info("Running evaluation for %s model", label)
    model = load_model(cfg, checkpoint, dataset.CLASSES, device)

    seg_model = build_seg_inference(model, dataset, cfg, args.dataset_config)
    data_loader = build_seg_dataloader(dataset, dist=False)
    predictions = single_gpu_test(seg_model, data_loader, pre_eval=False)

    metrics = evaluate_predictions(dataset, predictions)
    grouped = group_metrics(metrics, list(dataset.CLASSES))

    overlay_dir = os.path.join(output_dir, f"{label}_overlays")
    save_overlays(dataset, predictions, overlay_dir, background_id, ignore_index)

    gradcam_dir = os.path.join(output_dir, f"{label}_gradcam")
    save_gradcams(model, dataset, gradcam_dir, args.num_gradcam, device)

    summary = {
        "overall": {"mIoU": float(metrics["mIoU"]), "F1": float(metrics["mDice"]),},
        "per_class": {
            name: {"mIoU": float(metrics["IoU"][i]), "F1": float(metrics["Dice"][i])}
            for i, name in enumerate(dataset.CLASSES)
        },
        "groups": grouped,
    }
    return summary


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)

    dataset = build_seg_dataset(args.dataset_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_results = run_model(
        "base",
        cfg,
        dataset,
        device,
        args.base_checkpoint,
        args.output_dir,
        args.background_class_id,
        args.ignore_index,
    )
    finetuned_results = run_model(
        "finetuned",
        cfg,
        dataset,
        device,
        args.finetuned_checkpoint,
        args.output_dir,
        args.background_class_id,
        args.ignore_index,
    )

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"base": base_results, "finetuned": finetuned_results}, f, indent=2)
