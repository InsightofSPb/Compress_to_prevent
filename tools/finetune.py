"""
Single-GPU fine-tuning script for LPOSS/MaskCLIP models.
"""

import argparse
import datetime
import json
import sys
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple
from collections import deque

import mmcv  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from mmseg.datasets import build_dataloader, build_dataset
from tqdm import tqdm
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.logger import get_logger
from models import build_model


# ============================================================
#                       MODEL WRAPPERS
# ============================================================

class IdentityHead(nn.Module):
    def forward(self, feats: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        return feats


class EmbeddingMixer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 256):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, feats: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        mixed = self.proj_out(self.act(self.proj_in(feats)))
        return feats + mixed


class FineTuneWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        mixers: Optional[List[nn.Module]] = None,
        mix_strategy: str = "add",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.mixers = nn.ModuleList(mixers or [IdentityHead()])
        self.mix_strategy = mix_strategy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        backbone = getattr(self.base_model, "clip_backbone", None) or self.base_model

        try:
            logits, feats = backbone(x, return_feat=True)
        except TypeError:
            output = backbone(x)
            if isinstance(output, tuple):
                logits, feats = output[0], output[1] if len(output) > 1 else None
            else:
                logits, feats = output, None

        if feats is None:
            feats = logits

        mixed_feats = feats
        for head in self.mixers:
            mixed_feats = head(mixed_feats, logits)

        decode_head = getattr(self.base_model, "decode_head", None)
        if decode_head is None and hasattr(self.base_model, "clip_backbone"):
            decode_head = getattr(self.base_model.clip_backbone, "decode_head", None)

        if decode_head is None:
            raise AttributeError("No decode head found for fine-tuning")

        mixed_logits = decode_head.cls_seg(mixed_feats)

        if self.mix_strategy == "replace":
            combined = mixed_logits
        elif self.mix_strategy == "concat":
            combined = torch.stack([logits, mixed_logits], dim=0).mean(dim=0)
        else:
            combined = (logits + mixed_logits) / 2

        return combined, {"base_logits": logits, "mixed_logits": mixed_logits}


# ============================================================
#                       FREEZING
# ============================================================

def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_last(blocks: nn.ModuleList, depth: int) -> None:
    if depth <= 0:
        return
    for block in blocks:
        _freeze_module(block)
    for block in blocks[-depth:]:
        for param in block.parameters():
            param.requires_grad = True


def configure_trainable_layers(model: nn.Module, depth: int) -> None:
    if hasattr(model, "clip_backbone"):
        backbone = getattr(model.clip_backbone, "backbone", None)
        if backbone is not None and hasattr(backbone.visual.transformer, "resblocks"):
            _unfreeze_last(backbone.visual.transformer.resblocks, depth)

    if hasattr(model, "dino_encoder") and hasattr(model.dino_encoder, "blocks"):
        _unfreeze_last(model.dino_encoder.blocks, depth)

    if hasattr(model, "decode_head"):
        for param in model.decode_head.parameters():
            param.requires_grad = True


# ============================================================
#                       DATALOADERS
# ============================================================

def build_dataloaders(train_cfg: str, batch_size: int, workers: int, val_cfg: Optional[str] = None):
    train_cfg = mmcv.Config.fromfile(train_cfg)
    train_dataset = build_dataset(train_cfg.data.train)

    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=workers,
        dist=False,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )

    if val_cfg:
        val_cfg_obj = mmcv.Config.fromfile(val_cfg)
        val_dataset = build_dataset(val_cfg_obj.data.val, dict(test_mode=False))

        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=workers,
            dist=False,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True,
        )
    else:
        val_loader = None
    return train_loader, val_loader


# ============================================================
#                       LOSS + CLASS WEIGHTS
# ============================================================

def compute_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits, size=target.shape[-2:], mode="bilinear", align_corners=False
        )

    num_classes = logits.shape[1]

    if target.min() < 0 or target.max() >= num_classes:
        target = target.clone()
        invalid_mask = (target < 0) | (target >= num_classes)
        target[invalid_mask] = ignore_index

    return F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=ignore_index,
    )


def compute_class_weights(loader, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    pixel_counts = torch.zeros(num_classes, dtype=torch.float64)

    for data in loader:
        gt = data["gt_semantic_seg"].data[0].squeeze(1)  # (B, H, W)
        for c in range(num_classes):
            pixel_counts[c] += (gt == c).sum().item()

    # На всякий случай — игнорим индекс, если он вдруг попал в диапазон (обычно 255, но пусть будет защита)
    if 0 <= ignore_index < num_classes:
        pixel_counts[ignore_index] = 0.0

    pixel_counts = pixel_counts + 1.0  # защита от деления на ноль
    weights = 1.0 / pixel_counts
    weights = weights / weights.mean()  # нормализация к среднему 1

    print("Class pixel counts:", pixel_counts.tolist())
    print("Class weights:", weights.tolist())

    return weights.float()


def validate_class_coverage(
    train_dataset, val_dataset, num_classes: int, logger
) -> None:
    """Validate that all stages see the same class layout, including background."""

    train_classes = list(getattr(train_dataset, "CLASSES", []))
    if len(train_classes) != num_classes:
        raise ValueError(
            f"Train dataset reports {len(train_classes)} classes,"
            f" but {num_classes} were inferred."
        )

    if val_dataset is not None:
        val_classes = list(getattr(val_dataset, "CLASSES", []))
        if train_classes != val_classes:
            raise ValueError(
                "Train/val class lists differ; all classes (including background) must match."
            )

        val_ignore = getattr(val_dataset, "ignore_index", 255)
        train_ignore = getattr(train_dataset, "ignore_index", 255)
        if val_ignore != train_ignore:
            logger.warning(
                "Train ignore_index=%s but val ignore_index=%s; ensure background/ignored pixels align.",
                train_ignore,
                val_ignore,
            )


def log_parameter_counts(model: nn.Module, logger) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model parameters: total=%d, trainable=%d, frozen=%d",
        total_params,
        trainable_params,
        total_params - trainable_params,
    )


# ============================================================
#                       EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader) -> dict:
    model.eval()
    results = []
    for data in loader:
        imgs = data["img"].data[0].cuda()
        logits, _ = model(imgs)

        target_shape = data["gt_semantic_seg"].data[0].shape[-2:]
        logits = F.interpolate(logits, size=target_shape, mode="bilinear", align_corners=False)

        preds = logits.argmax(dim=1).cpu().numpy()
        results.extend(list(preds))

    metrics = loader.dataset.evaluate(results, metric="mIoU", logger=get_logger())

    # Жёсткая проверка коллапса на первом примере
    u, c = np.unique(results[0], return_counts=True)
    print("[VAL DEBUG] First sample class dist:", dict(zip(u.tolist(), c.tolist())))

    return metrics


# ============================================================
#                       VISUALIZATION
# ============================================================

def overlay_mask(image, mask, palette, ignore_index=255):
    overlay = image.copy()
    color_mask = np.zeros_like(image)

    for idx, color in enumerate(palette):
        color_mask[mask == idx] = color

    valid = mask != ignore_index
    overlay[valid] = (0.6 * overlay[valid] + 0.4 * color_mask[valid]).astype(np.uint8)
    return overlay


def draw_legend(img, classes, palette):
    import cv2
    y = 20
    for i, name in enumerate(classes):
        color = palette[i]
        cv2.rectangle(img, (10, y - 10), (30, y + 10), color, -1)
        cv2.putText(img, name, (40, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22
    return img


@torch.no_grad()
def save_val_visualizations(epoch, model, val_loader, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dataset = val_loader.dataset
    palette = dataset.PALETTE
    classes = dataset.CLASSES

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:10]

    model.eval()

    for idx in indices:
        data = dataset[idx]
        img_info = dataset.img_infos[idx]
        img_path = img_info.get("filename") or img_info.get("img_path")

        if not os.path.isabs(img_path):
            img_path = os.path.join(dataset.img_dir, os.path.basename(img_path))

        img = mmcv.imread(img_path)
        gt = dataset.get_gt_seg_map_by_idx(idx)

        img_tensor = data["img"].data[0]
        if img_tensor.dim() == 3 and img_tensor.size(0) == 1:
            img_tensor = img_tensor.expand(3, -1, -1)

        logits, _ = model(img_tensor.unsqueeze(0).cuda())
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        gt_overlay = overlay_mask(img, gt, palette)
        pred_overlay = overlay_mask(img, pred, palette)

        canvas = np.concatenate([img, gt_overlay, pred_overlay], axis=1)
        canvas = draw_legend(canvas, classes, palette)

        out_path = os.path.join(out_dir, f"epoch{epoch:03d}_idx{idx:03d}.png")
        mmcv.imwrite(canvas, out_path)


# ============================================================
#                       CHECKPOINT
# ============================================================

def save_checkpoint(model, optimizer, out_dir, epoch, metrics, average_loss, best_pool):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "average_loss": average_loss,
    }

    ckpt_path = out_dir / f"epoch_{epoch:04d}_loss_{average_loss:.4f}.pth"
    torch.save(checkpoint, ckpt_path)

    best_pool.append((average_loss, ckpt_path))
    best_pool.sort(key=lambda x: x[0])
    while len(best_pool) > 3:
        _, old_path = best_pool.pop()
        if old_path.exists():
            old_path.unlink()
    return best_pool


# ============================================================
#                       ARGS & MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU fine-tuning entrypoint")
    parser.add_argument("config")
    parser.add_argument("--train-dataset-config", required=True)
    parser.add_argument("--val-dataset-config")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--unfreeze-depth", type=int, default=2)
    parser.add_argument("--mix-strategy", choices=["add", "concat", "replace"], default="add")
    parser.add_argument("--use-embedding-mixer", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = PROJECT_ROOT / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=args.config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"finetune_{timestamp}_lr{args.learning_rate}_depth{args.unfreeze_depth}_bs{args.batch_size}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(cfg)
    logger.info("Starting fine-tuning")

    train_loader, val_loader = build_dataloaders(
        args.train_dataset_config,
        args.batch_size,
        args.num_workers,
        args.val_dataset_config,
    )

    class_names = train_loader.dataset.CLASSES
    num_classes = len(class_names)

    validate_class_coverage(train_loader.dataset, val_loader.dataset if val_loader else None, num_classes, logger)

    # Здесь уже можно увидеть, "особенный" ли класс 0:
    logger.info("Class names (index -> name): %s", {i: n for i, n in enumerate(class_names)})

    # Веса классов по train-разметке
    ignore_index = getattr(train_loader.dataset, "ignore_index", 255)
    class_weights = compute_class_weights(train_loader, num_classes, ignore_index).cuda()

    base_model = build_model(cfg.model, class_names=class_names).cuda()
    configure_trainable_layers(base_model, args.unfreeze_depth)

    mixers: List[nn.Module] = []
    if args.use_embedding_mixer:
        channels = getattr(base_model.decode_head, "text_channels", 512)
        mixers.append(EmbeddingMixer(channels))

    wrapper = FineTuneWrapper(
        base_model, mixers=mixers, mix_strategy=args.mix_strategy
    ).cuda()

    log_parameter_counts(wrapper, logger)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, wrapper.parameters()),
        lr=args.learning_rate
    )

    best_checkpoints: List[Tuple[float, Path]] = []

    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        total_loss = 0.0
        loss_window = deque(maxlen=100)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for data in progress:
            imgs = data["img"].data[0].cuda()
            targets = data["gt_semantic_seg"].data[0].long().squeeze(1).cuda()

            optimizer.zero_grad()
            logits, _ = wrapper(imgs)
            loss = compute_loss(logits, targets, class_weights, ignore_index=ignore_index)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_window.append(loss_value)
            avg_recent = sum(loss_window) / len(loss_window)

            total_loss += loss_value * imgs.size(0)

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                avg100=f"{avg_recent:.4f}",
            )

        avg_loss = total_loss / len(train_loader.dataset)
        logger.info("Epoch %d avg loss: %.4f", epoch, avg_loss)

        metrics = {}
        if val_loader:
            metrics = evaluate(wrapper, val_loader)

            viz_dir = run_dir / "val_viz" / f"epoch_{epoch:03d}"
            save_val_visualizations(epoch, wrapper, val_loader, viz_dir)

        best_checkpoints = save_checkpoint(
            wrapper, optimizer, run_dir, epoch, metrics, avg_loss, best_checkpoints
        )

    logger.info("Training finished")


if __name__ == "__main__":
    main()
