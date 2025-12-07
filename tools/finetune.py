"""Single-GPU fine-tuning script for LPOSS/MaskCLIP models.

This script keeps the original evaluation metrics while providing:
- configurable layer unfreezing depth for the backbones
- placeholder embedding mixers to plug extra heads later on
- run-directory creation with timestamped names and top-3 checkpoint pruning
"""
import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize
from mmseg.datasets import build_dataloader, build_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.logger import get_logger
from models import build_model


class IdentityHead(nn.Module):
    """Placeholder head that leaves embeddings unchanged."""

    def forward(self, feats: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401
        return feats


class EmbeddingMixer(nn.Module):
    """Lightweight residual block to adjust dense embeddings.

    This is intentionally simple so that future custom heads can be swapped in
    without touching the training loop.
    """

    def __init__(self, channels: int, hidden_channels: int = 256):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, feats: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401
        mixed = self.proj_out(self.act(self.proj_in(feats)))
        return feats + mixed


class FineTuneWrapper(nn.Module):
    """Wrap the base model to allow embedding mixers before classification."""

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
        logits, feats = self.base_model(x, return_feat=True)
        mixed_feats = feats
        for head in self.mixers:
            mixed_feats = head(mixed_feats, logits)

        mixed_logits = self.base_model.decode_head.cls_seg(mixed_feats)

        if self.mix_strategy == "replace":
            combined = mixed_logits
        elif self.mix_strategy == "concat":
            combined = torch.stack([logits, mixed_logits], dim=0).mean(dim=0)
        else:  # default to averaging
            combined = (logits + mixed_logits) / 2

        return combined, {"base_logits": logits, "mixed_logits": mixed_logits}


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
    """Freeze all backbone layers except the last ``depth`` transformer blocks."""

    if hasattr(model, "clip_backbone"):
        backbone = getattr(model.clip_backbone, "backbone", None)
        if backbone is not None and hasattr(backbone.visual.transformer, "resblocks"):
            _unfreeze_last(backbone.visual.transformer.resblocks, depth)
    if hasattr(model, "dino_encoder") and hasattr(model.dino_encoder, "blocks"):
        _unfreeze_last(model.dino_encoder.blocks, depth)

    if hasattr(model, "decode_head"):
        for param in model.decode_head.parameters():
            param.requires_grad = True


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
        val_dataset = build_dataset(val_cfg_obj.data.val, dict(test_mode=True))
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


def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    log_probs = torch.log(logits.clamp(min=1e-8))
    return F.nll_loss(log_probs, target, ignore_index=255)


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> dict:
    model.eval()
    results = []
    for data in loader:
        imgs = data["img"].data[0].cuda()
        logits, _ = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        results.extend(list(preds))
    metrics = loader.dataset.evaluate(results, metric="mIoU", logger=get_logger())
    return metrics


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, out_dir: Path, epoch: int, metrics: dict, average_loss: float, best_pool: list):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU fine-tuning entrypoint")
    parser.add_argument("config", help="Hydra config name (e.g., lposs.yaml)")
    parser.add_argument("--train-dataset-config", required=True, help="MMCV config with training data definition")
    parser.add_argument("--val-dataset-config", help="Optional MMCV config for validation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--unfreeze-depth", type=int, default=2, help="How many final transformer blocks to unfreeze")
    parser.add_argument("--mix-strategy", choices=["add", "concat", "replace"], default="add")
    parser.add_argument("--use-embedding-mixer", action="store_true", help="Enable placeholder embedding mixers")
    parser.add_argument("--output-root", default="outputs", help="Root folder for run artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"finetune_{timestamp}_lr{args.learning_rate}_depth{args.unfreeze_depth}_bs{args.batch_size}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.output = str(run_dir)
    logger = get_logger(cfg)
    logger.info("Starting fine-tuning with config: %s", args.config)

    train_loader, val_loader = build_dataloaders(args.train_dataset_config, args.batch_size, args.num_workers, args.val_dataset_config)
    class_names = train_loader.dataset.CLASSES

    base_model = build_model(cfg.model, class_names=class_names)
    base_model.cuda()
    base_model.device = "cuda"

    configure_trainable_layers(base_model, args.unfreeze_depth)

    mixers: List[nn.Module] = []
    if args.use_embedding_mixer:
        channels = getattr(base_model.decode_head, "text_channels", 512)
        mixers.append(EmbeddingMixer(channels))
    wrapper = FineTuneWrapper(base_model, mixers=mixers, mix_strategy=args.mix_strategy).cuda()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, wrapper.parameters()), lr=args.learning_rate)

    with open(run_dir / "train_settings.json", "w") as f:
        json.dump({
            "config": args.config,
            "train_dataset_config": args.train_dataset_config,
            "val_dataset_config": args.val_dataset_config,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "unfreeze_depth": args.unfreeze_depth,
            "mix_strategy": args.mix_strategy,
            "use_embedding_mixer": args.use_embedding_mixer,
        }, f, indent=2)

    best_checkpoints: List[Tuple[float, Path]] = []

    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        total_loss = 0.0
        for step, data in enumerate(train_loader, start=1):
            imgs = data["img"].data[0].cuda()
            targets = data["gt_semantic_seg"].data[0].long().squeeze(1).cuda()

            optimizer.zero_grad()
            logits, _ = wrapper(imgs)
            loss = compute_loss(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            if step % 10 == 0:
                logger.info(
                    "Epoch %d | Step %d/%d | Loss: %.4f", epoch, step, len(train_loader), loss.item()
                )

        avg_loss = total_loss / len(train_loader.dataset)
        logger.info("Epoch %d completed. Average loss: %.4f", epoch, avg_loss)

        metrics = {}
        if val_loader is not None:
            metrics = evaluate(wrapper, val_loader)
            logger.info("Validation metrics at epoch %d: %s", epoch, metrics)

        best_checkpoints = save_checkpoint(wrapper, optimizer, run_dir, epoch, metrics, avg_loss, best_checkpoints)

    logger.info("Training finished. Top checkpoints:")
    for score, path in best_checkpoints:
        logger.info("Loss %.4f -> %s", score, path)


if __name__ == "__main__":
    main()
