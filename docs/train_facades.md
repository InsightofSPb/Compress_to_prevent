# Facade fine-tuning quick recipes

## 1) Enable offline zoom augmentation on 448×448 tiles

In `configs/augmentation.yaml`:

```yaml
augmentations:
  zoom:
    enabled: true
    p: 0.35
    scale_min: 0.70
    scale_max: 1.00
```

Optional damage-aware centering:

```yaml
augmentations:
  zoom:
    damage_center_prob: 0.6
```

Then regenerate tiles:

```bash
python tools/augment_dataset.py -c configs/augmentation.yaml
```

## 2) Fully unfreeze MaskCLIP safely

`--unfreeze-depth -1` unfreezes all ViT blocks, `0` keeps backbone frozen, `k` unfreezes last `k` blocks.

Example:

```bash
python tools/finetune.py lposs.yaml \
  --train-dataset-config segmentation/configs/_base_/datasets/facades_train.py \
  --val-dataset-config segmentation/configs/_base_/datasets/facades_val.py \
  --unfreeze-depth -1 --learning-rate 2e-5 --backbone-lr-mult 0.1 --warmup-steps 1000
```

## 3) Weighted loss for class imbalance

### Weighted CE baseline

```bash
python tools/finetune.py lposs.yaml ... \
  --loss-mode ce_weighted --class-weights auto --class-weight-mode median_freq
```

### Focal + Dice

```bash
python tools/finetune.py lposs.yaml ... \
  --loss-mode focal_dice --focal-gamma 2.0 --dice-weight 1.0 --class-weights auto
```

You can also pass manual weights:

```bash
--class-weights 1.0,3.0,4.5,2.0
```
