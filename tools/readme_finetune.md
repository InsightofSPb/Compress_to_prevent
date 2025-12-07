# Fine-tuning entrypoint (`tools/finetune.py`)

This script fine-tunes LPOSS/MaskCLIP models on a single GPU while keeping the
original evaluation flow. Key behaviors:

- **Layer control:** freeze backbones except for the last `--unfreeze-depth`
  transformer blocks and always train the decode head.
- **Embedding mixers:** optional lightweight residual mixers can modify the
  dense embeddings before segmentation logits are produced; more complex heads
  can be swapped in later without changing the loop.
- **Checkpoints:** each run creates a timestamped output directory under
  `--output-root`, saves `train_settings.json`, and keeps only the three best
  checkpoints ranked by average training loss.

## Required configs

- `--config`: Hydra config name from the `configs/` directory (e.g.
  `lposs.yaml`). This defines the base model/optimizer setup.
- `--train-dataset-config`: MMCV config that describes the training dataset and
  dataloader settings. It should expose a `data.train` section compatible with
  `mmseg.datasets.build_dataset`. A minimal example:

  ```python
  data = dict(
      samples_per_gpu=2,
      workers_per_gpu=2,
      train=dict(
          type="CustomDataset",
          data_root="/path/to/data/",
          img_dir="images/train",
          ann_dir="annotations/train",
          pipeline=[...],
      ),
  )
  ```

- `--val-dataset-config` (optional): same format as the training config, but
  providing a `data.val` section. If provided, mIoU is computed after every
  epoch using the dataset's `evaluate` implementation.

## Running

Example command (with validation):

```bash
python tools/finetune.py lposs.yaml \
  --train-dataset-config datasets/my_train_config.py \
  --val-dataset-config datasets/my_val_config.py \
  --epochs 20 --batch-size 2 --learning-rate 1e-4 \
  --unfreeze-depth 2 --use-embedding-mixer
```

Outputs land in `outputs/finetune_<timestamp>_lr...`, containing checkpoints,
logs, and `train_settings.json` for reproducibility.