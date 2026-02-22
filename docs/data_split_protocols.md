# Data split protocols (A/B) and deterministic preparation

`tools/split_dataset.py` prepares reproducible source/tile manifests and tiled datasets with anti-leakage rules.

## What it does

- Supports **Protocol A (time split)**:
  - years `2025` and `2026` are forced to `test`
  - other years are split into `train/val` by `--val-ratio` and `--seed`
  - split unit is `source_id` (prevents tile leakage between train/val)
- Supports **Protocol B (time + object split)**:
  - years `2025` and `2026` are forced to `test`
  - `test_facades` are excluded from train/val pools
  - train/val split is group-based by `facade_id`
- Generates tiles for all splits and applies augmentations **only** to `train/val` (currently: albumentations + zoom; no mixup/cutmix in `split_dataset.py`).
- Stores config + stats JSON with deterministic settings and `test_augmented_samples == 0` check.

## Input format

### Mode 1 (implemented)

`--data-root` must contain:

- `images/`
- `masks/`

Mask filename must match image filename (same relative path is preferred, fallback to same basename).

### Mode 2 (planned)

`--coco-json` is reserved for future COCO-index input and currently raises `NotImplementedError`.

## Metadata extraction from filename

For each source image filename, script extracts:

- `year`:
  1. suffix `_(20xx)` before extension
  2. `PXL_YYYYMMDD_...`
  3. `photo_*_YYYY-MM-DD_*`
  4. if starts with `IMG_` and no year found => `2025`
  5. otherwise fail (explicit error)
- `facade_id`:
  - remove hash prefix `^[0-9a-fA-F]{6,}-`
  - remove trailing `_{year}` if present
  - use remaining stem
- `source_id`:
  - stem without extension and hash prefix

## Tiling defaults

By default, tiling params are loaded from `configs/augmentation.yaml`:

- tile size: `448x448`
- stride: `224x224`
- pad mode: `constant`
- `min_content_ratio`: `0.6`

CLI can override with:

- `--tile-size`
- `--stride`
- `--pad-mode`
- `--min-content-ratio`

## Usage

Run both protocols:

```bash
python tools/split_dataset.py \
  --data-root /path/to/dataset_root \
  --out-root /path/to/data_prepared \
  --protocol A,B \
  --test-years 2025 2026 \
  --val-ratio 0.1 \
  --seed 42 \
  --augment \
  --augment-config configs/augmentation.yaml
```

Run only protocol A/B:

```bash
python tools/split_dataset.py --data-root /path/to/dataset_root --out-root data_prepared --protocol A
python tools/split_dataset.py --data-root /path/to/dataset_root --out-root data_prepared --protocol B
```

Debug modes:

- `--tile-only` => no augmentations at all
- `--no-augment` => disables train/val augmentations

## Output layout

```text
data_prepared/
  protocol_A/
    manifests/
      train_sources.csv
      val_sources.csv
      test_sources.csv
      train.csv
      val.csv
      test.csv
      split_config.json
      split_stats.json
    tiles/
      train/images, train/masks
      val/images, val/masks
      test/images, test/masks
  protocol_B/
    ...
```

`train.csv` / `val.csv` / `test.csv` columns:

- `rel_image_path`, `rel_mask_path`
- `year`
- `facade_id`
- `source_id`
- `tile_id`


### Tile naming

- `tile_id` format in manifests: `{source_id}_x{x}_y{y}_tile{tile_idx}` where `x,y` are left-top tile coordinates.
