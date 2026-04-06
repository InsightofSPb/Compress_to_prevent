# Facade compression/change toolkit port

## Status after scaffold upgrade

The initial scaffold from PR #48 has been upgraded for paper-usable baseline work with:

- stronger learned entropy baseline option (`bigram`) in addition to `unigram`
- explicit achieved-vs-model bit semantics in benchmark/eval CSVs (`score_type`, `bit_length`)
- real WebP path when Pillow WebP support is available
- FNLIC-inspired minimal path (`fnlic_lite_predictive+zstd`, with fallback to LZMA if zstd is unavailable)
- mini real-facade sanity workflow recipe

## Core CLIs

```bash
python cli/make_facade_pairs.py \
  --manifest-csv data/facades/manifest_images.csv \
  --out-dir data/facades/compression/pairs

python cli/build_facade_residual_dataset.py \
  --pairs-csv data/facades/compression/pairs/pairs_all.csv \
  --out-root data/facades/compression/residuals

python cli/bench_facade_residual_codecs.py \
  --residual-manifest data/facades/compression/residuals/residual_manifest.csv \
  --out-csv outputs/compression/codec_bench.csv \
  --methods zstd,lzma,webp,fnlic --level 3

python cli/train_residual_entropy.py \
  --residual-manifest data/facades/compression/residuals/residual_manifest.csv \
  --model-out outputs/compression/entropy_bigram.json \
  --model-mode bigram --train-split train

python cli/eval_residual_entropy.py \
  --residual-manifest data/facades/compression/residuals/residual_manifest.csv \
  --model-path outputs/compression/entropy_bigram.json \
  --split val \
  --out-csv outputs/compression/entropy_val.csv \
  --tile-size 32 \
  --tile-out-csv outputs/compression/entropy_val_tiles.csv

python cli/eval_facade_change_tiles.py \
  --residual-manifest data/facades/compression/residuals/residual_manifest.csv \
  --out-scores-csv outputs/compression/change_tiles.csv \
  --heatmap-dir outputs/compression/heatmaps \
  --tile-size 32

python cli/eval_facade_change_metrics.py \
  --tile-scores-csv outputs/compression/change_tiles.csv \
  --labels-csv data/facades/compression/tile_labels.csv \
  --out-csv outputs/compression/change_metrics.csv
```

## Achieved vs model bits

- Classical codecs (`zstd`, `lzma`, `webp`, `fnlic`) emit `score_type=achieved_bits` with realized coded `bit_length`.
- Learned entropy evaluations emit `score_type=model_bits` with `bit_length` computed from NLL.
- CSV schemas intentionally avoid ambiguous generic `bits` fields.

## Mini real-facade sanity workflow

Use `configs/compression/sanity_facade_mini.yaml` as the recipe root. Expected tiny manifest columns:

- `facade_id`
- `year`
- `image_path`
- `aligned_image_path` (or leave empty; fallback to previous image)
- `split`

For an example command sequence, see `docs/compression_baselines.md`.

## Known implementation limits

- WebP requires Pillow with WebP codec support; if unavailable, benchmark rows are marked `status=unsupported` with a diagnostic note.
- FNLIC path is a documented minimal approximation (`fnlic_lite_predictive+zstd/lzma`) and not a full external FNLIC runtime.

## Non-goals retained

- No semantic-conditioned compressor (C1) in this stage.
- No temporal semantic branch redesign (S2) in this stage.
