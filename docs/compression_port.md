# Facade compression/change toolkit port

This document describes the first integrated compression/change toolkit in `Compress_to_prevent`.

## What was ported

A minimal facade-first dependency closure was implemented in:

- `compression/` reusable logic
- `cli/` thin runnable entrypoints

The toolkit supports:

1. temporal facade pair construction
2. residual dataset building using `R_t = (I_t - I_{t-1->t}) mod 256`
3. classical residual codec benchmarking (Zstd, LZMA, WebP/FNLIC stubs)
4. byte-level learned entropy baseline train/eval/merge
5. tile-wise residual change heatmap generation
6. tile-level change metric evaluation

## CLIs and example commands

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
  --codecs zstd,lzma --level 3

python cli/train_residual_entropy.py \
  --residual-manifest data/facades/compression/residuals/residual_manifest.csv \
  --model-out outputs/compression/entropy_model.json \
  --train-split train

python cli/eval_residual_entropy.py \
  --residual-manifest data/facades/compression/residuals/residual_manifest.csv \
  --model-path outputs/compression/entropy_model.json \
  --split val \
  --out-csv outputs/compression/entropy_val.csv

python cli/merge_residual_entropy_scores.py \
  --inputs outputs/compression/entropy_val_r.csv outputs/compression/entropy_val_g.csv outputs/compression/entropy_val_b.csv \
  --out-csv outputs/compression/entropy_val_merged.csv

python cli/eval_facade_change_tiles.py \
  --residual-manifest data/facades/compression/residuals/residual_manifest.csv \
  --out-scores-csv outputs/compression/tile_scores.csv \
  --heatmap-dir outputs/compression/heatmaps \
  --tile-size 32

python cli/eval_facade_change_metrics.py \
  --tile-scores-csv outputs/compression/tile_scores.csv \
  --labels-csv data/facades/compression/tile_labels.csv \
  --out-csv outputs/compression/change_metrics.csv
```

## Adaptation assumptions

- Pair construction expects a facade image manifest with at least image paths and split; if `year` is missing, it is inferred from a `_YYYY` stem suffix.
- If `prev_aligned_path` is not provided yet, residual building falls back to `prev_image_path` and keeps the aligned-path field for later alignment pipeline integration.
- LM baseline is a lightweight byte-unigram entropy model intended as a first baseline.
- Tile labels are consumed from a simple CSV protocol: `pair_id,tile_x,tile_y,label`.

## Notes on next steps

- Semantic-conditioned compression (C1) is intentionally not part of this port.
- Temporal semantic branch redesign (S2) is intentionally not part of this port.
- Current toolkit is designed to be the compression/change baseline that C1 and S2 will build on.
