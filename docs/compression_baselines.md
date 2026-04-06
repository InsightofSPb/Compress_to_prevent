# Compression baselines for facade monitoring

## Baseline inventory

| Baseline | Domain | Output semantics | score_type | Notes |
|---|---|---|---|---|
| zstd | residual byte-stream | realized bitstream length | achieved_bits | stream codec |
| lzma | residual byte-stream | realized bitstream length | achieved_bits | stream codec |
| webp | residual image-domain | realized bitstream length | achieved_bits | lossless WebP, requires Pillow WebP support |
| fnlic (lite) | residual image-domain predictive transform + entropy coding | realized bitstream length | achieved_bits | documented approximation |
| unigram LM | residual symbols | idealized coding length from NLL | model_bits | weak baseline kept for ablation |
| bigram LM | residual symbols | idealized coding length from NLL | model_bits | stronger default learned baseline |

## Stronger learned baseline

The default learned baseline is now **bigram** (`--model-mode bigram`), which conditions each residual symbol on one previous symbol. This is stronger than unigram and remains lightweight enough for quick facade experiments.

## Mini real-facade sanity workflow

Use a tiny real subset (e.g., 2-3 facades, 2-4 years each, total 6-12 images):

```bash
# 1) Build pairs
python cli/make_facade_pairs.py \
  --manifest-csv data/facades/sanity/manifest_mini.csv \
  --out-dir outputs/sanity_facade/pairs

# 2) Build residual dataset
python cli/build_facade_residual_dataset.py \
  --pairs-csv outputs/sanity_facade/pairs/pairs_all.csv \
  --out-root outputs/sanity_facade/residuals

# 3) Benchmark classical codecs
python cli/bench_facade_residual_codecs.py \
  --residual-manifest outputs/sanity_facade/residuals/residual_manifest.csv \
  --out-csv outputs/sanity_facade/codec_bench.csv \
  --methods zstd,lzma,webp,fnlic --level 3

# 4) Train + eval learned baseline
python cli/train_residual_entropy.py \
  --residual-manifest outputs/sanity_facade/residuals/residual_manifest.csv \
  --model-out outputs/sanity_facade/entropy_bigram.json \
  --model-mode bigram --train-split train

python cli/eval_residual_entropy.py \
  --residual-manifest outputs/sanity_facade/residuals/residual_manifest.csv \
  --model-path outputs/sanity_facade/entropy_bigram.json \
  --split val \
  --out-csv outputs/sanity_facade/entropy_eval.csv \
  --tile-size 32 \
  --tile-out-csv outputs/sanity_facade/entropy_tiles.csv

# 5) Change heatmaps
python cli/eval_facade_change_tiles.py \
  --residual-manifest outputs/sanity_facade/residuals/residual_manifest.csv \
  --out-scores-csv outputs/sanity_facade/change_tiles.csv \
  --heatmap-dir outputs/sanity_facade/heatmaps \
  --tile-size 32

# 6) Tile metrics (requires mini labels)
python cli/eval_facade_change_metrics.py \
  --tile-scores-csv outputs/sanity_facade/change_tiles.csv \
  --labels-csv data/facades/sanity/tile_labels_mini.csv \
  --out-csv outputs/sanity_facade/change_metrics.csv
```

## Output structure (sanity run)

- `outputs/sanity_facade/pairs/`
- `outputs/sanity_facade/residuals/`
- `outputs/sanity_facade/codec_bench.csv`
- `outputs/sanity_facade/entropy_eval.csv`
- `outputs/sanity_facade/entropy_tiles.csv`
- `outputs/sanity_facade/change_tiles.csv`
- `outputs/sanity_facade/heatmaps/<pair_id>.pgm` *(preview artifact)*
- `outputs/sanity_facade/change_metrics.csv`
