# C1 semantic-conditioned residual codec

C1 is the first semantic-conditioned compression model in this repository. It models residual tiles as:

`p(R_tile | semantic_context_tile)`

where `R_tile` comes from the existing residual dataset pipeline and semantic context is reused from S2 exports (artifact index + temporal semantic features).

## What C1 uses from S2

C1 context assembly (`compression/conditioned/context.py`) reuses:
- LPOSS mask statistics (`lposs_mask_stats`)
- LPOSS probability summaries (`lposs_probs`)
- DINOv2 tile features (`dinov2_features`)
- CLIP tile features (`clip_features`)
- SigLIP2 tile features (`siglip2_features`)
- temporal semantic tile features table (`semantic_temporal_features`)
- fused semantic score scalar (`semantic_fused_score`)

No duplicated S2 backend logic is implemented inside C1.

## Context modes

Supported context modes:
- `none`
- `lposs_only`
- `features_only`
- `temporal_semantic_only`
- `full`
- `custom` (comma-separated source list)

## Conditioning mechanisms

C1 currently supports:
- `concat_context`
- `film_context` (FiLM-lite via context-feature interactions)

Both mechanisms share a common linear-Gaussian residual density model with pseudo-autoregressive dependence on previous residual byte.

## Score semantics

C1 outputs **model-estimated** coding quantities only:
- `model_bits`
- `nll_bits`
- `bits_per_byte`

These are not achieved arithmetic-coded bits. Every table includes explicit `method`, `score_type`, `context_mode`, and `conditioning_mechanism`.

## CLI examples

Train:

```bash
python cli/train_semantic_conditioned_codec.py \
  --residual-manifest data/facades/compression/residuals/residual_manifest.csv \
  --pairs-csv data/facades/compression/pairs/pairs_all.csv \
  --artifact-index-csv outputs/temporal_semantics/artifact_index.csv \
  --temporal-features-csv outputs/temporal_semantics/pair_tile_features.csv \
  --context-mode full \
  --conditioning-mechanism concat_context \
  --model-out outputs/compression/c1_full_concat.json
```

Evaluate:

```bash
python cli/eval_semantic_conditioned_codec.py \
  --residual-manifest data/facades/compression/residuals/residual_manifest.csv \
  --pairs-csv data/facades/compression/pairs/pairs_all.csv \
  --artifact-index-csv outputs/temporal_semantics/artifact_index.csv \
  --temporal-features-csv outputs/temporal_semantics/pair_tile_features.csv \
  --model-path outputs/compression/c1_full_concat.json \
  --split val \
  --context-mode full \
  --conditioning-mechanism concat_context \
  --out-tile-csv outputs/compression/c1_val_tiles.csv \
  --out-pair-csv outputs/compression/c1_val_pairs.csv
```

Preview rendering:

```bash
python cli/render_semantic_conditioned_codec_previews.py \
  --pairs-csv data/facades/compression/pairs/pairs_all.csv \
  --conditioned-tile-csv outputs/compression/c1_val_tiles.csv \
  --unconditioned-tile-csv outputs/compression/c1_none_val_tiles.csv \
  --semantic-features-csv outputs/temporal_semantics/pair_tile_features.csv \
  --out-dir outputs/compression/c1_previews
```

## What remains before full semantic+compression fusion

- true bitstream arithmetic coding for C1 likelihoods
- stronger sequence model families (e.g., transformers)
- calibrated risk fusion that combines S2 and C1 into final facade-level decisions
