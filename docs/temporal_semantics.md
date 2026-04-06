# Temporal semantics (S2) pipeline

This document describes the S2 temporal semantic branch implemented in this repository.

## Goals

S2 exports semantic artifacts per image, then computes per-tile temporal semantic features for aligned facade pairs, compares multiple backends, and produces fused semantic heatmaps compatible with the compression tile protocol.

## Implemented backends

### Fully working
- `lposs` (project semantic branch proxy artifacts: mask/probabilities/features/overlay)
- `dinov2` (standalone dense feature proxy backend)
- `clip` (additional standalone dense feature proxy backend)

### Experimental
- `florence2` scaffold only (`status=experimental`, no dense extraction yet)

## Artifact export workflow

```bash
python cli/export_temporal_semantic_artifacts.py \
  --manifest-csv data/facades/sanity/manifest_mini.csv \
  --out-dir outputs/temporal_semantics \
  --backends lposs,dinov2,clip \
  --tile-size 32
```

Main output index:
- `outputs/temporal_semantics/artifact_index.csv`

Index columns:
- `sample_id, backend, image_path, mask_path, probs_path, features_path, overlay_path, feature_grid_h, feature_grid_w, split, status, notes`

## Pair feature workflow

```bash
python cli/build_temporal_semantic_features.py \
  --pairs-csv data/facades/compression/pairs/pairs_all.csv \
  --artifact-index-csv outputs/temporal_semantics/artifact_index.csv \
  --out-csv outputs/temporal_semantics/pair_tile_features.csv \
  --backends lposs,dinov2,clip \
  --tile-size 32
```

Per-tile fields include:
- tile coordinates and geometry (`x0,y0,x1,y1,center_x,center_y`)
- LPOSS metrics: `mask_change_density`, `class_histogram_drift`, `prob_entropy_change`
- feature metrics: `feature_cosine_distance`, `feature_l2_distance`
- cross-backend: `backend_agreement_score`
- aggregate scores: `semantic_score_backend`, `semantic_score_fused`

## Evaluation

```bash
python cli/eval_temporal_semantic_features.py \
  --features-csv outputs/temporal_semantics/pair_tile_features.csv \
  --out-summary-csv outputs/temporal_semantics/summary.csv \
  --out-topk-csv outputs/temporal_semantics/topk_tiles.csv
```

This exports backend summary statistics and top-K suspicious tiles per pair.

## Previews and heatmaps

```bash
python cli/render_temporal_semantic_previews.py \
  --features-csv outputs/temporal_semantics/pair_tile_features.csv \
  --pairs-csv data/facades/compression/pairs/pairs_all.csv \
  --out-dir outputs/temporal_semantics/previews \
  --tile-size 32
```

Outputs:
- per-backend semantic heatmaps (`.pgm`)
- per-backend overlay previews (`.ppm`)

These use the same tile stepping convention as the compression branch (`tile_size`, stride = `tile_size`).

## Joining S2 with compression branch for future C1

S2 tile table and compression tile tables can be joined directly by:
- `pair_id`
- tile geometry (`tile_id` or `x0,y0,x1,y1`)

This preserves compatibility for future C1 fusion without changing the existing compression branch outputs.
