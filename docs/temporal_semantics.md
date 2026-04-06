# Temporal semantics (S2) platform

S2 is implemented as a **multi-backend platform**, not a single model.

## Backend platform model

Each backend declares explicit capabilities via the registry:
- `backend`, `model_id`
- `provides_dense_features`
- `provides_global_features`
- `provides_masks`
- `provides_logits_or_probs`
- `tile_compatible`
- `expected_feature_grid_type`
- `experimental`
- `notes`

List backends and capabilities:

```bash
python cli/export_temporal_semantic_artifacts.py --list-backends
```

## Backend status

### Fully working (tile-compatible)
- `lposs`
- `dinov2`
- `clip` (OpenCLIP-style proxy backend)
- `siglip2`

### Experimental / partial
- `florence2` scaffold (`experimental=true`, `tile_compatible=false`)

## Artifact export

```bash
python cli/export_temporal_semantic_artifacts.py \
  --manifest-csv data/facades/sanity/manifest_mini.csv \
  --out-dir outputs/temporal_semantics \
  --backends lposs,dinov2,clip,siglip2 \
  --tile-size 32
```

Main index:
- `outputs/temporal_semantics/artifact_index.csv`

Index fields include:
- sample/backend identifiers and paths
- feature grid shape
- image resolution
- status
- capability snapshot (`capabilities_json`)

## Temporal pair tile features

```bash
python cli/build_temporal_semantic_features.py \
  --pairs-csv data/facades/compression/pairs/pairs_all.csv \
  --artifact-index-csv outputs/temporal_semantics/artifact_index.csv \
  --out-csv outputs/temporal_semantics/pair_tile_features.csv \
  --backends lposs,dinov2,clip,siglip2 \
  --tile-size 32
```

Per-tile features include:
- LPOSS-style semantics: `mask_change_density`, `class_histogram_drift`, `prob_entropy_change`
- feature deltas: `feature_cosine_distance`, `feature_l2_distance`, `feature_norm_delta`
- cross-backend stats: `backend_agreement_score`, `backend_disagreement_score`
- scores: `semantic_score_backend`, `semantic_score_fused`

Tile convention follows the compression branch (`tile_size`, stride=`tile_size`).

## Evaluation and comparison

```bash
python cli/eval_temporal_semantic_features.py \
  --features-csv outputs/temporal_semantics/pair_tile_features.csv \
  --out-summary-csv outputs/temporal_semantics/summary.csv \
  --out-topk-csv outputs/temporal_semantics/topk_tiles.csv
```

Outputs:
- backend comparison summary (`summary.csv`)
- top-K suspicious tiles (`topk_tiles.csv`)
- backend top-K overlap (`summary_overlap.csv`)

## Heatmaps and previews

```bash
python cli/render_temporal_semantic_previews.py \
  --features-csv outputs/temporal_semantics/pair_tile_features.csv \
  --pairs-csv data/facades/compression/pairs/pairs_all.csv \
  --out-dir outputs/temporal_semantics/previews \
  --tile-size 32 \
  --include-fused
```

Outputs:
- per-backend heatmaps + overlays
- fused heatmaps + overlays

## Mini real-data recipe

Use a tiny facade subset manifest (`data/facades/sanity/manifest_mini.csv`) and the command chain above.
No real data is committed in-repo; only templates/configs are provided.

## Path to C1

S2 outputs can later be joined with compression outputs by:
- `pair_id`
- tile geometry (`tile_id`, `x0,y0,x1,y1`)

This keeps S2/C1 integration straightforward without changing existing compression schemas.
