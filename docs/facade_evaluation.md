# Facade quantitative evaluation

Install evaluation dependencies:

```bash
pip install -r requirements-eval.txt
```

## Inputs

- `residual_manifest.csv` with fields: `pair_id,facade_id,split,residual_path,prev_aligned_path,curr_image_path,height,width`
- labels CSV: `pair_id,tile_x,tile_y,label` where `label` is 0/1.

## Full paper suite

```bash
python cli/run_facade_evaluation_suite.py \
  --residual-manifest outputs/sanity_facade/residuals/residual_manifest.csv \
  --labels-csv data/facades/compression/tile_labels.csv \
  --out-dir outputs/evaluation_facade_full \
  --tile-size 32 \
  --device cuda \
  --feature-cache-dir outputs/evaluation_facade_full/feature_cache \
  --dinov2-model-name dinov2_vitb14 \
  --lpips-net alex
```

This computes: `proposed_residual, absdiff_l1, absdiff_l2, grayscale_absdiff, ssim_change, dinov2_patch_cosine, lpips_change`.

## Lightweight debug run

```bash
python cli/run_facade_evaluation_suite.py \
  --residual-manifest outputs/sanity_facade/residuals/residual_manifest.csv \
  --labels-csv data/facades/compression/tile_labels.csv \
  --out-dir outputs/evaluation_facade_light \
  --baseline-methods absdiff_l1,absdiff_l2,grayscale_absdiff,ssim_change \
  --skip-deep-baselines
```

## DINOv2 and LPIPS notes

- DINOv2 baseline (`dinov2_patch_cosine`) uses `torch.hub` loading (`facebookresearch/dinov2`) and supports `--dinov2-cache-dir` and `--dinov2-weights-path`.
- LPIPS baseline (`lpips_change`) uses `lpips` package (`--lpips-net alex` by default).
- Full mode fails with actionable errors if missing dependencies/models.

## Metrics

`summary_metrics.csv` includes:
`method,score_type,n_pairs,n_tiles,n_pos,n_neg,roc_auc,average_precision,precision_at_1pct,precision_at_5pct,precision_at_10pct,recall_at_5pct,recall_at_10pct,best_f1,best_f1_threshold,topk_hit_rate_5,topk_hit_rate_10`.

Interpretation:
- ROC-AUC/AP: ranking quality globally.
- Precision/recall at top-X%: practical alert budget behavior.
- top-k hit rate: pair-level localization success.

## Limitations (for paper reporting)

- Tile labels are proxy annotations.
- Heatmaps localize visual novelty, not direct proof of material degradation.
- Robustness to illumination, shadows, seasonal effects, occlusions, and viewpoint shifts is improved by baselines but not fully solved.
- Feature baselines are included for standard comparison/ablation, not to claim universal superiority.

## Manual deep-baseline check

```bash
python cli/eval_facade_baseline_tiles.py \
  --residual-manifest outputs/sanity_facade/residuals/residual_manifest.csv \
  --methods absdiff_l1,absdiff_l2,grayscale_absdiff,ssim_change,dinov2_patch_cosine,lpips_change \
  --out-scores-csv outputs/sanity_facade/baseline_tiles_full.csv \
  --tile-size 32 --device cuda
```
