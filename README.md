# AI-Assisted Monitoring of Historical Facade Changes with Temporal Heatmaps

This repository contains the facade-monitoring experimental pipeline built on top of the original [LPOSS](https://arxiv.org/abs/2503.19777) open-vocabulary semantic segmentation codebase. The current study analyses temporally separated street-level RGB images of historical facades after geometric alignment. The primary signal is a compression-derived heatmap computed on aligned RGB residuals; semantic segmentation is used as an auxiliary interpretation branch and for reference-based quantitative evaluation.

The current experimental objective is **inspection-relevant facade state change localization**, not automatic diagnosis of physical degradation alone. Target changes may include deterioration, repairs, added or removed textual/pictorial content, and other visible interventions relevant to inspection.

## Repository roles

This repository provides:

- dataset preparation, facade-disjoint splitting, segmentation tiling and fine-tuning;
- aligned RGB pair manifests and valid-mask-aware RGB residual construction;
- aligned manual-reference maps for quantitative temporal heatmap evaluation;
- classical and perceptual temporal-change baselines;
- evaluation of heatmaps against inspection-relevant and damage/repair-oriented references.

The learned RGB residual compression model is trained in the companion repository [`InsightofSPb/MasksComp`](https://github.com/InsightofSPb/MasksComp), using the aligned pair manifests produced here.

Stock open-vocabulary MaskCLIP/LPOSS inference is provided by
`python -m ovs_heritage.infer_ovs`; see [the P1a inference guide](docs/stock_ovs_lposs_inference.md).
The historically named `tools/lposs_inference.py` and both `tools/finetune*.py` scripts are
legacy MaskCLIP-branch workflows, not full LPOSS execution.

## Current experimental protocol

### Data split

All splitting is performed by `facade_id`, so images from one facade cannot appear in both training and evaluation splits. Single non-temporal facade images are used only for segmentation training. Aligned temporal RGB pairs are assigned using the same facade split as the segmentation data.

The validation split is used for model/checkpoint selection and heatmap score threshold selection. The test split is reserved for final reporting.

### Temporal representation

For each temporal pair, the earlier RGB image is geometrically warped to the coordinate system of the later image. The RGB residual is then defined on valid aligned pixels as:

```text
residual = (current_RGB - warped_previous_RGB) mod 256
```

Pixels outside the valid alignment region are excluded from training and scoring. Full residual images may contain zero-filled invalid pixels only as storage/context placeholders; these pixels are not treated as valid change evidence.

### Reference maps

Manual semantic masks are aligned using the same homography as the RGB images, with nearest-neighbour interpolation. Reference maps encode:

```text
0   no target change
1   target change
255 invalid / ignored pixel
```

The principal reference is `inspection_relevant_change`, which retains changes involving:

- damage labels: `CRACK`, `SPALLING`, `DELAMINATION`, `MISSING_ELEMENT`, `WATER_STAIN`, `EFFLORESCENCE`, `CORROSION`;
- `REPAIRS`;
- `TEXT_OR_IMAGES`.

Secondary references include `damage_or_repair_change`, `damage_type_change`, `intervention_or_content_change`, `damage_presence_change`, and `any_semantic_change`.

The annotation ontology currently groups signage, graffiti and other visual textual/pictorial content into `TEXT_OR_IMAGES`. Consequently, appearance or disappearance of that category is represented, while within-category changes such as signage-to-graffiti cannot be distinguished.

### Heatmap evaluation

The current common evaluation grid is:

| Setting | Value |
|---|---:|
| Tile size | 32 × 32 |
| Tile stride | 32 |
| Minimum valid tile ratio for val/test scoring | 0.50 |
| Primary positive-tile criterion | changed valid pixel ratio ≥ 0.05 |

Primary ranking metrics are `AUPRC`, `AUROC`, `Precision@Top-K%`, and correlation with the continuous changed-pixel fraction. Thresholded `F1` and `IoU` are supplementary: their score threshold is selected on validation only and must be interpreted relative to the all-positive baseline.

## End-to-end workflow

The examples below use local variables rather than hard-coded machine paths:

```bash
DATA_ROOT=/path/to/data_26_05_all
SPLIT_ROOT=$DATA_ROOT/evaluation/group_split_v1
PAIRS_ROOT=$DATA_ROOT/compression/aligned_pairs_split_v1
REF_ROOT=$DATA_ROOT/evaluation/pair_change_references_v1
RES_ROOT=$DATA_ROOT/compression/rgb_residuals_valid_v1
BASELINE_ROOT=$DATA_ROOT/evaluation/baselines_valid_v1
```

### 1. Build aligned manual-reference maps

```bash
python tools/build_pair_change_reference_maps.py \
  --pairs-manifest $PAIRS_ROOT/pairs_all.csv \
  --masks-dir $DATA_ROOT/masks \
  --ref-spx-out $DATA_ROOT/facades_images_with_years/ref_spx_batch_out \
  --out-dir $REF_ROOT
```

The builder verifies that the recovered homography reproduces the saved RGB warp and writes reference manifests for train, validation and test pairs.

### 2. Build valid-mask-aware RGB residuals

```bash
python tools/build_rgb_residual_dataset.py \
  --pairs-manifest $PAIRS_ROOT/pairs_all.csv \
  --out-dir $RES_ROOT
```

### 3. Run classical aligned RGB baselines

```bash
python tools/run_temporal_change_baselines.py \
  --residual-manifest $RES_ROOT/residual_manifest.csv \
  --out-csv $BASELINE_ROOT/basic_tile_scores.csv \
  --methods absdiff_l1,absdiff_l2,grayscale_absdiff,ssim_change \
  --tile-size 32 \
  --min-valid-ratio 0.50 \
  --device cpu
```

The same CLI also supports `lpips_change` and `dinov2_patch_cosine` when their model dependencies are available.

### 4. Evaluate temporal tile scores

Primary inspection-relevant evaluation:

```bash
python tools/evaluate_temporal_tile_scores.py \
  --scores-csv $BASELINE_ROOT/basic_tile_scores.csv \
  --references-csv $REF_ROOT/pair_change_references.csv \
  --reference inspection_relevant_change \
  --target-min-change-ratio 0.05 \
  --top-k-percent 5,10,20 \
  --out-dir $BASELINE_ROOT/inspection_relevant_change_r005
```

Damage/repair-oriented supplementary evaluation:

```bash
python tools/evaluate_temporal_tile_scores.py \
  --scores-csv $BASELINE_ROOT/basic_tile_scores.csv \
  --references-csv $REF_ROOT/pair_change_references.csv \
  --reference damage_or_repair_change \
  --target-min-change-ratio 0.05 \
  --top-k-percent 5,10,20 \
  --out-dir $BASELINE_ROOT/damage_or_repair_change_r005
```

## Semantic segmentation branch

### Important architecture note

The current supervised fine-tuning implementation evaluates and adapts the **MaskCLIP-based segmentation branch initialised within the LPOSS codebase**. The training wrapper calls the MaskCLIP backbone and decode head directly; full LPOSS DINO-refinement is not invoked in this fine-tuning path. Accordingly, results should not be described as supervised fine-tuning of the complete LPOSS+DINO pipeline.

### Prepared tiled dataset

Training uses pre-generated image/mask tiles; augmentations are applied only to training tiles. `CutOut` writes ignored mask pixels with label `255`, and the segmentation loss/evaluation pipeline uses `ignore_index=255`. Validation and test images are tiled without augmentations and their overlapping logits are stitched before computing metrics.

### Fine-tuning with stitched validation

```bash
export FACADES_SEG_TRAIN_ROOT=$SPLIT_ROOT/segmentation_train_tiles
export FACADES_SEG_TRAIN_SPLIT=$SPLIT_ROOT/train_subsets/train_allclean_plus1aug.txt
export FACADES_SEG_VAL_TILES_ROOT=$SPLIT_ROOT/segmentation_eval_tiles

python tools/finetune_tiled.py lposs \
  --train-dataset-config mmseg/datasets/facades_group_split_train.py \
  --val-dataset-config mmseg/datasets/facades_group_split_val_tiles.py \
  --val-tiles-manifest $SPLIT_ROOT/segmentation_eval_tiles/val/tiles_manifest.csv \
  --train-probe-split $SPLIT_ROOT/train_subsets/clean_probe_512.txt \
  --train-probe-batch-size 32 \
  --train-probe-fixed-count 20 \
  --train-probe-top-k 20 \
  --train-probe-min-damage-ratio 0.01 \
  --epochs 10 \
  --batch-size 64 \
  --num-workers 2 \
  --learning-rate 2e-5 \
  --backbone-lr-mult 0.1 \
  --warmup-steps 300 \
  --unfreeze-depth 1 \
  --loss-mode focal_dice \
  --focal-gamma 2.0 \
  --dice-weight 1.0 \
  --class-weights auto \
  --class-weight-mode median_freq \
  --select-best-by DAMAGE_MACRO_MIOU \
  --log-online-train-metrics \
  --val-save-visualizations -1 \
  --output-root outputs/facade_segmentation_group_split_v1_depth1_damage
```

The principal checkpoint-selection metric is `DAMAGE_MACRO_MIOU`, the mean IoU across the seven annotated damage labels. Grouped metrics such as `STRUCTURAL_DAMAGE` remain useful for practical interpretation, especially under severe class imbalance.

### Stock versus fine-tuned evaluation

To quantify the effect of supervised adaptation, the stock pretrained branch and the validation-selected fine-tuned checkpoint must be evaluated with the same stitched tiled protocol.

Stock validation example:

```bash
python tools/evaluate_segmentation_tiled.py lposs \
  --eval-dataset-config mmseg/datasets/facades_group_split_val_tiles.py \
  --tiles-manifest $SPLIT_ROOT/segmentation_eval_tiles/val/tiles_manifest.csv \
  --split val \
  --model-label stock \
  --save-visualizations -1 \
  --output-root $DATA_ROOT/evaluation/segmentation_stock_vs_finetuned_v1
```

For a fine-tuned run, add:

```bash
--checkpoint /path/to/validation_selected_checkpoint.pth --model-label finetuned
```

Checkpoint selection is performed on validation only; test is evaluated once after selection.

## Learned RGB compression branch

The learned byte-level RGB residual model is implemented in the companion `MasksComp` repository. Use:

```text
tools/build_aligned_rgb_residual_tiles_valid.py
tools/train_aligned_rgb_temporal_lm_valid.py
tools/eval_aligned_rgb_temporal_lm_valid.py
```

The resulting tile-score CSVs are compatible with `tools/evaluate_temporal_tile_scores.py` in this repository.

## Upstream LPOSS attribution

This repository is based on:

> Vladan Stojnić, Yannis Kalantidis, Jiří Matas, Giorgos Tolias, **LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation**, CVPR 2025.

```bibtex
@InProceedings{stojnic2025_lposs,
    author    = {Stojni\'c, Vladan and Kalantidis, Yannis and Matas, Ji\v{r}i and Tolias, Giorgos},
    title     = {LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```

The original LPOSS setup and benchmark evaluation conventions remain applicable to the upstream open-vocabulary segmentation code; the facade temporal-monitoring pipeline documented above is project-specific.
