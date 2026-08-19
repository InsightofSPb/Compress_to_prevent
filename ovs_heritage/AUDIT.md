# P0 audit of the current working tree

This document records code observations, not assumptions. Legacy results and paths are not changed.

## 1. Actual fine-tuning path

**Confirmed:** `tools/finetune.py:FineTuneWrapper.forward` selects `base_model.clip_backbone` when present, calls that MaskCLIP object with `return_feat=True`, then directly calls `decode_head.cls_seg`. It does not call `LPOSS.forward`. `tools/finetune_tiled.py` imports this wrapper as `common.FineTuneWrapper` and follows the same path. Consequently this is MaskCLIP-branch fine-tuning, not full-LPOSS fine-tuning. DINO graph refinement is absent from training and from `evaluate_stitched`; the latter stitches wrapper logits. The full LPOSS path exists separately in `models/lposs/lposs.py:LPOSS.forward`.

`configure_trainable_layers` first freezes the whole model, unfreezes the requested suffix (`depth=-1` means all) of CLIP visual transformer blocks, and unfreezes the entire decode head. This includes `decode_head.proj`; the text prototypes are buffers, not parameters. Mixers are separate trainable modules added after this configuration.

## 2. Imbalance versus catastrophic forgetting

Class imbalance changes the relative number/gradient contribution of supervised examples. Catastrophic forgetting changes foundation feature geometry: enabled visual blocks and the image projection (`decode_head.proj`) are optimized for the closed heritage labels, while text embeddings remain fixed. That can damage text/image alignment for unseen concepts. Class weighting or oversampling can rebalance heritage gradients, but does not constrain preservation of the original CLIP geometry and therefore cannot guarantee open-vocabulary retention.

## 3. Softmax before cross entropy

**Confirmed mathematical defect in the legacy training path.** `models/maskclip/maskclip.py:MaskClipHead.cls_seg` computes cosine convolution, multiplies by 100, then returns `F.softmax(...)`. `FineTuneWrapper` averages these probabilities and its direct second `cls_seg` result; `tools/finetune.py:compute_loss` passes that tensor to `F.cross_entropy`, which expects raw logits. Full `models/lposs/lposs.py:LPOSS.forward` does its own graph label-propagation scorer from normalized CLIP/DINO features rather than merely consuming the MaskCLIP probability map. P0 provides a new isolated raw scorer/loss; this does not retroactively repair checkpoints or historical measurements.

## P1a stock inference correction

`tools/lposs_inference.py` remains runnable but is now explicitly labelled as a **legacy
MaskCLIP-only** path. `tools/finetune.py` and `tools/finetune_tiled.py` likewise train only
that branch. Historical improvements including mIoU 0.0551→0.1676 therefore provide no
evidence for DINO propagation, LPOSS+, or retention of open-vocabulary behavior.

The P1a entry point is `python -m ovs_heritage.infer_ovs`. Its mode ledger fields distinguish
raw MaskCLIP cosine seeds, genuine patch propagation, and pixel refinement. Graph modes fail
closed during preflight; they cannot relabel raw predictions as LPOSS output.

## 4. Vocabulary-specific state

`MaskClipHead.__init__` uses `register_buffer("class_embeddings", ...)`, so prototypes are persistent in `state_dict` and checkpoint shape/order depends on aliases and vocabulary. `class_mapping` is a plain tensor attribute, not a registered buffer, and is not saved. `class_names` is a plain Python value. `proj.weight` is persistent vocabulary-independent image projection state. P0 `PrototypeSet` is returned at runtime and `RawCosineScorer.state_dict()` is empty.

## 5. Aliases and update_vocab

Semicolon-separated labels are expanded in `_get_class_embeddings`: every alias produces another prototype and hence another `cls_seg` output channel. `class_mapping` is created but never used by `cls_seg`; `reduce_to_true_classes` in the LPOSS inferencer only collapses extra leading background expansion and is not a general alias reduction. `update_vocab` replaces embeddings but does not update `self.class_names`, does not move the newly created text model to CUDA, and does not explicitly preserve the caller device. `_embed_label` hard-codes prompts to `cuda`; constructor hard-codes `model.cuda()`. These choices permit CPU failure and device mismatch. List iteration preserves input order before alias expansion. P0 aggregates all prompt variants into exactly one prototype in exact runtime order.

## 6. ADVERTISEMENTS loss risk

**Confirmed.** `tools/finetune.py:_sanitize_targets` replaces every target outside `[0, num_classes)` with 255. With the legacy eleven-channel datasets, ID 11 is therefore silently ignored. P0 `validate_mask_ids`, dataset validation, and `supervised_cross_entropy` raise and enumerate invalid IDs; they never rewrite labels.

## 7. Consumers of the old ontology

| file | symbol/location | current assumption | evidence | required action | status |
|---|---|---|---|---|---|
| `tools/convert_brush_coco_to_masks.py` | `LABELS` | IDs 0..10; no advertisements | eleven literal entries | retain as historical converter; use a reviewed v2 conversion path later | legacy intentionally preserved |
| `tools/convert_brush_coco_to_masks.py` | annotation loop | overlap is input-order “last annotation wins” | unconditional `mask[ann_mask > 0] = label_id` | document/resolve overlap policy before any migration | deferred to P1/P2 |
| `mmseg/datasets/facades_train.py` and sibling facade datasets | `classes`, `palette` | eleven classes/channels | eleven literals | keep v1 meaning; new P0 adapter reads v2 source | legacy intentionally preserved |
| `segmentation/configs/_base_/datasets/facades_test.py` | `classes`, `palette` | eleven classes | eleven literals | do not use with masks containing ID 11 | legacy intentionally preserved |
| `tools/finetune.py` | metric groups | HUMAN_ACTIVITY omits advertisements | two-name set | consume v2 groups in future trainer | deferred to P1/P2 |
| `tools/compare_models_facades.py` | groups | eleven-class evaluation | local sets | re-evaluate both models on common v2 test set | deferred to P1/P2 |
| `tools/render_temporal_qualitative_grids.py` | defaults | eleven names/colors | literal lists | legacy figures remain reproducible | legacy intentionally preserved |
| `models/maskclip/maskclip.py` | head outputs | channel count follows expanded strings | embedding convolution | legacy file is unchanged; separate P0 scorer supports runtime C | legacy intentionally preserved |
| `ovs_heritage/configs/datasets/heritage_facades_v2.py` | adapter exports | twelve concepts projected to 11 main channels plus one ornament channel | values loaded from canonical source and projection | use only with explicit two-map v2 manifests | changed |
| README temporal semantics | ontology prose | text/signage combined | explicitly says combined class | update only when downstream temporal contract migrates | legacy intentionally preserved |

The legacy/base code audited before P0 had no tracked `ADVERTISEMENTS` class in its converter or eleven-class dataset configurations. This PR now contains the canonical P0 `ADVERTISEMENTS` concept at semantic ID 11 with visualization RGB `(216, 27, 96)`, while the legacy consumers listed above remain unchanged. No annotation pixels were created, moved, or converted.

## 8. LPOSS inference

**Confirmed syntax/semantic defect:** `segmentation/evaluation/lposs_eval.py:LPOSS_Infrencer.forward` has a duplicated conditional expression immediately after `else i`. Python parses this as attempting to call `i` (often a Tensor) with the following parenthesized result. P0.1 removed the earlier AST test because it passed only while the bug remained and therefore encoded the defect as expected behavior. The safe legacy fix and a regression test that exercises successful inference belong with the P1 LPOSS wrapper/integration work. A later wrapper must distinguish DINO graph refinement (feature graph propagation in LPOSS), LPOSS+ pixel refinement (`pixel_refine`, CuPy Laplacian), and fallback: without CuPy pixel refinement is explicitly skipped; FAISS/CUDA availability affects graph implementations and is not equivalent to LPOSS+.

`LPOSS_Infrencer.encode_decode` currently calls `self.model(img)` and does not reference an undefined `x`; the earlier audit statement claiming otherwise was incorrect and has been removed. The duplicated conditional-expression defect above remains separately verified. Any LPOSS fix and positive integration regression test remain outside P0.

## 9. Historical metrics

Values such as mIoU 0.0551→0.1676 or DAMAGE_MACRO_MIOU 0.0209→0.0802, wherever retained as experiment references, are not P0 results. Legacy single-mask metrics and v2 two-map metrics are not directly comparable. Stock and adapted models must be evaluated again on the identical 12-concept two-map test set. Future reports must distinguish `stock_repo_exact` from `stock_shared_scorer` (stock dense features with the P0 scorer).

## 10. P0 two-map correction

The current P0 representation supersedes the earlier single-raster v2 assumption.
Canonical concept 8 is now `ornament_region`: visible decorative geometry in an
independent binary target, not a mutually exclusive “intact ornament” class.
Legacy converters and v1 masks flatten overlaps and cannot be losslessly
reinterpreted without original overlapping annotations. The new projection,
losses, and validator do not modify legacy files or implement a two-head model.

The repository has a temporal semantic backend registry, but no shared neutral
experiment-metadata record suitable for segmentation validation/loss settings.
P0 therefore exposes a small deterministic JSON-serializable record only; it
does not duplicate the backend registry or implement the future experiment
ledger. LPOSS model integration remains outside P0.
