# Stock open-vocabulary MaskCLIP/LPOSS inference (P1a)

## Scope and lineage

P1a is inference only: pretrained MaskCLIP/CLIP and DINO remain frozen; there is no heritage
checkpoint, supervised training, adapter, or threshold tuning. P1b adaptation is separate
future work. This repository retains the upstream MIT license and attribution.

The behavioral comparison covered upstream `main_eval.py`, `models/lposs/lposs.py`,
`segmentation/evaluation/lposs_eval.py`, `configs/lposs.yaml`, and `configs/lposs_plus.yaml`
against their corresponding vendored files. The exact locally inspected import snapshot is
repository commit `12b72a50cbbbba0479820c7498f194dc419ff80b`. **The official upstream Git commit
SHA could not be independently resolved because GitHub access returned HTTP 403 in the build
environment.** This is intentionally recorded rather than inventing provenance; resolve and
replace `UPSTREAM_REFERENCE_COMMIT` before treating a GPU parity run as release evidence.

## Modes and data flow

* `maskclip_raw`: CLIP-normalized RGB → frozen dense MaskCLIP features → P0 runtime prompt
  prototypes → unscaled cosine seed scores. It never imports graph dependencies or calls DINO.
* `lposs`: the same seeds plus frozen DINO features → top-k DINO/geometry affinity graph →
  conjugate-gradient label propagation.
* `lposs_plus`: `lposs` followed by the upstream Lab-neighborhood pixel graph refinement.

The uniform cosine scale defaults to `1.0`, matching upstream's feature/prototype dot product.
No softmax is applied before propagation. Outputs are deliberately called scores, not
calibrated probabilities. The only probability conversion is the documented ornament sigmoid.

Whole-image execution follows upstream padding, CLIP/DINO normalization, bilinear feature-grid
alignment, graph solve, and resize to the original RGB grid. Independent tile argmax is never
used. The CLI currently rejects `--inference slide` rather than silently approximating the
upstream location-aware multi-window graph; use `whole` for this focused P1a runner.

## Dependencies and failure behavior

`maskclip_raw` needs the configured PyTorch/OpenCLIP stack but not FAISS or CuPy. `lposs` and
`lposs_plus` preflight CUDA, FAISS GPU, CuPy, cupyx sparse matrices, and its solver before model
construction. Missing functionality raises an actionable error. There is no mode substitution,
downscale, or raw-MaskCLIP fallback. The requested device is used for model, inputs, prompt
tokens, graph seeds, and solver results.

Weights are selected by `configs/lposs*.yaml` (`decode_head.pretrained`, DINO repository/model,
and optional weights). Torch/OpenCLIP may populate their normal caches during a real run; tests
do not construct these models and download nothing.

## Commands

Raw mode:

```bash
python -m ovs_heritage.infer_ovs --image /data/facade.png \
  --model-config configs/lposs.yaml --vocabulary ovs_heritage/configs/heritage_vocab.yaml \
  --mode maskclip_raw --device cuda:0 --output-dir outputs/stock_raw --save-scores
```

First genuine end-to-end GPU smoke test (LPOSS+):

```bash
CUDA_VISIBLE_DEVICES=0 python -m ovs_heritage.infer_ovs \
  --image /absolute/path/to/one-small-rgb.png \
  --model-config configs/lposs_plus.yaml \
  --vocabulary ovs_heritage/configs/heritage_vocab.yaml \
  --ornament-contrast ovs_heritage/configs/ornament_contrast_v1.yaml \
  --mode lposs_plus --device cuda:0 --inference whole \
  --output-dir outputs/stock_ovs_lposs_plus_smoke --save-scores \
  --ledger-dir /absolute/path/to/ledger
```

Inspect `run_manifest.json`: `dino_executed`, `patch_propagation_executed`, and
`pixel_refinement_executed` must all be true; requested/effective `k`, every graph parameter,
grid sizes, finite tensors, timing, memory, configuration/vocabulary/input hashes, and artifact
hashes are recorded. Execution alone is not a scientific quality or parity claim.

## Output contract

The canonical main score tensor has semantic IDs `0,1,2,3,4,5,6,7,9,10,11` in that order and
the PNG mask stores those stable IDs losslessly. Ornament (ID 8) is independent:
`ornament_score = propagated_positive - propagated_non_ornamental_surface`; its sigmoid and an
optional untuned threshold are separate artifacts. A corrosion pixel may simultaneously be
ornament. Runtime concepts with `semantic_id: null` are named score maps and never enter the
main argmax. A canonical mask is omitted for incomplete or ambiguous runtime vocabularies.

## Legacy results

`tools/lposs_inference.py` is MaskCLIP-only despite its name. `tools/finetune.py` and
`tools/finetune_tiled.py` fine-tune that same branch, whose decode head returns softmax
probabilities. Historical masks, checkpoints, and measurements (including mIoU
0.0551→0.1676) remain untouched for reproduction and are **not** full-LPOSS, LPOSS+, DINO, or
open-vocabulary-retention evidence.
