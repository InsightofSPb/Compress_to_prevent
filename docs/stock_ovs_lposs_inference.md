# Stock open-vocabulary MaskCLIP/LPOSS inference (P1a)

## Pinned reference and stock configurations

The behavioral reference is the official MIT-licensed LPOSS repository at commit
[`e489a7445528922ddfe4e39631ef2fe34827c873`](https://github.com/vladan-stojnic/LPOSS/commit/e489a7445528922ddfe4e39631ef2fe34827c873).
Comparison covered `main_eval.py`, `models/lposs/lposs.py`,
`segmentation/evaluation/lposs_eval.py`, `configs/lposs.yaml`, and
`configs/lposs_plus.yaml`. The project vendored those algorithms and later changed its
existing `configs/lposs*.yaml` to DINOv2/DINOv3 experiments. Those files are not stock.

Only these dedicated configurations are stock references:

* `configs/stock_lposs.yaml`: original DINO ViT-B/16 value features, patch 16,
  `alpha=.95`, `gamma=3`, `k=400`, `sigma=.01`, `pix_dist_pow=1`, without pixel refinement.
* `configs/stock_lposs_plus.yaml`: the same patch graph plus upstream LPOSS+ parameters
  `tau=.01`, `r=13` and pixel refinement.

Both pin implementation ID `stock-maskclip-lposs-p1a-v1`, upstream repository/commit,
OpenCLIP `ViT-B-16` with `laion2b_s34b_b88k`, and the original
`facebookresearch/dino:7c446df5b9f45747937fb7d72314ebf7b66930c` `dino_vitb16` Torch Hub entrypoint. Every manifest records these
identifiers, resolved graph values, available local weight hashes, device/dependency facts,
ontology/prototype metadata, and artifact hashes.

## Execution modes

| mode | dense CLIP seeds | DINO | patch LP | pixel refinement |
|---|---:|---:|---:|---:|
| `maskclip_raw` | yes | no | no | no |
| `lposs` | yes | yes | yes | no |
| `lposs_plus` | yes | yes | yes | yes |

The feature model contains the unchanged pretrained CLIP image projection and text encoder,
but no vocabulary-specific segmentation head or prototype buffer. P0 constructs normalized
runtime prototypes. `RawCosineScorer(scale=1.0)` produces unscaled cosine seeds; no softmax is
applied before propagation. LPOSS results are propagated scores, not calibrated probabilities.

`whole` constructs one image graph. `slide` extracts overlapping window features in one batch,
retains each `(y1,y2,x1,x2)` location, constructs one graph over every window patch node,
propagates once, and overlap-averages resized window scores. LPOSS+ refines only the reconstructed
full-resolution map. Crop/stride, coverage, node count, requested/effective `k`, and the configured
node safety limit are validated and recorded. There is no per-tile argmax or fallback.

## Preflight and devices

All inputs, runtime vocabulary data, stock config compatibility, threshold, and the new/empty
output directory are validated before model construction. Graph modes then validate an explicitly
indexed CUDA device, CuPy/cupyx sparse solver, and FAISS GPU before weights load. PyTorch, CuPy,
and FAISS receive the same logical index. `maskclip_raw` neither imports nor initializes DINO,
CuPy, cupyx, or FAISS.

Relative dataset-v2 `image_path` values resolve against the JSONL manifest directory, independent
of the current working directory. Blank lines are allowed; malformed JSON reports its line.
Duplicate/unsafe IDs and paths, unreadable images, collisions, and nonempty output directories fail.

## Commands

One small image in each mode:

```bash
python -m ovs_heritage.infer_ovs --image /data/facade.png \
  --model-config configs/stock_lposs.yaml --mode maskclip_raw --device cuda:0 \
  --vocabulary ovs_heritage/configs/heritage_vocab.yaml --ornament-threshold 0.5 \
  --output-dir outputs/stock_raw_run --save-scores

python -m ovs_heritage.infer_ovs --image /data/facade.png \
  --model-config configs/stock_lposs.yaml --mode lposs --device cuda:0 \
  --vocabulary ovs_heritage/configs/heritage_vocab.yaml --ornament-threshold 0.5 \
  --output-dir outputs/stock_lposs_run --save-scores

python -m ovs_heritage.infer_ovs --image /data/facade.png \
  --model-config configs/stock_lposs_plus.yaml --mode lposs_plus --device cuda:0 \
  --vocabulary ovs_heritage/configs/heritage_vocab.yaml --ornament-threshold 0.5 \
  --output-dir outputs/stock_lposs_plus_run --save-scores
```

Canonical dataset-v2 manifest with a single location-aware slide graph:

```bash
python -m ovs_heritage.infer_ovs --manifest /data/dataset-v2/manifest.jsonl \
  --model-config configs/stock_lposs_plus.yaml --mode lposs_plus --device cuda:1 \
  --inference slide --crop-size 512 512 --stride 341 341 \
  --vocabulary ovs_heritage/configs/heritage_vocab.yaml --ornament-threshold 0.5 \
  --output-dir outputs/stock_slide_immutable_run --save-scores \
  --ledger-dir /data/research-ledger
```

The threshold is configurable metadata, not a tuned or validated recommendation.

## Outputs

Each sample directory contains lossless `main_semantic.png` with stable IDs
`{0,1,2,3,4,5,6,7,9,10,11}` and `ornament_mask.png` with uint8 values `{0,1}`. If requested,
`ornament_visualization.png` uses `{0,255}` strictly for display. `scores.pt` contains seed and
propagated scores, raw ornament contrast, ornament sigmoid, named extra-concept maps, channel names,
nullable semantic IDs, vocabulary hash, and prompt settings. Writes are staged, atomically renamed,
read back, validated, and hashed. Existing scientific runs are never overwritten.

The ledger begins before model loading, records snapshots and stage transitions, registers verified
artifacts with `ArtifactDescriptor`, and ends in `run.completed` or sanitized `run.failed`.

## Verification ladder and exact commands

These four outcomes are deliberately distinct. A CPU contract pass validates routing, exports,
configuration rejection, provenance, and memory guards; it is **not** model execution or numerical
LPOSS parity.

1. **Structural CPU contract test** (no downloads or GPU):

   ```bash
   pytest -q ovs_heritage/tests/test_stock_lposs.py
   ```

2. **Smallest real-GPU model smoke test** (successful execution, not upstream parity):

   ```bash
   python -m ovs_heritage.infer_ovs --image fixtures/lposs/small.png \
     --model-config configs/stock_lposs.yaml --mode maskclip_raw --device cuda:0 \
     --vocabulary ovs_heritage/configs/heritage_vocab.yaml --ornament-threshold 0.5 \
     --output-dir runs/lposs-smoke --save-scores
   ```

3. **Pinned official-reference generation and numerical comparison**:

   ```bash
   git -C third_party/LPOSS checkout e489a7445528922ddfe4e39631ef2fe34827c873
   python tools/check_stock_lposs_gpu.py --upstream-root third_party/LPOSS \
     --image fixtures/lposs/small.png --device cuda:0 --work-dir runs/lposs-parity \
     --atol 1e-5 --rtol 1e-4
   ```

   The checker rejects dirty/wrong upstream checkouts and never accepts an opaque `.pt` input.
   At the reviewed official commit there is no stable arbitrary-vocabulary tensor-export API; the
   bundled adapter therefore reports that exact blocker rather than modifying upstream semantics or
   claiming parity. A `parity_manifest.json` is written only after a real comparison succeeds.

4. **Stock graph inference**:

   ```bash
   python -m ovs_heritage.infer_ovs --manifest dataset-v2/test.jsonl \
     --model-config configs/stock_lposs.yaml --mode lposs --device cuda:0 \
     --vocabulary ovs_heritage/configs/heritage_vocab.yaml --ornament-threshold 0.5 \
     --output-dir runs/dataset-v2-lposs --save-scores
   ```

5. **Stock LPOSS+ inference**:

   ```bash
   python -m ovs_heritage.infer_ovs --manifest dataset-v2/test.jsonl \
     --model-config configs/stock_lposs_plus.yaml --mode lposs_plus --device cuda:0 \
     --vocabulary ovs_heritage/configs/heritage_vocab.yaml --ornament-threshold 0.5 \
     --output-dir runs/dataset-v2-lposs-plus --save-scores
   ```

6. **Later scientific dataset-v2 benchmark**: after successful model execution and real numerical
   parity, run the project evaluation command against the sealed facade-disjoint dataset-v2 split:

   ```bash
   python tools/evaluate_segmentation_tiled.py --help
   ```

   This PR does not run or claim that benchmark. Direct `--image`/`--image-dir` manifests explicitly
   report unavailable dataset/split metadata; JSONL inputs retain every source-record field plus the
   source manifest path and hash.
