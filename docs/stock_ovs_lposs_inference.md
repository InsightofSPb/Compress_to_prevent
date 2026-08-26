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
`facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a` `dino_vitb16` Torch Hub entrypoint. Every manifest records these
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

The following outcomes are deliberately distinct. First create a deterministic 320×320 RGB input;
this yields exactly 400 ViT-B/16 patch nodes, so the upstream stock `k=400` is not reduced:

```bash
mkdir -p runs/lposs-fixture
python - <<'PYFIXTURE'
from pathlib import Path
import numpy as np
from PIL import Image
y, x = np.indices((320, 320))
rgb = np.stack(((x * 13 + y * 3) % 256, (x * 5 + y * 11) % 256,
                (x * 7 + y * 17) % 256), axis=-1).astype(np.uint8)
Image.fromarray(rgb, "RGB").save(Path("runs/lposs-fixture/deterministic-320.png"))
PYFIXTURE
```

1. **CPU unit and contract tests** (no model download and not numerical parity):

   ```bash
   pytest -q ovs_heritage/tests
   ```

2. **Local CPU configuration/preflight smoke** (no model execution):

   ```bash
   python - <<'PYCONTRACT'
from pathlib import Path
from ovs_heritage.infer_ovs import load_config
from ovs_heritage.stock_lposs import DeviceInfo, patch_graph_preflight, pixel_graph_preflight
cfg = load_config(Path("configs/stock_lposs_plus.yaml"))
p = {**cfg["graph"], "available_gpu_bytes": 16 * 1024**3, "gpu_memory_reserve_bytes": 1024**3}
d = DeviceInfo("cuda:0", "cpu", 0, "contract-only")
print(patch_graph_preflight(400, 14, 4, d, p))
print(pixel_graph_preflight(320, 320, 14, 4, d, p))
PYCONTRACT
   ```

3. **Real CUDA model execution** (successful local execution, not upstream parity):

   ```bash
   python -m ovs_heritage.infer_ovs --image runs/lposs-fixture/deterministic-320.png \
     --model-config configs/stock_lposs.yaml --mode lposs --device cuda:0 \
     --vocabulary ovs_heritage/configs/heritage_vocab.yaml --ornament-threshold 0.5 \
     --output-dir runs/lposs-smoke --save-scores
   ```

4. **Pinned official-reference generation and all-mode numerical comparison**:

   ```bash
   git clone https://github.com/vladan-stojnic/LPOSS third_party/LPOSS
   git -C third_party/LPOSS checkout e489a7445528922ddfe4e39631ef2fe34827c873
   test "$(git -C third_party/LPOSS status --porcelain)" = ""
   git init --bare runs/dino-revision-check.git
   git -C runs/dino-revision-check.git fetch --depth=1 \
     https://github.com/facebookresearch/dino.git 7c446df5b9f45747937fb0d72314eb9f7b66930a
   python tools/check_stock_lposs_gpu.py --upstream-root third_party/LPOSS \
     --image runs/lposs-fixture/deterministic-320.png --device cuda:0 \
     --work-dir runs/lposs-parity --atol 1e-5 --rtol 1e-4
   ```

   The checker launches the unmodified official checkout in an isolated subprocess, injects the
   exact same versioned prototype tensor into both implementations, validates checkout/tree/model/
   input/configuration/prototype provenance, compares normalized dense CLIP and DINO features, and
   compares raw, LPOSS, and LPOSS+ score stages. It never
   accepts an externally supplied official score file. `parity_manifest.json` is created with
   `real_gpu_parity=true` only after every required real-CUDA comparison succeeds; any missing or
   mismatched stage exits non-zero.

   The harness being implemented and CPU-contract-tested is not evidence that parity passed. Until
   this command completes on a compatible CUDA/CuPy/FAISS-GPU environment and writes the validated
   manifest, real upstream/local numerical parity remains pending.

5. **Stock dataset-v2 `lposs` inference**:

   ```bash
   python -m ovs_heritage.infer_ovs --manifest dataset-v2/test.jsonl \
     --model-config configs/stock_lposs.yaml --mode lposs --device cuda:0 \
     --vocabulary ovs_heritage/configs/heritage_vocab.yaml --ornament-threshold 0.5 \
     --output-dir runs/dataset-v2-lposs --save-scores
   ```

6. **Stock dataset-v2 `lposs_plus` inference**:

   ```bash
   python -m ovs_heritage.infer_ovs --manifest dataset-v2/test.jsonl \
     --model-config configs/stock_lposs_plus.yaml --mode lposs_plus --device cuda:0 \
     --vocabulary ovs_heritage/configs/heritage_vocab.yaml --ornament-threshold 0.5 \
     --output-dir runs/dataset-v2-lposs-plus --save-scores
   ```

7. **Later scientific dataset-v2 benchmark**: only after successful model execution and real
   numerical parity, use the project evaluation tooling against the sealed facade-disjoint split.
   This PR does not run or claim that benchmark.

Direct `--image`/`--image-dir` records explicitly report unavailable dataset/split metadata. Generic
JSONL is supported but marked non-canonical; canonical dataset-v2 status requires a validated schema,
ontology, facade, split, image path, and the two canonical mask paths emitted by the converter.
Declared dataset-v2 JSONL is checked once, as a complete manifest, by the converter's authoritative
`validate_manifest` implementation; missing artifacts, invalid mask domains, invalid splits, and
image/mask grid mismatches fail closed rather than producing canonical provenance.
`facade_disjoint_split_verified` remains false for a single inference manifest because cross-split
validation evidence is unavailable. The original JSON mapping and its source manifest path, line, and
single computed manifest hash are retained in the existing ledger.
