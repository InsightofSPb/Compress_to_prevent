# Heritage open-vocabulary foundations (P0)

P0 defines target representation, ontology projection, runtime prototypes, raw
scoring, strict losses, validation, deterministic metadata, and CPU tests. It
does **not** implement the complete two-head model, LPOSS integration, adapter
training, retention evaluation, a general registry, or an experiment ledger;
those integrations belong to P1 or later.

## V2: 12 concepts, two target maps

`heritage_facades_v2_12concepts_two_heads` has stable semantic IDs 0–11, but
these are not 12 mutually exclusive model channels. Stored targets are:

* `Y_main`: `{0,1,2,3,4,5,6,7,9,10,11,255}`. Values remain semantic IDs on
  disk. Concept 8 is invalid here.
* `Y_ornament`: `{0,1,255}`. Positive `1` means visible decorative geometry
  (`ornament_region`, semantic concept 8); the binary mask does not store 8.

`ornament_region` is a pixel-level visible-geometry label independent of damage
or surface condition. Thus corrosion (`Y_main=7`) or water stain (`Y_main=5`)
may overlap `Y_ornament=1`. A completely missing ornament has `Y_ornament=0`;
its absence can be `missing_element` in `Y_main`. This mask never represents an
expected/reconstructed historical footprint.

The legacy name `ornament_intact` was misleading. It is only a deprecated alias
requiring explicit resolution. It does not trigger dataset migration. A legacy
flattened raster cannot reveal the hidden main label under an ornament region;
without original Label Studio/COCO annotations that overlap cannot be recovered
losslessly and must not be invented.

Legacy `heritage_facades_v1_11classes` remains an explicit, separate single-mask
schema. V1 masks are never automatically interpreted as v2.

## Canonical output projection versus runtime vocabulary

Canonical future supervision expects raw `main_logits [N,11,H,W]` and raw
`ornament_logits [N,1,H,W]`. Main semantic IDs map as follows:

```text
semantic ID:  0 1 2 3 4 5 6 7 9 10 11
main channel: 0 1 2 3 4 5 6 7 8  9 10
semantic ID 8 -> ornament head, channel 0
```

`OntologyProjection` performs semantic/channel round trips and preserves 255.
Conversion to contiguous channels occurs only at the main loss boundary;
export converts argmax channels back to semantic IDs. Ornament inference uses
sigmoid and a configurable threshold and remains a separate binary mask. The
two predictions are never flattened together.

The stateless open-vocabulary scorer is different: it returns raw
`[N,C_main,H,W]` logits for an arbitrary reordered/subset/extended/mixed runtime
vocabulary. Runtime entries may have `semantic_id=None`. Such dynamic channels
must not be passed to the canonical supervised loss without an explicit mapping.
Prototype metadata records channel order, nullable semantic IDs, ontology hash,
prompt settings, and a deterministic specification hash.

## Raw-logit losses

`main_segmentation_loss` validates exactly 11 canonical channels, maps stored
semantic IDs to channel indices, and sends raw logits directly to cross entropy.
Main 255 is handled through `ignore_index`; all-ignore input returns a
differentiable zero rather than NaN.

`ornament_region_loss` sends raw logits directly to element-wise
`binary_cross_entropy_with_logits`. It replaces ignored targets with a safe zero,
then averages only pixels where `Y_ornament != 255`; ignored pixels contribute
to neither numerator nor denominator. All-ignore input is a differentiable
zero. `combined_two_head_loss` records finite non-negative `lambda_ornament` and
optional positive `pos_weight`; P0 does not tune either value or a threshold.

## Validation and reproducibility

V2 manifests explicitly contain `main_mask_path`, `ornament_mask_path`, and
`facade_id` (optionally `image_path` and `source_id`). The validator checks both
files, shape equality, strict dtypes/IDs, empty splits, missing facade IDs,
facade leakage, and reused mask paths. Missing advertisements is a warning.
Reports distinguish manifest rows, valid and failed samples, main and ornament
mask counts, and source counts; they do not call unchecked rows “images”.
Unknown IDs are excluded from valid statistics.

Validation also requires explicit `schema_version` and `ontology_version`
declarations, either at the dataset-config level or through the corresponding
CLI options. V2 uses `heritage_two_map_v2`; legacy v1 uses
`heritage_single_mask_v1`. Missing, unknown, conflicting, or ontology-mismatched
declarations are errors and never trigger schema inference. Per-split reports
separate manifest rows, valid/failed sample rows, and inventory-level split
errors; error-message count is not used as a sample count.

Reports include the component/schema versions, ontology version/hash, complete
projection, split fingerprints, overlaps, duplicated paths, warnings/errors,
and a deterministic neutral metadata record. Hashed payloads contain no current
time. `MetadataRecord` is the intended adapter point for a future shared ledger;
no competing registry or provenance JSONL is introduced.

```bash
python -m ovs_heritage.validate_dataset \
  --ontology ovs_heritage/configs/heritage_vocab.yaml \
  --schema-version heritage_two_map_v2 \
  --ontology-version heritage_facades_v2_12concepts_two_heads \
  --train train.csv --val val.csv --test test.csv \
  --output validation-report.json --strict
```

## Checks

```bash
python -m compileall -q ovs_heritage
pytest -q ovs_heritage/tests
ruff check ovs_heritage
```

The CPU GitHub Actions workflow runs these commands on Python 3.9 with versions
compatible with the project environment. No model or dataset download occurs in
the unit tests.
