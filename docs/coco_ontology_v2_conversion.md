# COCO to ontology-v2 conversion

`python -m ovs_heritage.coco_converter` converts COCO **polygon** annotations
into the existing `heritage_facades_v2_12concepts_two_heads` contract. It never
edits its COCO or image inputs. Run a read-only audit first:

```bash
python -m ovs_heritage.coco_converter audit --coco annotations.json --images-root images --output audit.json
python -m ovs_heritage.coco_converter convert --coco annotations.json --images-root images --metadata metadata.csv --output-dir dataset-v2
python -m ovs_heritage.coco_converter validate --manifest dataset-v2/manifest.jsonl --output validation.json
```

## Representation and rasterization

`main_masks/*.png` stores semantic IDs (not contiguous model-channel indices)
from `{0,1,2,3,4,5,6,7,9,10,11,255}`. Semantic ID 8 is forbidden there.
`ornament_masks/*.png` independently stores `{0,1,255}`; ornament may overlap
any main label. Both are lossless single-channel uint8 PNGs and are read back
after writing to verify shape, dtype, grid, values, and bytes.

Multiple polygons in an annotation are unioned. Polygon rasterization uses the
even-odd rule at pixel centres `(x + 0.5, y + 0.5)` with boundary intersections
included. RLE, empty/malformed polygons, non-finite coordinates, orphan
references, duplicate IDs/names, unknown categories, and dimension mismatches
fail explicitly. Masks are unioned per source category before overlap analysis.

Main overlaps use the versioned operational policy in
`ovs_heritage/configs/coco_conversion_v1.json`: missing element, spalling,
crack, delamination, corrosion, repairs, advertisements, text/images,
efflorescence, then water stain (highest to lowest). This is a deterministic
rasterization rule, not a scientific severity ranking. Ornament is never part
of that ranking. Annotation order therefore has no effect.

## Images and metadata

Resolution first tries the exact, case-sensitive COCO basename. Only if absent,
an eight-hex-digit Label Studio prefix plus hyphen is removed. Arbitrary
prefixes are not stripped. Matches must be unique; missing/ambiguous matches and
collisions between normalized COCO names fail. A unique case-insensitive match
is permitted and reported. Source and canonical names, resolved source path,
normalization, and case fallback are retained.

Metadata is CSV or a JSON array (also `{ "samples": [...] }`). Each row is
keyed by `image_id` (preferred), or `canonical_file_name`, and requires
`facade_id` and `split` (`train`, `validation`/`val`, or `test`). Optional fields
are `building_id`, `capture_date`, and `capture_year`. No value is inferred from
a filename. Missing mappings and facade leakage across splits fail closed.

The output contains copied portable images under `images/`, both mask trees,
`manifest.jsonl`, `overlap_report.json`, `filename_resolution_report.json`, and
`conversion_summary.json`. The summary records source, ontology, and policy
hashes plus artifact sizes/hashes. The overlap report exposes every intersecting
category pair and its automatic or independent-mask treatment. Conversion
requires a new/empty directory; `--overwrite` explicitly replaces the entire
output directory only after all inputs pass the initial COCO inspection.

The real export additionally requires the COCO file, complete source-image
root, and complete reviewed metadata mapping. Snapshot counts are deliberately
not embedded in converter logic.
