# Heritage open-vocabulary foundations (P0)

P0 supplies strict data and scoring primitives for later retention experiments. It does **not** implement training adapters, `prompt_only`, `adapter_distill`, LPOSS refinement/evaluation, stitched or open-vocabulary evaluation, checkpoint conversion, or Pareto selection.

## Single ontology and vocabulary

`configs/heritage_vocab.yaml` is the only source of truth and is loaded with
PyYAML's safe loader. Every logical class has an ID, canonical/display name,
description, prompts, aliases, role, heritage flag, groups, and color. The
loader validates genuine integer IDs, the exact v1/v2 class contract, palette,
bidirectional group membership, prompts, and `ignore_index=255`. Its SHA-256
hashes the parsed data as canonical JSON with sorted mapping keys, so paths,
YAML comments/whitespace, and mapping-key order cannot affect it. Runtime list
order remains meaningful and is the exact output-channel order.

`heritage_facades_v2_12classes` has 12 mask classes (0..11), 11 foreground classes (1..11), and 7 damage classes (1..7). `IGNORE=255` is neither a class nor a palette/vocabulary entry. `BACKGROUND=0` is valid. `TEXT_OR_IMAGES=10` means non-commercial writing/graffiti/images; `ADVERTISEMENTS=11` is separate commercial advertising. Prompts affect only text prototypes and never relabel masks.

A runtime vocabulary may be heritage-only, unseen-only, mixed, reordered, and any size. Each class's normalized prompt embeddings are averaged and normalized again. Aliases may add prompt variants but never classes/channels. The injectable encoder makes CPU mocks possible. Prototypes and metadata are returned runtime objects rather than persistent checkpoint weights, preventing checkpoint dependence on vocabulary length/order.

## Raw scoring and supervised loss

`RawCosineScorer` normalizes dense `[N,D,H,W]` (or `[D,H,W]`) features and `[C,D]` prototypes, applies scalar/per-class scale and bias, and returns **raw** `[N,C,H,W]` scores. It owns no fixed classifier or prototype state. `supervised_cross_entropy` validates every target against C plus ignore 255 and passes raw logits directly to PyTorch cross entropy. Applying softmax first changes the objective and gradients because cross entropy already performs log-softmax.

Class imbalance is unequal supervised representation; catastrophic forgetting is loss of foundation text/image geometry. Reweighting/oversampling addresses the former, but by itself does not constrain the latter.

## Validate masks before training

Repository manifests commonly use CSV `mask_path` plus optional `facade_id`; direct mask directories are also accepted. Relative mask paths resolve beside the manifest. Dataset configs use a JSON/YAML-subset `splits` mapping. Run:

```bash
python -m ovs_heritage.validate_dataset \
  --ontology ovs_heritage/configs/heritage_vocab.yaml \
  --dataset-config /path/to/existing_split_config.yaml \
  --output validation-report.json --strict
```

Alternatively pass `--train`, `--val`, and/or `--test`, each a manifest or mask directory. The JSON report contains timestamp/sources, ontology version/hash, ignore index, image/mask counts, IDs, per-ID pixels/frequencies/image incidence, missing classes, unknown IDs/files, warnings, errors, and facade overlaps. The validator is read-only, writes reports even on data errors, and strict mode exits nonzero. Missing advertisements is a warning; unknown IDs and cross-split facade overlap are errors. A v1 source with IDs 0..10 rejects ID 11.

Masks must have a non-boolean integer dtype. Floating-point masks (including
integral-looking values such as `11.0`), booleans, strings, and objects are
rejected with their dtype, observed values, and source filename before any ID
conversion. The supervised loss applies the equivalent check before `.long()`.

## Checks

```bash
pytest -q ovs_heritage/tests
python -m compileall -q ovs_heritage
python -m ovs_heritage.validate_dataset --help
```

P1 must integrate these interfaces into an explicitly designed retention training/evaluation path, decide converter overlap policy, and evaluate comparable models on one v2 test set. None of those outcomes is claimed by P0.
