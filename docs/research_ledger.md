# Research experiment ledger

The research ledger is an optional, append-only provenance stream for manually
launched experiments. It is not a runtime research agent, registry, database,
or orchestration framework.

## Temporal baseline usage

Add `--ledger-dir` to the existing CPU CLI. Omitting it retains the CLI's
existing behavior and output.

```bash
python tools/run_temporal_change_baselines.py \
  --residual-manifest data/residual_manifest.csv \
  --out-csv outputs/temporal_scores.csv \
  --methods absdiff_l1 --ledger-dir outputs/research-ledger
```

Each invocation creates `<ledger-dir>/<run-id>/events.jsonl`. The JSONL file is
the only authoritative store. Every canonical UTF-8 JSON event has a contiguous
sequence number, previous-event hash, and SHA-256 event hash. Appends use an
inter-process lock and are flushed and `fsync`ed before returning. Verification
fails closed on invalid JSON, a torn final line, discontinuity, or a broken hash
chain; it never repairs or rewrites the valid prefix.

## Verify and reconstruct

```python
from research_ledger import Ledger

ledger = Ledger("outputs/research-ledger", "RUN_ID")
ledger.verify()
run = ledger.reconstruct()
print(run.status, run.stages, run.artifacts)
```

Reconstruction is deterministic and in-memory. It reports the latest run and
stage states plus snapshots, warnings, artifacts, event count, and chain head.
Retries and corrections must be appended as new events.

## Captured information and limits

The integration records the Git commit and a content-based dirty-tree
fingerprint; resolved, secret-redacted CLI arguments and their canonical hash;
the source manifest path and SHA-256; sorted facade and temporal-pair IDs; split
definitions and fingerprints; stage outcomes; and the score CSV's absolute
path, media type, byte size, SHA-256, producing stage, and source event IDs.
`ovs_heritage.MetadataRecord` can be adapted with `ontology_snapshot`, retaining
its existing payload and hash rather than duplicating ontology semantics.

The ledger deliberately does not capture the environment wholesale, images,
masks, arrays, weights, or credential-shaped argument values. Files remain
mutable outside the ledger, clocks are not recorded, and integrity hashes detect
tampering but provide neither signatures nor an external notarized chain head.
The planned H2 follow-up may connect these events to narrow capability seams;
provider protocols, a capability registry, dependency injection, and a runtime
agent are explicitly outside this slice.
