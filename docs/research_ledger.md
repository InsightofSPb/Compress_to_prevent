# Research experiment ledger

The research ledger is an **automatic laboratory notebook** for manually launched
experiments. It records what a run used, what happened, and what files it
produced. It is not a runtime agent, registry, or database.

## Run the temporal baseline

Ledger recording is opt-in. Without `--ledger-dir`, the command retains its
legacy behavior, including overwriting an existing output.

```bash
python tools/run_temporal_change_baselines.py \
  --residual-manifest data/residual_manifest.csv \
  --out-csv outputs/temporal_scores.csv \
  --methods absdiff_l1 \
  --ledger-dir outputs/research-ledger
```

A successful recorded run prints its run ID and ledger path:

```text
Run ID: 82e4...
Ledger: /repo/outputs/research-ledger/82e4.../events.jsonl
```

Each invocation owns `<ledger-dir>/<run-id>/events.jsonl`; that JSONL stream is
the only authoritative ledger record. With recording enabled, the score CSV and
its `.report.json` path must not already exist. A second run must use a new
output path, preventing a recorded artifact from being silently overwritten.

## Verify a run

```python
from research_ledger import Ledger

ledger = Ledger("outputs/research-ledger", "RUN_ID")
ledger.verify()            # verifies envelopes, lifecycle, references, and hash chain
ledger.verify_artifacts()  # separately checks current files against size and SHA-256
run = ledger.reconstruct()
print(run.status, run.stages, run.artifacts)
```

Artifact verification is intentionally separate from event-chain verification:
the immutable event can remain valid even when an external output file was
moved, deleted, or modified.

## What is recorded

Every immutable event uses schema version 1 and has an event ID, run ID,
contiguous sequence, RFC3339 UTC timestamp, stable producer identity, type,
payload, previous-event hash, and SHA-256 event hash. Scientific fingerprints
are canonical hashes of their content and do not depend on timestamps.

The notebook captures:

- the repository Git commit and content-based dirty-tree fingerprint;
- resolved, recursively secret-redacted CLI arguments and their canonical hash;
- the complete manifest hash and complete facade split definition;
- selected pair/facade IDs and a selected source inventory;
- resolved paths, byte sizes, and streaming SHA-256 hashes for selected previous
  images, current images, optional valid masks, and explicitly used model files;
- Git commit and dirty-state fingerprint for a supplied local DINOv2 repository;
- an allowlisted environment snapshot: Python, platform, selected device, NumPy,
  Pillow, and dependencies relevant to requested scorers;
- stage success or bounded, sanitized failure information;
- both the score CSV and report JSON, including stable artifact ID, logical role,
  path, media type, size, streaming hash, producing stage, configuration hash,
  and source event references.

It never dumps environment variables and does not store images, masks, arrays,
weights, credentials, signed URLs, or private endpoints in event payloads.

## Failures and integrity

`run.started` is always first. The ledger permits one terminal `run.completed`
or `run.failed`, forbids later events, validates stage transitions and artifact
references, and rejects unknown or malformed events. Snapshot, validation, and
scoring failures append `run.failed` and re-raise the original exception. A
scoring failure also records `stage.failed`; interruption is never recorded as
success. A torn last line is reported without rewriting its valid prefix.
Retries and corrections are new events or new runs—committed lines are not edited.

Appends use an inter-process lock, flush, and `fsync`. The current locking
implementation uses `fcntl.flock` and targets Linux/POSIX, including WSL. It is
not compatible with native Windows Python. Hash chains are tamper-evident but
are not signatures, trusted timestamps, or an externally notarized chain head.

H2 capability seams and a future orchestrator remain follow-up work. H3 metric
and paper-table lineage beyond these two CLI artifacts is also intentionally
deferred; this H1 slice adds no protocols, registry, dependency injection, or
runtime agent.
