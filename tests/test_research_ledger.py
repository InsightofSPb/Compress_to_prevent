from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from ovs_heritage.metadata import make_metadata
from research_ledger import (
    Ledger,
    LedgerError,
    NewEvent,
    canonical_bytes,
    canonical_hash,
    ontology_snapshot,
)
from research_ledger.ledger import validate_facade_splits


def _append(root: str, run_id: str, number: int) -> None:
    Ledger(root, run_id).append(NewEvent("warning.recorded", {"number": number}))


def test_canonical_serialization_hashing_and_nonfinite_rejection() -> None:
    assert canonical_bytes({"é": 1, "a": [True]}) == b'{"a":[true],"\xc3\xa9":1}'
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        canonical_bytes({"bad": float("nan")})


def test_round_trip_verify_reconstruct_and_secret_redaction(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, "run")
    first = ledger.append(
        NewEvent("run.started", {"api_token": "do-not-store", "x": 1})
    )
    second = ledger.append(NewEvent("warning.recorded", {"message": "ok"}))
    ledger.append(NewEvent("run.completed", {}))
    assert ledger.read()[0].payload["api_token"] == "[REDACTED]"
    assert ledger.read()[1] == second
    projection = ledger.reconstruct()
    assert projection.status == "completed"
    assert projection.event_count == 3
    assert projection.head_hash == ledger.read()[-1].event_hash
    assert first.sequence == 1


@pytest.mark.parametrize(
    "operation", ["mutation", "deletion", "insertion", "reordering"]
)
def test_verify_detects_stream_tampering(tmp_path: Path, operation: str) -> None:
    ledger = Ledger(tmp_path, "run")
    for number in range(4):
        ledger.append(NewEvent("warning.recorded", {"number": number}))
    lines = ledger.path.read_text().splitlines()
    if operation == "mutation":
        item = json.loads(lines[1])
        item["payload"]["number"] = 99
        lines[1] = json.dumps(item)
    elif operation == "deletion":
        del lines[1]
    elif operation == "insertion":
        lines.insert(1, lines[0])
    else:
        lines[1], lines[2] = lines[2], lines[1]
    ledger.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError):
        ledger.read()


def test_concurrent_writers_are_contiguous(tmp_path: Path) -> None:
    processes = [
        multiprocessing.Process(target=_append, args=(str(tmp_path), "run", n))
        for n in range(12)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0
    assert [event.sequence for event in Ledger(tmp_path, "run").read()] == list(
        range(1, 13)
    )


def test_torn_line_preserves_prefix(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, "run")
    ledger.append(NewEvent("run.started", {}))
    prefix = ledger.path.read_bytes()
    with ledger.path.open("ab") as stream:
        stream.write(b'{"partial"')
    with pytest.raises(LedgerError, match="torn"):
        ledger.read()
    assert ledger.path.read_bytes().startswith(prefix)


def test_stage_failure_is_recorded_and_original_propagates(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, "run")
    error = RuntimeError("boom")
    with pytest.raises(RuntimeError) as caught:
        with ledger.stage("score"):
            raise error
    assert caught.value is error
    assert [event.event_type for event in ledger.read()] == [
        "stage.started",
        "stage.failed",
    ]


def test_facade_leakage_rejected() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        validate_facade_splits({"train": ["f1"], "test": ["f1"]})


def test_metadata_adapter_uses_record_hash() -> None:
    record = make_metadata(
        component_name="x",
        component_version="1",
        ontology_version="2",
        ontology_hash="abc",
        mapping={"wall": 1},
    )
    event = ontology_snapshot(record)
    assert event.event_type == "ontology.snapshot"
    assert event.payload["metadata"]["hash"] == record.hash
