from __future__ import annotations

from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import re

import pytest

from ovs_heritage.metadata import make_metadata
from research_ledger import (
    ArtifactDescriptor,
    Ledger,
    LedgerError,
    NewEvent,
    canonical_bytes,
    canonical_hash,
    ontology_snapshot,
    redact_secrets,
    sanitize_error,
)

FIXED = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def clock():
    return FIXED


def _append(root: str, run_id: str, number: int) -> None:
    Ledger(root, run_id).append(NewEvent("warning.recorded", {"number": number}))


def _terminal_run(root: str, run_id: str) -> None:
    ledger = Ledger(root, run_id)
    ledger.append(NewEvent("run.started", {}))
    ledger.append(NewEvent("run.completed", {}))


def started_ledger(tmp_path: Path) -> Ledger:
    ledger = Ledger(tmp_path, "run", clock=clock)
    ledger.append(NewEvent("run.started", {}))
    return ledger


def test_canonical_serialization_hashing_and_nonfinite_and_keys() -> None:
    assert canonical_bytes({"é": 1, "a": [True]}) == b'{"a":[true],"\xc3\xa9":1}'
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        canonical_bytes({"bad": float("nan")})
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_bytes({1: "collision", "1": "other"})
    with pytest.raises(TypeError, match="payload must be a mapping"):
        NewEvent("run.started", [])  # type: ignore[arg-type]


def test_versioned_timestamped_envelope_and_injected_clock(tmp_path: Path) -> None:
    first = started_ledger(tmp_path).read()[0]
    assert first.schema_version == 1
    assert first.timestamp_utc == "2025-01-02T03:04:05.000000Z"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT.*Z", first.timestamp_utc)
    other = Ledger(tmp_path / "other", "run", clock=clock)
    other_first = other.append(NewEvent("run.started", {}, event_id=first.event_id))
    # The run root does not affect the envelope; an injected ID and clock make it reproducible.
    assert other_first.event_hash == first.event_hash


def test_round_trip_reconstruction_and_secret_redaction(tmp_path: Path) -> None:
    ledger = started_ledger(tmp_path)
    second = ledger.append(
        NewEvent("warning.recorded", {"api_token": "no", "message": "ok"})
    )
    ledger.append(NewEvent("run.completed", {}))
    assert ledger.read()[1] == second
    assert second.payload["api_token"] == "[REDACTED]"
    one = ledger.reconstruct()
    two = ledger.reconstruct()
    assert one == two
    assert one.status == "completed" and one.event_count == 3


def test_secret_values_are_redacted_recursively(tmp_path: Path) -> None:
    ledger = started_ledger(tmp_path)
    event = ledger.append(
        NewEvent(
            "warning.recorded",
            {
                "secret": "literal",
                "header": "Bearer abc123",
                "endpoint": "https://alice:password@example.invalid/private",
                "signed": "https://example.invalid/x?ok=yes&api_key=abc&signature=xyz",
                "nested": [{"value": "Basic Zm9vOmJhcg=="}],
                "benign": "ordinary text",
            },
        )
    )
    assert event.payload["secret"] == "[REDACTED]"
    assert event.payload["header"] == "Bearer [REDACTED]"
    assert "alice" not in event.payload["endpoint"]
    assert "abc" not in event.payload["signed"] and "xyz" not in event.payload["signed"]
    assert event.payload["nested"][0]["value"] == "Basic [REDACTED]"
    assert event.payload["benign"] == "ordinary text"
    error = sanitize_error(
        RuntimeError("Bearer abc https://alice:password@example.invalid/x?token=secret")
    )
    assert "abc" not in error["message"]
    assert "alice" not in error["message"]
    assert "secret" not in error["message"]


def test_assignment_secrets_and_malformed_urls_are_safe(tmp_path: Path) -> None:
    assignments = (
        "password=hunter2 passwd=p secret=s token=abc api_key=xyz "
        "api-key=q apikey=r authorization=auth access_key=k access-key=z"
    )
    sanitized = sanitize_error(RuntimeError(assignments))["message"]
    for leaked in (
        "hunter2",
        "token=abc",
        "api_key=xyz",
        "authorization=auth",
        "access_key=k",
    ):
        assert leaked not in sanitized
    malformed = "http://alice:password@example.invalid:notaport/path"
    payload = redact_secrets({"nested": [{"message": assignments}, malformed]})
    assert "hunter2" not in payload["nested"][0]["message"]
    assert "alice:password" not in payload["nested"][1]
    assert sanitize_error(RuntimeError(malformed))["message"]


def test_stage_malformed_url_reraises_original_exception(tmp_path: Path) -> None:
    ledger = started_ledger(tmp_path)
    original = RuntimeError("http://example.invalid:notaport/path")
    with pytest.raises(RuntimeError) as caught:
        with ledger.stage("score"):
            raise original
    assert caught.value is original
    assert ledger.read()[-1].event_type == "stage.failed"


@pytest.mark.parametrize(
    "operation", ["mutation", "deletion", "insertion", "reordering"]
)
def test_verify_detects_stream_tampering(tmp_path: Path, operation: str) -> None:
    ledger = started_ledger(tmp_path)
    for number in range(3):
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


def test_concurrent_writers_are_contiguous_and_valid(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, "run")
    ledger.append(NewEvent("run.started", {}))
    processes = [
        multiprocessing.Process(target=_append, args=(str(tmp_path), "run", n))
        for n in range(12)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0
    events = ledger.read()
    assert [event.sequence for event in events] == list(range(1, 14))


def test_torn_line_preserves_prefix(tmp_path: Path) -> None:
    ledger = started_ledger(tmp_path)
    prefix = ledger.path.read_bytes()
    with ledger.path.open("ab") as stream:
        stream.write(b'{"partial"')
    with pytest.raises(LedgerError, match="torn"):
        ledger.read()
    assert ledger.path.read_bytes().startswith(prefix)


@pytest.mark.parametrize("removed_lines", [1, 2])
def test_terminal_seal_detects_complete_suffix_deletion(
    tmp_path: Path, removed_lines: int
) -> None:
    ledger = started_ledger(tmp_path)
    ledger.append(NewEvent("warning.recorded", {"message": "before terminal"}))
    ledger.append(NewEvent("run.completed", {}))
    lines = ledger.path.read_bytes().splitlines(keepends=True)
    ledger.path.write_bytes(b"".join(lines[:-removed_lines]))
    with pytest.raises(LedgerError, match="does not match its seal"):
        ledger.verify()
    with pytest.raises(LedgerError, match="does not match its seal"):
        ledger.reconstruct()


def test_terminal_seal_detects_missing_entire_event_stream(tmp_path: Path) -> None:
    sealed = started_ledger(tmp_path)
    sealed.append(NewEvent("run.completed", {}))
    sealed.path.unlink()
    with pytest.raises(LedgerError, match="does not match its seal"):
        sealed.read()
    with pytest.raises(LedgerError, match="does not match its seal"):
        sealed.reconstruct()
    with pytest.raises(LedgerError, match="does not match its seal"):
        sealed.append(NewEvent("run.started", {}))

    new_run = Ledger(tmp_path, "unrelated-new-run")
    assert new_run.read() == []
    new_run.append(NewEvent("run.started", {}))
    assert new_run.read()[0].event_type == "run.started"


def test_concurrent_terminal_seals_remain_valid(tmp_path: Path) -> None:
    processes = [
        multiprocessing.Process(
            target=_terminal_run, args=(str(tmp_path), "run-{}".format(n))
        )
        for n in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0
    for n in range(8):
        Ledger(tmp_path, "run-{}".format(n)).verify()
    assert len((tmp_path / "terminal_seals.jsonl").read_text().splitlines()) == 8


def test_stage_failure_records_and_propagates_original(tmp_path: Path) -> None:
    ledger = started_ledger(tmp_path)
    error = RuntimeError("boom")
    with pytest.raises(RuntimeError) as caught:
        with ledger.stage("score"):
            raise error
    assert caught.value is error
    assert [event.event_type for event in ledger.read()][-2:] == [
        "stage.started",
        "stage.failed",
    ]


@pytest.mark.parametrize("run_id", [".", "..", "/tmp/x", "a/b", "a\\b", "", " space"])
def test_unsafe_run_ids_are_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError):
        Ledger(tmp_path, run_id)
    assert not (tmp_path.parent / "events.jsonl").exists()


def test_first_event_and_duplicate_start_are_rejected(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, "run")
    with pytest.raises(LedgerError, match="first"):
        ledger.append(NewEvent("warning.recorded", {}))
    ledger.append(NewEvent("run.started", {}))
    with pytest.raises(LedgerError, match="duplicate"):
        ledger.append(NewEvent("run.started", {}))


def test_terminal_rules(tmp_path: Path) -> None:
    ledger = started_ledger(tmp_path)
    ledger.append(NewEvent("run.completed", {}))
    with pytest.raises(LedgerError, match="after terminal"):
        ledger.append(NewEvent("run.failed", {}))
    with pytest.raises(LedgerError, match="after terminal"):
        ledger.append(NewEvent("warning.recorded", {}))


def test_completed_run_rejects_active_stage(tmp_path: Path) -> None:
    ledger = started_ledger(tmp_path)
    ledger.append(NewEvent("stage.started", {"stage": "score"}))
    with pytest.raises(LedgerError, match="active stage"):
        ledger.append(NewEvent("run.completed", {}))


def test_failed_run_resolves_unrecorded_stage_failure_as_abandoned(
    tmp_path: Path,
) -> None:
    ledger = started_ledger(tmp_path)
    ledger.append(NewEvent("stage.started", {"stage": "score"}))
    ledger.append(
        NewEvent("run.failed", {"error_type": "RuntimeError", "message": "x"})
    )
    assert ledger.reconstruct().stages["score"] == "abandoned_on_run_failure"


def test_stage_state_and_reference_rules(tmp_path: Path) -> None:
    ledger = started_ledger(tmp_path)
    with pytest.raises(LedgerError, match="stage result"):
        ledger.append(
            NewEvent("stage.completed", {"stage": "x", "source_event_ids": ["missing"]})
        )
    start = ledger.append(NewEvent("stage.started", {"stage": "x"}))
    ledger.append(
        NewEvent(
            "stage.completed", {"stage": "x", "source_event_ids": [start.event_id]}
        )
    )
    with pytest.raises(LedgerError, match="transition"):
        ledger.append(
            NewEvent(
                "stage.failed", {"stage": "x", "source_event_ids": [start.event_id]}
            )
        )


def test_invalid_artifact_reference_and_incomplete_stage(tmp_path: Path) -> None:
    ledger = started_ledger(tmp_path)
    payload = {
        "artifact_id": "a",
        "logical_role": "score_csv",
        "path": "/x",
        "media_type": "text/csv",
        "byte_size": 1,
        "sha256": "0" * 64,
        "producing_stage": "score",
        "config_hash": "1" * 64,
        "source_event_ids": ["missing"],
    }
    with pytest.raises(LedgerError, match="invalid source"):
        ledger.append(NewEvent("artifact.created", payload))


def test_duplicate_event_id_and_unknown_type(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, "run")
    ledger.append(NewEvent("run.started", {}, event_id="same"))
    with pytest.raises(LedgerError, match="duplicate event ID"):
        ledger.append(NewEvent("warning.recorded", {}, event_id="same"))
    with pytest.raises(ValueError, match="unsupported"):
        ledger.append(NewEvent("unknown", {}))


def test_artifact_verification_detects_mutation(tmp_path: Path) -> None:
    ledger = started_ledger(tmp_path)
    with ledger.stage("score"):
        pass
    completed = ledger.read()[-1]
    artifact_path = tmp_path / "score.csv"
    artifact_path.write_text("a\n1\n")
    descriptor = ArtifactDescriptor.from_path(
        artifact_path, "score_csv", "text/csv", "score", "c" * 64, [completed.event_id]
    )
    ledger.append(NewEvent("artifact.created", descriptor.to_dict()))
    ledger.verify_artifacts()
    artifact_path.write_text("changed")
    with pytest.raises(LedgerError, match="mismatch"):
        ledger.verify_artifacts()


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
