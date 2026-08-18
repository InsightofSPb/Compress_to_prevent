"""A deliberately small JSONL experiment ledger.

The event stream is the sole authoritative representation.  Projections are
rebuilt in memory, never persisted as a competing index.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterator, Mapping, Sequence
from types import MappingProxyType
from uuid import uuid4

EVENT_TYPES = frozenset(
    {
        "run.started",
        "source.snapshot",
        "config.snapshot",
        "dataset.snapshot",
        "split.snapshot",
        "ontology.snapshot",
        "stage.started",
        "stage.completed",
        "stage.failed",
        "artifact.created",
        "warning.recorded",
        "run.completed",
        "run.failed",
    }
)
SECRET_MARKERS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "credential",
)


class LedgerError(ValueError):
    """The stream is incomplete or fails its integrity contract."""


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("ledger numbers must be finite")
        return value
    raise TypeError(f"unsupported ledger value: {type(value).__name__}")


def redact_secrets(value: Any) -> Any:
    """Recursively redact credential-shaped mapping values."""
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if any(marker in str(key).lower() for marker in SECRET_MARKERS)
            else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    git_commit: str
    dirty_tree_fingerprint: str


@dataclass(frozen=True)
class NewEvent:
    event_type: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(_plain(self.payload)))


@dataclass(frozen=True)
class StoredEvent:
    run_id: str
    sequence: int
    event_id: str
    event_type: str
    payload: Mapping[str, Any]
    prev_event_hash: str | None
    event_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(_plain(self.payload)))


@dataclass(frozen=True)
class ArtifactDescriptor:
    path: str
    media_type: str
    byte_size: int
    sha256: str
    producing_stage: str
    source_event_ids: tuple[str, ...] = ()

    @classmethod
    def from_path(
        cls,
        path: Path,
        media_type: str,
        producing_stage: str,
        source_event_ids: Sequence[str] = (),
    ) -> "ArtifactDescriptor":
        data = path.read_bytes()
        return cls(
            str(path.resolve()),
            media_type,
            len(data),
            sha256(data).hexdigest(),
            producing_stage,
            tuple(source_event_ids),
        )


@dataclass(frozen=True)
class RunProjection:
    run_id: str
    status: str
    snapshots: Mapping[str, tuple[Mapping[str, Any], ...]]
    stages: Mapping[str, str]
    artifacts: tuple[Mapping[str, Any], ...]
    warnings: tuple[Mapping[str, Any], ...]
    event_count: int
    head_hash: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", _freeze(self.snapshots))
        object.__setattr__(self, "stages", _freeze(self.stages))
        object.__setattr__(self, "artifacts", _freeze(self.artifacts))
        object.__setattr__(self, "warnings", _freeze(self.warnings))


def repository_snapshot(root: Path) -> RunSnapshot:
    def git(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", *args], cwd=root, stderr=subprocess.DEVNULL
        )

    commit = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    # Include working-tree bytes, not timestamps, for every dirty/untracked path.
    pieces = [status]
    for entry in status.split(b"\0"):
        if not entry:
            continue
        rel = entry[3:].decode("utf-8", "surrogateescape")
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1]
        path = root / rel
        pieces.append(rel.encode("utf-8", "surrogateescape"))
        if path.is_file():
            pieces.append(sha256(path.read_bytes()).digest())
    return RunSnapshot("", commit, sha256(b"\0".join(pieces)).hexdigest())


def ontology_snapshot(record: Any) -> NewEvent:
    """Adapt MetadataRecord without reimplementing its ontology hash contract."""
    from ovs_heritage.metadata import MetadataRecord

    if not isinstance(record, MetadataRecord):
        raise TypeError("record must be ovs_heritage.MetadataRecord")
    return NewEvent("ontology.snapshot", {"metadata": record.to_dict()})


class Ledger:
    def __init__(self, root: Path | str, run_id: str):
        if not run_id or "/" in run_id or "\\" in run_id:
            raise ValueError("run_id must be a non-empty path component")
        self.root, self.run_id = Path(root), run_id
        self.run_dir = self.root / run_id
        self.path = self.run_dir / "events.jsonl"
        self.lock_path = self.run_dir / ".lock"

    def append(self, event: NewEvent) -> StoredEvent:
        if event.event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported event type: {event.event_type}")
        payload = redact_secrets(_plain(event.payload))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prior = self.read(verify=True) if self.path.exists() else []
            body = {
                "run_id": self.run_id,
                "sequence": len(prior) + 1,
                "event_id": str(uuid4()),
                "event_type": event.event_type,
                "payload": payload,
                "prev_event_hash": prior[-1].event_hash if prior else None,
            }
            body["event_hash"] = canonical_hash(body)
            encoded = canonical_bytes(body) + b"\n"
            with self.path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            return StoredEvent(**body)

    def read(self, verify: bool = True) -> list[StoredEvent]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise LedgerError("torn final JSONL line")
        events: list[StoredEvent] = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            try:
                events.append(StoredEvent(**json.loads(line)))
            except Exception as exc:
                raise LedgerError(f"invalid event at line {line_number}") from exc
        if verify:
            self.verify(events)
        return events

    def verify(self, events: Sequence[StoredEvent] | None = None) -> None:
        checked = list(events) if events is not None else self.read(verify=False)
        previous = None
        for expected, event in enumerate(checked, 1):
            if event.run_id != self.run_id or event.sequence != expected:
                raise LedgerError(
                    f"non-contiguous or foreign event at sequence {expected}"
                )
            if event.prev_event_hash != previous:
                raise LedgerError(f"broken hash chain at sequence {expected}")
            observed = event.event_hash
            body = {
                "run_id": event.run_id,
                "sequence": event.sequence,
                "event_id": event.event_id,
                "event_type": event.event_type,
                "payload": event.payload,
                "prev_event_hash": event.prev_event_hash,
            }
            if canonical_hash(body) != observed:
                raise LedgerError(f"event hash mismatch at sequence {expected}")
            previous = observed

    def reconstruct(self) -> RunProjection:
        events = self.read()
        snapshots: dict[str, list[Mapping[str, Any]]] = {}
        stages: dict[str, str] = {}
        artifacts, warnings = [], []
        status = "unknown"
        for event in events:
            family = event.event_type
            if family.endswith(".snapshot") or family == "run.started":
                snapshots.setdefault(family, []).append(event.payload)
            if family.startswith("stage."):
                stages[str(event.payload["stage"])] = family.removeprefix("stage.")
            if family == "artifact.created":
                artifacts.append(event.payload)
            if family == "warning.recorded":
                warnings.append(event.payload)
            if family == "run.started":
                status = "running"
            if family == "run.completed":
                status = "completed"
            if family == "run.failed":
                status = "failed"
        return RunProjection(
            self.run_id,
            status,
            {key: tuple(value) for key, value in sorted(snapshots.items())},
            dict(sorted(stages.items())),
            tuple(artifacts),
            tuple(warnings),
            len(events),
            events[-1].event_hash if events else None,
        )

    @contextmanager
    def stage(self, name: str, **payload: Any) -> Iterator[StoredEvent]:
        started = self.append(NewEvent("stage.started", {"stage": name, **payload}))
        try:
            yield started
        except BaseException as exc:
            self.append(
                NewEvent(
                    "stage.failed",
                    {
                        "stage": name,
                        "source_event_ids": [started.event_id],
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            )
            raise
        else:
            self.append(
                NewEvent(
                    "stage.completed",
                    {
                        "stage": name,
                        "source_event_ids": [started.event_id],
                    },
                )
            )


def validate_facade_splits(splits: Mapping[str, Sequence[str]]) -> None:
    owners: dict[str, str] = {}
    for split, facades in splits.items():
        for facade in facades:
            if facade in owners and owners[facade] != split:
                raise ValueError(
                    f"facade {facade!r} overlaps {owners[facade]!r} and {split!r}"
                )
            owners[facade] = split
