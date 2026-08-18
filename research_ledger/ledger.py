"""Durable append-only JSONL provenance for manually launched experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import warnings
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4

SCHEMA_VERSION = 1
PRODUCER = MappingProxyType(
    {"component": "compress-to-prevent.research-ledger", "version": "1"}
)
EVENT_TYPES = frozenset(
    {
        "run.started",
        "source.snapshot",
        "config.snapshot",
        "dataset.snapshot",
        "split.snapshot",
        "environment.snapshot",
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
TERMINAL_TYPES = frozenset({"run.completed", "run.failed"})
SECRET_MARKERS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "credential",
    "access_key",
    "accesskey",
    "signature",
    "authorization",
)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_ERROR_MESSAGE = 2000


class LedgerError(ValueError):
    """A stream or external artifact fails its ledger contract."""


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("ledger mapping keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
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
    raise TypeError("unsupported ledger value: {}".format(type(value).__name__))


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return value
    hostname = parsed.hostname or ""
    if parsed.port is not None:
        hostname = "{}:{}".format(hostname, parsed.port)
    if parsed.username is not None or parsed.password is not None:
        hostname = "[REDACTED]@{}".format(hostname)
    query = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        redacted = (
            "[REDACTED]"
            if any(marker in key.lower() for marker in SECRET_MARKERS)
            else item
        )
        query.append((key, redacted))
    return urlunsplit(
        (parsed.scheme, hostname, parsed.path, urlencode(query), parsed.fragment)
    )


def _redact_string(value: str) -> str:
    value = re.sub(
        r"(?i)\b(bearer|basic)\s+\S+",
        lambda match: "{} [REDACTED]".format(match.group(1)),
        value,
    )
    return re.sub(r"https?://[^\s]+", lambda match: _redact_url(match.group(0)), value)


def redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("ledger mapping keys must be strings")
        return {
            key: "[REDACTED]"
            if any(marker in key.lower() for marker in SECRET_MARKERS)
            else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def sanitize_error(exc: BaseException) -> Mapping[str, str]:
    message = " ".join(str(exc).replace("\x00", "").split())[:MAX_ERROR_MESSAGE]
    message = redact_secrets(message)
    return {"error_type": type(exc).__name__, "message": message}


def _recording_warning(
    exc: BaseException, context: str, recording_error: Exception
) -> None:
    note = "ledger could not record {}: {}".format(context, recording_error)
    if hasattr(exc, "add_note"):
        exc.add_note(note)
    warnings.warn(note, RuntimeWarning)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
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


def file_descriptor(path: Union[Path, str]) -> Mapping[str, Any]:
    resolved = Path(path).resolve()
    digest = sha256()
    size = 0
    with resolved.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(resolved), "byte_size": size, "sha256": digest.hexdigest()}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(clock: Callable[[], Union[datetime, str]]) -> str:
    value = clock()
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ledger clock must return a timezone-aware datetime")
        value = (
            value.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z", value
    ):
        raise ValueError("timestamp_utc must be RFC3339 UTC")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp_utc must be RFC3339 UTC") from exc
    return value


@dataclass(frozen=True)
class RunSnapshot:
    git_commit: str
    dirty_tree_fingerprint: str


@dataclass(frozen=True)
class NewEvent:
    event_type: str
    payload: Mapping[str, Any]
    event_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.event_id is not None and (
            not isinstance(self.event_id, str) or not self.event_id
        ):
            raise ValueError("event_id must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise TypeError("event payload must be a mapping")
        object.__setattr__(self, "payload", _freeze(_plain(self.payload)))


@dataclass(frozen=True)
class StoredEvent:
    schema_version: int
    event_id: str
    run_id: str
    sequence: int
    timestamp_utc: str
    event_type: str
    producer: Mapping[str, str]
    payload: Mapping[str, Any]
    prev_event_hash: Optional[str]
    event_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("event payload must be a mapping")
        object.__setattr__(self, "producer", _freeze(_plain(self.producer)))
        object.__setattr__(self, "payload", _freeze(_plain(self.payload)))


@dataclass(frozen=True)
class ArtifactDescriptor:
    artifact_id: str
    logical_role: str
    path: str
    media_type: str
    byte_size: int
    sha256: str
    producing_stage: str
    config_hash: str
    source_event_ids: Tuple[str, ...]

    @classmethod
    def from_path(
        cls,
        path: Path,
        logical_role: str,
        media_type: str,
        producing_stage: str,
        config_hash: str,
        source_event_ids: Sequence[str],
    ) -> "ArtifactDescriptor":
        info = file_descriptor(path)
        identity = canonical_hash(
            {
                "logical_role": logical_role,
                "path": info["path"],
                "sha256": info["sha256"],
                "config_hash": config_hash,
            }
        )
        return cls(
            identity,
            logical_role,
            str(info["path"]),
            media_type,
            int(info["byte_size"]),
            str(info["sha256"]),
            producing_stage,
            config_hash,
            tuple(source_event_ids),
        )

    def to_dict(self) -> Mapping[str, Any]:
        return _plain(self.__dict__)


@dataclass(frozen=True)
class RunProjection:
    run_id: str
    status: str
    snapshots: Mapping[str, Tuple[Mapping[str, Any], ...]]
    stages: Mapping[str, str]
    artifacts: Tuple[Mapping[str, Any], ...]
    warnings: Tuple[Mapping[str, Any], ...]
    event_count: int
    head_hash: Optional[str]

    def __post_init__(self) -> None:
        for name in ("snapshots", "stages", "artifacts", "warnings"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


def repository_snapshot(root: Path, exclude_paths: Sequence[Path] = ()) -> RunSnapshot:
    def git(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", *args], cwd=root, stderr=subprocess.DEVNULL
        )

    commit = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    excluded = [path.resolve() for path in exclude_paths]
    pieces = []
    for entry in status.split(b"\0"):
        if not entry:
            continue
        rel = entry[3:].decode("utf-8", "surrogateescape")
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1]
        path = root / rel
        resolved = path.resolve()
        if any(resolved == item or item in resolved.parents for item in excluded):
            continue
        pieces.append(entry)
        pieces.append(rel.encode("utf-8", "surrogateescape"))
        if path.is_file():
            pieces.append(bytes.fromhex(str(file_descriptor(path)["sha256"])))
    return RunSnapshot(commit, sha256(b"\0".join(pieces)).hexdigest())


def ontology_snapshot(record: Any) -> NewEvent:
    from ovs_heritage.metadata import MetadataRecord

    if not isinstance(record, MetadataRecord):
        raise TypeError("record must be ovs_heritage.MetadataRecord")
    return NewEvent("ontology.snapshot", {"metadata": record.to_dict()})


def _require_payload(event: StoredEvent, *keys: str) -> None:
    if not isinstance(event.payload, Mapping) or any(
        key not in event.payload for key in keys
    ):
        raise LedgerError("malformed payload for {}".format(event.event_type))


class Ledger:
    def __init__(
        self,
        root: Union[Path, str],
        run_id: str,
        clock: Callable[[], Union[datetime, str]] = utc_now,
    ):
        if not isinstance(run_id, str):
            raise ValueError("run_id must be one safe path component")
        candidate = Path(run_id)
        if (
            run_id in (".", "..")
            or candidate.is_absolute()
            or not RUN_ID_RE.fullmatch(run_id)
            or candidate.name != run_id
        ):
            raise ValueError("run_id must be one safe path component")
        self.root, self.run_id, self.clock = Path(root), run_id, clock
        self.run_dir = self.root / run_id
        if self.run_dir.is_symlink():
            raise ValueError("run directory must not be a symbolic link")
        self.path = self.run_dir / "events.jsonl"
        self.lock_path = self.run_dir / ".lock"
        self.seals_path = self.root / "terminal_seals.jsonl"
        self.seals_lock_path = self.root / ".terminal-seals.lock"

    def append(self, event: NewEvent) -> StoredEvent:
        if event.event_type not in EVENT_TYPES:
            raise ValueError("unsupported event type: {}".format(event.event_type))
        payload = redact_secrets(_plain(event.payload))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prior = self.read(verify=True) if self.path.exists() else []
            body = {
                "schema_version": SCHEMA_VERSION,
                "event_id": event.event_id or str(uuid4()),
                "run_id": self.run_id,
                "sequence": len(prior) + 1,
                "timestamp_utc": _timestamp(self.clock),
                "event_type": event.event_type,
                "producer": dict(PRODUCER),
                "payload": payload,
                "prev_event_hash": prior[-1].event_hash if prior else None,
            }
            body["event_hash"] = canonical_hash(body)
            stored = StoredEvent(**body)
            self.verify(prior + [stored], check_terminal_seal=False)
            with self.path.open("ab") as stream:
                stream.write(canonical_bytes(body) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            if stored.event_type in TERMINAL_TYPES:
                self._append_terminal_seal(stored)
            return stored

    def read(self, verify: bool = True) -> list[StoredEvent]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise LedgerError("torn final JSONL line")
        events = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            try:
                data = json.loads(line)
                if set(data) != {
                    "schema_version",
                    "event_id",
                    "run_id",
                    "sequence",
                    "timestamp_utc",
                    "event_type",
                    "producer",
                    "payload",
                    "prev_event_hash",
                    "event_hash",
                }:
                    raise ValueError("invalid envelope fields")
                events.append(StoredEvent(**data))
            except Exception as exc:
                raise LedgerError(
                    "invalid event at line {}".format(line_number)
                ) from exc
        if verify:
            self.verify(events)
        return events

    def verify(
        self,
        events: Optional[Sequence[StoredEvent]] = None,
        check_terminal_seal: bool = True,
    ) -> None:
        checked = list(events) if events is not None else self.read(verify=False)
        previous = None
        ids = set()
        stage_states = {}
        artifact_ids = set()
        terminal = False
        for expected, event in enumerate(checked, 1):
            if (
                event.schema_version != SCHEMA_VERSION
                or event.event_type not in EVENT_TYPES
            ):
                raise LedgerError(
                    "unknown schema or event type at sequence {}".format(expected)
                )
            if _plain(event.producer) != dict(PRODUCER):
                raise LedgerError("unknown producer at sequence {}".format(expected))
            _timestamp(lambda: event.timestamp_utc)
            if event.run_id != self.run_id or event.sequence != expected:
                raise LedgerError(
                    "non-contiguous or foreign event at sequence {}".format(expected)
                )
            if event.event_id in ids:
                raise LedgerError("duplicate event ID")
            if not isinstance(event.event_id, str) or not event.event_id:
                raise LedgerError("invalid event ID")
            ids.add(event.event_id)
            if expected == 1 and event.event_type != "run.started":
                raise LedgerError("run.started must be the first event")
            if expected > 1 and event.event_type == "run.started":
                raise LedgerError("duplicate run.started")
            if terminal:
                raise LedgerError("event after terminal event")
            if event.prev_event_hash != previous:
                raise LedgerError("broken hash chain at sequence {}".format(expected))
            body = {
                key: _plain(getattr(event, key))
                for key in (
                    "schema_version",
                    "event_id",
                    "run_id",
                    "sequence",
                    "timestamp_utc",
                    "event_type",
                    "producer",
                    "payload",
                    "prev_event_hash",
                )
            }
            if canonical_hash(body) != event.event_hash:
                raise LedgerError("event hash mismatch at sequence {}".format(expected))
            if event.event_type == "stage.started":
                _require_payload(event, "stage")
                stage_states[event.event_id] = {
                    "name": event.payload["stage"],
                    "status": "started",
                }
            elif event.event_type in ("stage.completed", "stage.failed"):
                _require_payload(event, "stage", "source_event_ids")
                refs = event.payload["source_event_ids"]
                if (
                    not isinstance(refs, tuple)
                    or len(refs) != 1
                    or refs[0] not in stage_states
                ):
                    raise LedgerError(
                        "stage result must reference its stage.started event"
                    )
                state = stage_states[refs[0]]
                if (
                    state["name"] != event.payload["stage"]
                    or state["status"] != "started"
                ):
                    raise LedgerError("invalid stage transition")
                state["status"] = event.event_type.removeprefix("stage.")
            elif event.event_type == "artifact.created":
                _require_payload(
                    event,
                    "artifact_id",
                    "logical_role",
                    "path",
                    "media_type",
                    "byte_size",
                    "sha256",
                    "producing_stage",
                    "config_hash",
                    "source_event_ids",
                )
                refs = event.payload["source_event_ids"]
                if event.payload["artifact_id"] in artifact_ids:
                    raise LedgerError("duplicate artifact ID")
                artifact_ids.add(event.payload["artifact_id"])
                if not isinstance(refs, tuple) or any(ref not in ids for ref in refs):
                    raise LedgerError("artifact has invalid source event reference")
                completed = [
                    ref
                    for ref in refs
                    if ref in stage_states
                    and stage_states[ref]["status"] == "completed"
                    and stage_states[ref]["name"] == event.payload["producing_stage"]
                ]
                if not completed:
                    # Artifact references the stage.completed event, not only its start.
                    completed_event_ids = {
                        item.event_id
                        for item in checked[: expected - 1]
                        if item.event_type == "stage.completed"
                        and item.payload["stage"] == event.payload["producing_stage"]
                    }
                    if not completed_event_ids.intersection(refs):
                        raise LedgerError("artifact producing stage is not completed")
            if event.event_type in TERMINAL_TYPES:
                if event.event_type == "run.completed" and any(
                    state["status"] == "started" for state in stage_states.values()
                ):
                    raise LedgerError("run.completed cannot contain an active stage")
                terminal = True
            previous = event.event_hash
        if check_terminal_seal:
            self._verify_terminal_seal(checked)

    def _read_terminal_seals(self) -> list[Mapping[str, Any]]:
        if not self.seals_path.exists():
            return []
        raw = self.seals_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise LedgerError("torn terminal seal line")
        seals = []
        previous = None
        for line_number, line in enumerate(raw.splitlines(), 1):
            try:
                seal = json.loads(line)
                observed = seal.pop("seal_hash")
                if seal.get("previous_seal_hash") != previous:
                    raise ValueError("broken terminal seal chain")
                if canonical_hash(seal) != observed:
                    raise ValueError("terminal seal hash mismatch")
                seal["seal_hash"] = observed
                previous = observed
                seals.append(seal)
            except Exception as exc:
                raise LedgerError(
                    "invalid terminal seal at line {}".format(line_number)
                ) from exc
        return seals

    def _append_terminal_seal(self, terminal: StoredEvent) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.seals_lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            seals = self._read_terminal_seals()
            if any(seal["run_id"] == self.run_id for seal in seals):
                raise LedgerError("terminal seal already exists for run")
            body = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "terminal_event_type": terminal.event_type,
                "event_count": terminal.sequence,
                "head_event_hash": terminal.event_hash,
                "timestamp_utc": terminal.timestamp_utc,
                "previous_seal_hash": seals[-1]["seal_hash"] if seals else None,
            }
            body["seal_hash"] = canonical_hash(body)
            with self.seals_path.open("ab") as stream:
                stream.write(canonical_bytes(body) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _verify_terminal_seal(self, events: Sequence[StoredEvent]) -> None:
        seals = self._read_terminal_seals()
        matches = [seal for seal in seals if seal["run_id"] == self.run_id]
        terminal = (
            events[-1] if events and events[-1].event_type in TERMINAL_TYPES else None
        )
        if matches:
            if len(matches) != 1 or terminal is None:
                raise LedgerError("terminal event stream does not match its seal")
            seal = matches[0]
            if (
                seal["terminal_event_type"] != terminal.event_type
                or seal["event_count"] != len(events)
                or seal["head_event_hash"] != terminal.event_hash
                or seal["timestamp_utc"] != terminal.timestamp_utc
            ):
                raise LedgerError("terminal event stream does not match its seal")
        elif terminal is not None:
            raise LedgerError("terminal event has no independent seal")

    def verify_artifacts(self) -> None:
        for event in self.read():
            if event.event_type != "artifact.created":
                continue
            try:
                actual = file_descriptor(str(event.payload["path"]))
            except OSError as exc:
                raise LedgerError(
                    "artifact is missing: {}".format(event.payload["path"])
                ) from exc
            if (
                actual["byte_size"] != event.payload["byte_size"]
                or actual["sha256"] != event.payload["sha256"]
            ):
                raise LedgerError("artifact mismatch: {}".format(event.payload["path"]))

    def reconstruct(self) -> RunProjection:
        events = self.read()
        snapshots = {}
        stages, artifacts, warnings = {}, [], []
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
        if status == "failed":
            stages = {
                name: ("abandoned_on_run_failure" if state == "started" else state)
                for name, state in stages.items()
            }
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
            failure = {"stage": name, "source_event_ids": [started.event_id]}
            failure.update(sanitize_error(exc))
            try:
                self.append(NewEvent("stage.failed", failure))
            except Exception as recording_error:
                _recording_warning(exc, "stage failure", recording_error)
            raise
        else:
            self.append(
                NewEvent(
                    "stage.completed",
                    {"stage": name, "source_event_ids": [started.event_id]},
                )
            )
