"""Neutral immutable metadata records for future experiment-ledger adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("metadata mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    raise TypeError(f"metadata value is not JSON serializable: {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _thaw(payload), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    )


def payload_hash(payload: Mapping[str, Any]) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MetadataRecord:
    payload: Mapping[str, Any]
    _hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        frozen = _freeze(self.payload)
        object.__setattr__(self, "payload", frozen)
        object.__setattr__(self, "_hash", payload_hash(frozen))

    @property
    def hash(self) -> str:
        return self._hash

    def to_dict(self) -> dict[str, Any]:
        return {"payload": _thaw(self.payload), "hash": self.hash}

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        )


def make_metadata(
    *, component_name: str, component_version: str, ontology_version: str,
    ontology_hash: str, mapping: Mapping[str, Any], validator_schema_version: str | None = None,
    source_fingerprints: Mapping[str, str] | None = None,
    vocabulary_specification_hash: str | None = None,
    loss_settings: Mapping[str, Any] | None = None,
    ornament_threshold: float | None = None,
) -> MetadataRecord:
    if not component_name.strip() or not component_version.strip():
        raise ValueError("component name and version must be non-empty")
    if ornament_threshold is not None and (
        not math.isfinite(ornament_threshold) or not 0 <= ornament_threshold <= 1
    ):
        raise ValueError("ornament_threshold must be finite and in 0..1")
    payload = {
        "component": {"name": component_name, "version": component_version},
        "ontology": {"version": ontology_version, "hash": ontology_hash},
        "mapping": dict(mapping),
        "validator_schema_version": validator_schema_version,
        "source_fingerprints": dict(sorted((source_fingerprints or {}).items())),
        "vocabulary_specification_hash": vocabulary_specification_hash,
        "loss_settings": dict(loss_settings or {}),
        "ornament_inference_threshold": ornament_threshold,
    }
    return MetadataRecord(payload)
