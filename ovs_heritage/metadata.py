"""Neutral deterministic metadata records for future experiment-ledger adapters."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def payload_hash(payload: Mapping[str, Any]) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MetadataRecord:
    payload: Mapping[str, Any]

    @property
    def hash(self) -> str:
        return payload_hash(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return {"payload": dict(self.payload), "hash": self.hash}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def make_metadata(
    *, component_name: str, component_version: str, ontology_version: str,
    ontology_hash: str, mapping: Mapping[str, Any], validator_schema_version: str | None = None,
    source_fingerprints: Mapping[str, str] | None = None,
    vocabulary_specification_hash: str | None = None,
    loss_settings: Mapping[str, Any] | None = None,
    ornament_threshold: float | None = None,
) -> MetadataRecord:
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
