"""Read-only validation for explicit legacy-v1 and two-map-v2 target schemas."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml
from yaml import YAMLError

from .metadata import make_metadata
from .ontology import DEFAULT_ONTOLOGY, Ontology, V1_VERSION, V2_VERSION, extract_mask_ids, load_ontology
from .projection import MAIN_SEMANTIC_IDS, OntologyProjection

COMPONENT_NAME = "ovs_heritage.dataset_validator"
COMPONENT_VERSION = "0.2.0"
VALIDATOR_SCHEMA_VERSION = "heritage-target-validation-v2"
V1_MASK_COLUMNS = ("mask_path", "seg_map_path", "annotation", "mask", "label_path")


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("samples", data.get("items", data.get("data", [])))
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError(f"{path}: manifest must contain a list of sample mappings")
    return data


def _read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"mask does not exist: {path}")
    array = np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else np.asarray(Image.open(path))
    if array.ndim != 2:
        raise ValueError(f"{path}: mask must be single-channel, got shape {array.shape}")
    return array


def _resolve_path(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def _inventory(source: Path, ontology: Ontology) -> tuple[list[dict[str, Any]], str, bool]:
    if source.is_dir():
        if ontology.version != V1_VERSION:
            raise ValueError("v2 requires an explicit manifest with main_mask_path and ornament_mask_path")
        paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in {".png", ".tif", ".tiff", ".npy"})
        rows = [{"mask_path": str(path), "facade_id": None} for path in paths]
        fingerprint = sha256("\n".join(str(path) for path in paths).encode()).hexdigest()
        return rows, fingerprint, False
    rows = _manifest_rows(source)
    return rows, _file_hash(source), "source_id" in (rows[0] if rows else {})


def _validate_v2_row(row: dict[str, Any], index: int, manifest: Path) -> dict[str, Any]:
    label = f"{manifest}: row {index + 1}"
    for field in ("main_mask_path", "ornament_mask_path", "facade_id"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"{label} requires non-empty {field}")
    main_path = _resolve_path(row["main_mask_path"], manifest)
    ornament_path = _resolve_path(row["ornament_mask_path"], manifest)
    main = _read_mask(main_path)
    ornament = _read_mask(ornament_path)
    if main.shape != ornament.shape:
        raise ValueError(f"{label}: main/ornament shape mismatch {main.shape} != {ornament.shape}")
    main_ids = extract_mask_ids(main, str(main_path))
    ornament_ids = extract_mask_ids(ornament, str(ornament_path))
    invalid_main = sorted(main_ids - set(MAIN_SEMANTIC_IDS) - {255})
    if invalid_main:
        raise ValueError(f"{main_path}: invalid Y_main semantic IDs {invalid_main}")
    invalid_ornament = sorted(ornament_ids - {0, 1, 255})
    if invalid_ornament:
        raise ValueError(f"{ornament_path}: invalid Y_ornament values {invalid_ornament}")
    return {
        "facade_id": row["facade_id"],
        "paths": (str(main_path.resolve()), str(ornament_path.resolve())),
        "main": main,
        "ornament": ornament,
        "source_id": row.get("source_id"),
    }


def _validate_v1_row(row: dict[str, Any], index: int, manifest: Path) -> dict[str, Any]:
    key = next((key for key in V1_MASK_COLUMNS if row.get(key)), None)
    if key is None:
        raise ValueError(f"{manifest}: row {index + 1} has no legacy mask path")
    path = _resolve_path(str(row[key]), manifest)
    mask = _read_mask(path)
    ids = extract_mask_ids(mask, str(path))
    invalid = sorted(ids - set(range(11)) - {255})
    if invalid:
        raise ValueError(f"{path}: invalid legacy-v1 IDs {invalid}")
    return {
        "facade_id": row.get("facade_id") or None,
        "paths": (str(path.resolve()),),
        "main": mask,
        "ornament": None,
        "source_id": row.get("source_id"),
    }


def validate_splits(sources: dict[str, str | Path], ontology: Ontology) -> dict[str, Any]:
    projection = OntologyProjection.canonical_v2()
    report: dict[str, Any] = {
        "component": {"name": COMPONENT_NAME, "version": COMPONENT_VERSION},
        "validator_schema_version": VALIDATOR_SCHEMA_VERSION,
        "ontology_version": ontology.version,
        "ontology_hash": ontology.hash,
        "semantic_projection": projection.as_dict() if ontology.version == V2_VERSION else None,
        "ignore_index": ontology.ignore_index,
        "sources": {name: str(value) for name, value in sources.items()},
        "source_fingerprints": {},
        "splits": {},
        "facade_overlaps": [],
        "duplicated_paths": [],
        "warnings": [],
        "errors": [],
    }
    facade_sets: dict[str, set[str]] = {}
    path_sets: dict[str, set[str]] = {}
    for split, source_value in sources.items():
        source = Path(source_value)
        valid_samples: list[dict[str, Any]] = []
        failures = []
        try:
            rows, fingerprint, uses_source_id = _inventory(source, ontology)
            report["source_fingerprints"][split] = fingerprint
        except Exception as exc:
            rows, uses_source_id = [], False
            failures.append(str(exc))
        if not rows:
            failures.append(f"{source}: split is empty")
        for index, row in enumerate(rows):
            try:
                sample = (
                    _validate_v2_row(row, index, source)
                    if ontology.version == V2_VERSION
                    else _validate_v1_row(row, index, source)
                )
                valid_samples.append(sample)
            except Exception as exc:
                failures.append(str(exc))
        main_counts: Counter[int] = Counter()
        ornament_counts: Counter[int] = Counter()
        for sample in valid_samples:
            for value, count in zip(*np.unique(sample["main"], return_counts=True)):
                if int(value) in set(MAIN_SEMANTIC_IDS) | set(range(11)) | {255}:
                    main_counts[int(value)] += int(count)
            if sample["ornament"] is not None:
                for value, count in zip(*np.unique(sample["ornament"], return_counts=True)):
                    if int(value) in {0, 1, 255}:
                        ornament_counts[int(value)] += int(count)
        facades = {sample["facade_id"] for sample in valid_samples if sample["facade_id"]}
        paths = {path for sample in valid_samples for path in sample["paths"]}
        facade_sets[split] = facades
        path_sets[split] = paths
        source_ids = {sample["source_id"] for sample in valid_samples if sample["source_id"]}
        if ontology.version == V2_VERSION and main_counts[11] == 0:
            report["warnings"].append(f"{split}: ADVERTISEMENTS (semantic ID 11) is absent")
        report["errors"].extend(f"{split}: {failure}" for failure in failures)
        report["splits"][split] = {
            "manifest_row_count": len(rows),
            "source_count": len(source_ids) if uses_source_id else len(valid_samples),
            "valid_sample_count": len(valid_samples),
            "failed_sample_count": len(failures),
            "main_mask_count": len(valid_samples),
            "ornament_mask_count": len(valid_samples) if ontology.version == V2_VERSION else 0,
            "main_pixel_count": {str(key): main_counts[key] for key in sorted(main_counts)},
            "ornament_pixel_count": {str(key): ornament_counts[key] for key in sorted(ornament_counts)},
            "errors": failures,
        }
    names = list(sources)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            facade_overlap = sorted(facade_sets.get(left, set()) & facade_sets.get(right, set()))
            path_overlap = sorted(path_sets.get(left, set()) & path_sets.get(right, set()))
            if facade_overlap:
                item = {"splits": [left, right], "facade_ids": facade_overlap}
                report["facade_overlaps"].append(item)
                report["errors"].append(f"facade_id overlap between {left} and {right}: {facade_overlap}")
            if path_overlap:
                item = {"splits": [left, right], "paths": path_overlap}
                report["duplicated_paths"].append(item)
                report["errors"].append(f"mask paths reused between {left} and {right}: {path_overlap}")
    metadata = make_metadata(
        component_name=COMPONENT_NAME,
        component_version=COMPONENT_VERSION,
        ontology_version=ontology.version,
        ontology_hash=ontology.hash,
        mapping=projection.as_dict() if ontology.version == V2_VERSION else {},
        validator_schema_version=VALIDATOR_SCHEMA_VERSION,
        source_fingerprints=report["source_fingerprints"],
    )
    report["reproducibility"] = metadata.to_dict()
    report["valid"] = not report["errors"]
    return report


def _dataset_config(path: Path) -> dict[str, str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except YAMLError as exc:
        raise ValueError(f"{path}: malformed YAML dataset config: {exc}") from exc
    splits = data.get("splits", data)
    return {
        ("val" if name == "validation" else name): str(
            (path.parent / value).resolve() if not Path(value).is_absolute() else Path(value)
        )
        for name, value in splits.items()
        if name in {"train", "val", "validation", "test"}
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    parser.add_argument("--dataset-config", type=Path)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    sources = _dataset_config(args.dataset_config) if args.dataset_config else {}
    sources.update({name: getattr(args, name) for name in ("train", "val", "test") if getattr(args, name)})
    if not sources:
        parser.error("provide --dataset-config or at least one split source")
    try:
        report = validate_splits(sources, load_ontology(args.ontology))
    except Exception as exc:
        report = {"valid": False, "errors": [str(exc)], "warnings": [], "sources": sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "errors": len(report["errors"]), "output": str(args.output)}))
    return 1 if args.strict and not report["valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
