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
V1_DATASET_SCHEMA = "heritage_single_mask_v1"
V2_DATASET_SCHEMA = "heritage_two_map_v2"
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


def _inventory(
    source: Path,
    ontology: Ontology,
    schema_version: str,
    ontology_version: str,
) -> tuple[list[dict[str, Any]], str, bool, dict[int, list[str]]]:
    if source.is_dir():
        if ontology.version != V1_VERSION:
            raise ValueError("v2 requires an explicit manifest with main_mask_path and ornament_mask_path")
        paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in {".png", ".tif", ".tiff", ".npy"})
        rows = [{"mask_path": str(path.resolve())} for path in paths]
        inventory = [f"{path.relative_to(source).as_posix()}:{_file_hash(path)}" for path in paths]
        fingerprint = sha256("\n".join(inventory).encode()).hexdigest()
        return rows, fingerprint, False, {}
    rows = _manifest_rows(source)
    declaration_errors: dict[int, list[str]] = {}
    for index, row in enumerate(rows):
        for field, expected in (
            ("schema_version", schema_version),
            ("ontology_version", ontology_version),
        ):
            declared = row.get(field)
            if declared not in (None, "", expected):
                declaration_errors.setdefault(index, []).append(
                    f"{source}: row {index + 1} declares conflicting {field}={declared!r}; "
                    f"expected {expected!r}"
                )
    return rows, _file_hash(source), "source_id" in (rows[0] if rows else {}), declaration_errors


def _validate_v2_row(row: dict[str, Any], index: int, manifest: Path) -> dict[str, Any]:
    label = f"{manifest}: row {index + 1}"
    for field in ("main_mask_path", "ornament_mask_path", "facade_id"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"{label} requires non-empty {field}")
        if field == "facade_id" and row[field] != row[field].strip():
            raise ValueError(f"{label}: facade_id must not contain surrounding whitespace")
    source_id = row.get("source_id")
    if source_id not in (None, ""):
        if not isinstance(source_id, str) or source_id != source_id.strip():
            raise ValueError(f"{label}: source_id must be a string without surrounding whitespace")
    main_path = _resolve_path(row["main_mask_path"], manifest)
    ornament_path = _resolve_path(row["ornament_mask_path"], manifest)
    image_path = None
    image_shape = None
    if "image_path" in row:
        if not isinstance(row.get("image_path"), str) or not row["image_path"].strip():
            raise ValueError(f"{label}: image_path must be a non-empty string when present")
        image_path = _resolve_path(row["image_path"], manifest)
    physical_paths = {
        "main_mask_path": main_path.resolve(),
        "ornament_mask_path": ornament_path.resolve(),
        **({"image_path": image_path.resolve()} if image_path is not None else {}),
    }
    if len(set(physical_paths.values())) != len(physical_paths):
        roles: dict[str, list[str]] = {}
        for role, path in physical_paths.items():
            roles.setdefault(str(path), []).append(role)
        reused = {path: names for path, names in roles.items() if len(names) > 1}
        raise ValueError(f"{label}: one physical file cannot serve multiple roles: {reused}")
    if image_path is not None:
        try:
            with Image.open(image_path) as image:
                image.load()
                image_shape = (image.height, image.width)
        except Exception as exc:
            raise ValueError(f"{image_path}: image is missing or unreadable: {exc}") from exc
    main = _read_mask(main_path)
    ornament = _read_mask(ornament_path)
    if main.shape != ornament.shape:
        raise ValueError(f"{label}: main/ornament shape mismatch {main.shape} != {ornament.shape}")
    if image_shape is not None and image_shape != main.shape:
        raise ValueError(f"{label}: image/mask grid mismatch {image_shape} != {main.shape}")
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
        "paths": {
            "main_mask_path": str(main_path.resolve()),
            "ornament_mask_path": str(ornament_path.resolve()),
            "image_path": str(image_path.resolve()) if image_path is not None else None,
        },
        "main": main,
        "ornament": ornament,
        "source_id": source_id,
        "image_verified": image_path is not None,
        "dimensions": {"image": image_shape, "main": main.shape, "ornament": ornament.shape},
    }


def _validate_v1_row(row: dict[str, Any], index: int, manifest: Path) -> dict[str, Any]:
    label = f"{manifest}: row {index + 1}"
    for identifier in ("facade_id", "source_id"):
        if identifier in row:
            value = row[identifier]
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(
                    f"{label}: {identifier} must be a non-empty string without surrounding whitespace"
                )
    key = next(
        (
            key
            for key in V1_MASK_COLUMNS
            if key in row and row[key] is not None and row[key] != ""
        ),
        None,
    )
    if key is None:
        raise ValueError(f"{label} has no legacy mask path")
    if not isinstance(row[key], str) or not row[key].strip():
        raise ValueError(f"{label}: {key} must be a non-empty string")
    path = _resolve_path(row[key], manifest)
    mask = _read_mask(path)
    ids = extract_mask_ids(mask, str(path))
    invalid = sorted(ids - set(range(11)) - {255})
    if invalid:
        raise ValueError(f"{path}: invalid legacy-v1 IDs {invalid}")
    return {
        "facade_id": row.get("facade_id"),
        "paths": {"mask_path": str(path.resolve())},
        "main": mask,
        "ornament": None,
        "source_id": row.get("source_id"),
        "image_verified": False,
        "dimensions": {"mask": mask.shape},
    }


def validate_splits(
    sources: dict[str, str | Path],
    ontology: Ontology,
    *,
    schema_version: str,
    ontology_version: str,
) -> dict[str, Any]:
    supported_schemas = {V1_DATASET_SCHEMA, V2_DATASET_SCHEMA}
    if schema_version not in supported_schemas:
        raise ValueError(f"unsupported dataset schema {schema_version!r}; supported: {sorted(supported_schemas)}")
    if ontology_version != ontology.version:
        raise ValueError(
            f"declared ontology_version {ontology_version!r} does not match loaded {ontology.version!r}"
        )
    expected_schema = V2_DATASET_SCHEMA if ontology.version == V2_VERSION else V1_DATASET_SCHEMA
    if schema_version != expected_schema:
        raise ValueError(f"ontology {ontology.version} requires dataset schema {expected_schema}")
    projection = OntologyProjection.from_ontology(ontology) if ontology.version == V2_VERSION else None
    report: dict[str, Any] = {
        "component": {"name": COMPONENT_NAME, "version": COMPONENT_VERSION},
        "validator_schema_version": VALIDATOR_SCHEMA_VERSION,
        "dataset_schema_version": schema_version,
        "ontology_version": ontology.version,
        "ontology_hash": ontology.hash,
        "semantic_projection": projection.as_dict() if projection is not None else None,
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
    source_id_sets: dict[str, set[str]] = {}
    path_roles: dict[str, dict[str, set[str]]] = {}
    report["source_id_overlaps"] = []
    for split, source_value in sources.items():
        source = Path(source_value)
        valid_samples: list[dict[str, Any]] = []
        sample_failures = []
        split_errors = []
        inventory_read = False
        try:
            rows, fingerprint, uses_source_id, declaration_errors = _inventory(
                source, ontology, schema_version, ontology_version,
            )
            inventory_read = True
            report["source_fingerprints"][split] = fingerprint
        except Exception as exc:
            rows, uses_source_id, declaration_errors = [], False, {}
            split_errors.append(str(exc))
        if inventory_read and not rows:
            split_errors.append(f"{source}: split is empty")
        declared_facades = {
            row["facade_id"] for row in rows
            if isinstance(row.get("facade_id"), str)
            and row["facade_id"]
            and row["facade_id"] == row["facade_id"].strip()
        }
        declared_source_ids = {
            row["source_id"] for row in rows
            if isinstance(row.get("source_id"), str)
            and row["source_id"]
            and row["source_id"] == row["source_id"].strip()
        }
        declared_paths: dict[str, set[str]] = {}
        for row in rows:
            for field in ("image_path", "main_mask_path", "ornament_mask_path", *V1_MASK_COLUMNS):
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    physical = str(_resolve_path(value, source).resolve())
                    declared_paths.setdefault(physical, set()).add(field)
        for index, row in enumerate(rows):
            if index in declaration_errors:
                sample_failures.extend(declaration_errors[index])
                continue
            try:
                sample = (
                    _validate_v2_row(row, index, source)
                    if ontology.version == V2_VERSION
                    else _validate_v1_row(row, index, source)
                )
                valid_samples.append(sample)
            except Exception as exc:
                sample_failures.append(str(exc))
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
        facade_sets[split] = declared_facades
        source_id_sets[split] = declared_source_ids
        path_roles[split] = declared_paths
        source_ids = {sample["source_id"] for sample in valid_samples if sample["source_id"]}
        if ontology.version == V2_VERSION and main_counts[11] == 0:
            report["warnings"].append(f"{split}: ADVERTISEMENTS (semantic ID 11) is absent")
        report["errors"].extend(f"{split}: {failure}" for failure in split_errors + sample_failures)
        report["splits"][split] = {
            "manifest_row_count": len(rows),
            "source_count": len(source_ids) if uses_source_id else len(valid_samples),
            "valid_sample_count": len(valid_samples),
            "failed_sample_count": len(sample_failures),
            "split_error_count": len(split_errors),
            "main_mask_count": len(valid_samples),
            "ornament_mask_count": len(valid_samples) if ontology.version == V2_VERSION else 0,
            "verified_image_count": sum(sample["image_verified"] for sample in valid_samples),
            "main_pixel_count": {str(key): main_counts[key] for key in sorted(main_counts)},
            "ornament_pixel_count": {str(key): ornament_counts[key] for key in sorted(ornament_counts)},
            "sample_errors": sample_failures,
            "split_errors": split_errors,
            "dimensions": [sample["dimensions"] for sample in valid_samples],
        }
    names = list(sources)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            facade_overlap = sorted(facade_sets.get(left, set()) & facade_sets.get(right, set()))
            source_id_overlap = sorted(source_id_sets.get(left, set()) & source_id_sets.get(right, set()))
            left_paths = path_roles.get(left, {})
            right_paths = path_roles.get(right, {})
            path_overlap = sorted(set(left_paths) & set(right_paths))
            if facade_overlap:
                item = {"splits": [left, right], "facade_ids": facade_overlap}
                report["facade_overlaps"].append(item)
                report["errors"].append(f"facade_id overlap between {left} and {right}: {facade_overlap}")
            if source_id_overlap:
                item = {"splits": [left, right], "source_ids": source_id_overlap}
                report["source_id_overlaps"].append(item)
                report["errors"].append(
                    f"source_id overlap between {left} and {right}: {source_id_overlap}"
                )
            if path_overlap:
                item = {
                    "splits": [left, right],
                    "paths": [
                        {
                            "path": path,
                            "roles": {
                                left: sorted(left_paths[path]),
                                right: sorted(right_paths[path]),
                            },
                        }
                        for path in path_overlap
                    ],
                }
                report["duplicated_paths"].append(item)
                report["errors"].append(
                    f"physical paths reused between {left} and {right}: {path_overlap}"
                )
    metadata = make_metadata(
        component_name=COMPONENT_NAME,
        component_version=COMPONENT_VERSION,
        ontology_version=ontology.version,
        ontology_hash=ontology.hash,
        mapping=projection.as_dict() if projection is not None else {},
        validator_schema_version=VALIDATOR_SCHEMA_VERSION,
        source_fingerprints=report["source_fingerprints"],
    )
    report["reproducibility"] = metadata.to_dict()
    report["valid"] = not report["errors"]
    return report


def _dataset_config(path: Path) -> tuple[dict[str, str], str, str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except YAMLError as exc:
        raise ValueError(f"{path}: malformed YAML dataset config: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: dataset config must be a mapping")
    schema_version = data.get("schema_version")
    ontology_version = data.get("ontology_version")
    if not isinstance(schema_version, str) or not isinstance(ontology_version, str):
        raise ValueError(f"{path}: dataset config requires schema_version and ontology_version")
    splits = data.get("splits", {})
    sources = {
        ("val" if name == "validation" else name): str(
            (path.parent / value).resolve() if not Path(value).is_absolute() else Path(value)
        )
        for name, value in splits.items()
        if name in {"train", "val", "validation", "test"}
    }
    return sources, schema_version, ontology_version


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--schema-version")
    parser.add_argument("--ontology-version")
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if args.dataset_config:
        sources, schema_version, ontology_version = _dataset_config(args.dataset_config)
        if args.schema_version and args.schema_version != schema_version:
            parser.error("--schema-version conflicts with dataset config")
        if args.ontology_version and args.ontology_version != ontology_version:
            parser.error("--ontology-version conflicts with dataset config")
    else:
        sources = {}
        schema_version = args.schema_version
        ontology_version = args.ontology_version
    sources.update({name: getattr(args, name) for name in ("train", "val", "test") if getattr(args, name)})
    if not sources:
        parser.error("provide --dataset-config or at least one split source")
    if not schema_version or not ontology_version:
        parser.error("explicit --schema-version and --ontology-version are required")
    try:
        report = validate_splits(
            sources,
            load_ontology(args.ontology),
            schema_version=schema_version,
            ontology_version=ontology_version,
        )
    except Exception as exc:
        report = {"valid": False, "errors": [str(exc)], "warnings": [], "sources": sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "errors": len(report["errors"]), "output": str(args.output)}))
    return 1 if args.strict and not report["valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
