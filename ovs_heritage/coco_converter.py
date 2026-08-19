"""Transactional COCO-polygon to ontology-v2 two-mask conversion."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from .ontology import DEFAULT_ONTOLOGY, V2_VERSION, Ontology, load_ontology
from .projection import MAIN_SEMANTIC_IDS

SCHEMA_VERSION = "heritage_two_map_v2"
POLICY_PATH = Path(__file__).parent / "configs" / "coco_conversion_v1.json"
HASH_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAIN_ALLOWED = frozenset(MAIN_SEMANTIC_IDS) | {255}
ORNAMENT_ALLOWED = frozenset({0, 1, 255})


class ConversionError(ValueError):
    """An input violates a fail-closed conversion contract."""


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConversionError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read JSON {path}: {exc}") from exc


def load_policy(
    path: Path = POLICY_PATH,
    ontology: Ontology | None = None,
) -> dict[str, Any]:
    policy = _read_json(path)
    if not isinstance(policy, dict):
        raise ConversionError("conversion policy must be a JSON object")
    required = {"version", "source_names", "main_priority_high_to_low"}
    if set(policy) != required:
        raise ConversionError(
            f"invalid conversion policy keys: {sorted(set(policy) ^ required)}"
        )
    if not isinstance(policy["version"], str) or not policy["version"].strip():
        raise ConversionError("conversion policy version must be non-empty")
    mapping = policy["source_names"]
    priority = policy["main_priority_high_to_low"]
    if not isinstance(mapping, dict) or not mapping:
        raise ConversionError("source_names must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in mapping):
        raise ConversionError("source category names must be non-empty strings")
    if any(not isinstance(target, str) or not target for target in mapping.values()):
        raise ConversionError("ontology mapping targets must be non-empty strings")
    if not isinstance(priority, list) or any(
        not isinstance(name, str) or not name for name in priority
    ):
        raise ConversionError("main_priority_high_to_low must be a list of names")
    if len(priority) != len(set(priority)):
        raise ConversionError("main priority contains duplicate classes")

    ontology = ontology or load_ontology(DEFAULT_ONTOLOGY)
    ontology_names = set(ontology.class_names)
    unknown_targets = sorted(set(mapping.values()) - ontology_names)
    if unknown_targets:
        raise ConversionError(
            f"source mapping has unknown ontology targets: {unknown_targets}"
        )
    expected_targets = {item.name for item in ontology.classes if item.id != 0}
    mapped_targets = list(mapping.values())
    if len(mapped_targets) != len(set(mapped_targets)):
        raise ConversionError("source mapping contains duplicate ontology targets")
    if set(mapped_targets) != expected_targets:
        raise ConversionError(
            "source mapping must cover every non-background ontology class "
            f"exactly once; missing={sorted(expected_targets - set(mapped_targets))}, "
            f"extra={sorted(set(mapped_targets) - expected_targets)}"
        )
    expected_main = {
        item.name for item in ontology.classes if item.id in MAIN_SEMANTIC_IDS and item.id
    }
    actual_priority = set(priority)
    unknown_priority = sorted(actual_priority - expected_main)
    missing_priority = sorted(expected_main - actual_priority)
    if unknown_priority or missing_priority or len(priority) != len(expected_main):
        raise ConversionError(
            "main priority must contain every non-background main class exactly "
            f"once; unknown={unknown_priority}, missing={missing_priority}"
        )
    ornament_name = ontology.classes[8].name
    if ornament_name in priority:
        raise ConversionError("ornament-only class must not appear in main priority")
    if mapping.get("ORNAMENT_INTACT") != ornament_name:
        raise ConversionError(
            "ORNAMENT_INTACT must map to the canonical ornament-only class"
        )
    return policy


def _load_coco(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    required = ("images", "annotations", "categories")
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), list) for key in required
    ):
        raise ConversionError(
            "COCO root requires images, annotations, and categories arrays"
        )
    return data


def _unique(
    items: list[dict[str, Any]], field: str, label: str
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or type(item.get(field)) is not int:
            raise ConversionError(f"every {label} requires an integer {field}")
        identifier = item[field]
        if identifier in result:
            raise ConversionError(f"duplicate {label} {field}: {identifier!r}")
        result[identifier] = item
    return result


def _polygon_area(polygon: list[int | float]) -> float:
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    x_values = points[:, 0]
    y_values = points[:, 1]
    return abs(
        float(
            np.dot(x_values, np.roll(y_values, 1))
            - np.dot(y_values, np.roll(x_values, 1))
        )
    ) / 2.0


def validate_segmentation(segmentation: Any, annotation_id: int) -> None:
    """Accept only finite, non-degenerate COCO polygon arrays."""
    if not isinstance(segmentation, list) or not segmentation:
        kind = "RLE" if isinstance(segmentation, dict) else type(segmentation).__name__
        raise ConversionError(
            f"annotation {annotation_id!r}: unsupported or empty segmentation "
            f"({kind}); polygon arrays are required"
        )
    for index, polygon in enumerate(segmentation):
        if (
            not isinstance(polygon, list)
            or len(polygon) < 6
            or len(polygon) % 2
        ):
            raise ConversionError(
                f"annotation {annotation_id!r} polygon {index}: requires at "
                "least three x/y pairs"
            )
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in polygon
        ):
            raise ConversionError(
                f"annotation {annotation_id!r} polygon {index}: coordinates "
                "must be finite numbers"
            )
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if len(np.unique(points, axis=0)) < 3 or _polygon_area(polygon) <= 0:
            raise ConversionError(
                f"annotation {annotation_id!r} polygon {index}: degenerate "
                "or zero-area geometry"
            )


def polygon_mask(
    segmentation: list[list[int | float]],
    width: int,
    height: int,
    annotation_id: int = -1,
) -> np.ndarray:
    """Decode and merge multipart polygons with the canonical COCO decoder.

    COCO clips geometry at the image grid. Coordinates outside the image are
    accepted, reported by preflight, and clipped; geometry decoding to no pixels
    is rejected.
    """
    validate_segmentation(segmentation, annotation_id)
    try:
        parts = coco_mask.frPyObjects(segmentation, height, width)
        merged = coco_mask.merge(parts)
        decoded = coco_mask.decode(merged)
    except Exception as exc:
        raise ConversionError(
            f"annotation {annotation_id!r}: COCO polygon decode failed: {exc}"
        ) from exc
    decoded = np.asarray(decoded, dtype=bool)
    if decoded.shape != (height, width):
        raise ConversionError(
            f"annotation {annotation_id!r}: decoded mask shape "
            f"{decoded.shape} != {(height, width)}"
        )
    if not np.any(decoded):
        raise ConversionError(
            f"annotation {annotation_id!r}: polygon has zero decoded area "
            "after clipping to the image"
        )
    return decoded


class ImageResolver:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ConversionError(f"images root is not a directory: {root}")
        files = sorted(path for path in self.root.rglob("*") if path.is_file())
        self.by_name: dict[str, list[Path]] = defaultdict(list)
        self.by_fold: dict[str, list[Path]] = defaultdict(list)
        for path in files:
            self.by_name[path.name].append(path)
            self.by_fold[path.name.casefold()].append(path)

    def resolve(self, coco_name: str) -> dict[str, Any]:
        basename = Path(coco_name).name
        canonical = HASH_PREFIX.sub("", basename, count=1)
        candidates = self.by_name[basename]
        normalization_applied = False
        if not candidates and canonical != basename:
            candidates = self.by_name[canonical]
            normalization_applied = True
        target = canonical if normalization_applied else basename
        case_insensitive = False
        if not candidates:
            candidates = self.by_fold[target.casefold()]
            case_insensitive = bool(candidates)
        if len(candidates) != 1:
            reason = "missing" if not candidates else "ambiguous"
            raise ConversionError(
                f"{reason} image for COCO file {coco_name!r}: "
                f"{len(candidates)} matches"
            )
        return {
            "source_coco_file_name": coco_name,
            "canonical_file_name": canonical,
            "resolved_image_path": str(candidates[0].resolve()),
            "normalization_applied": normalization_applied,
            "case_insensitive_match": case_insensitive,
            "path": candidates[0],
        }


def resolve_all(
    data: dict[str, Any], root: Path
) -> dict[int, dict[str, Any]]:
    canonical_sources: dict[str, str] = {}
    resolutions: dict[int, dict[str, Any]] = {}
    resolver = ImageResolver(root)
    for image in data["images"]:
        name = image["file_name"]
        normalized = HASH_PREFIX.sub("", Path(name).name, count=1)
        if (
            normalized in canonical_sources
            and canonical_sources[normalized] != name
        ):
            raise ConversionError(
                "COCO filename normalization collision: "
                f"{canonical_sources[normalized]!r} and {name!r} -> "
                f"{normalized!r}"
            )
        canonical_sources[normalized] = name
        resolutions[image["id"]] = resolver.resolve(name)
    return resolutions


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows: Any = list(csv.DictReader(stream))
    else:
        value = _read_json(path)
        rows = value.get("samples", value) if isinstance(value, dict) else value
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in rows
    ):
        raise ConversionError("metadata must be a CSV or JSON array of objects")
    return rows


def _metadata_index(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for original in rows:
        if not original.get("facade_id") or not original.get("split"):
            raise ConversionError(
                "every metadata row requires non-empty facade_id and split"
            )
        split = str(original["split"]).lower()
        split = "validation" if split == "val" else split
        if split not in {"train", "validation", "test"}:
            raise ConversionError(f"invalid metadata split {original['split']!r}")
        row = dict(original, split=split)
        if row.get("image_id") not in (None, ""):
            key = str(row["image_id"])
            if key in by_id:
                raise ConversionError(f"duplicate metadata image_id {key!r}")
            by_id[key] = row
        elif row.get("canonical_file_name"):
            key = str(row["canonical_file_name"])
            if key in by_name:
                raise ConversionError(
                    f"duplicate metadata canonical_file_name {key!r}"
                )
            by_name[key] = row
        else:
            raise ConversionError(
                "metadata row requires image_id or canonical_file_name"
            )
    return by_id, by_name


def _overlap_record(
    left: int,
    right: int,
    semantic_by_category: dict[int, int],
    source_names: dict[int, str],
    priority: dict[int, int],
    policy: dict[str, Any],
    ontology: Ontology,
) -> dict[str, Any]:
    left_semantic = semantic_by_category[left]
    right_semantic = semantic_by_category[right]
    ornament = 8 in {left_semantic, right_semantic}
    winner = None
    if not ornament:
        winner = min((left_semantic, right_semantic), key=priority.get)
    return {
        "source_category_names": [source_names[left], source_names[right]],
        "source_category_ids": [left, right],
        "semantic_ids": [left_semantic, right_semantic],
        "overlapping_pixel_count": 0,
        "image_count": 0,
        "image_ids": [],
        "representative_filenames": [],
        "overlap_type": (
            "ornament + main allowed overlap"
            if ornament
            else "automatically resolved main + main overlap"
        ),
        "applied_rule": "independent masks" if ornament else policy["version"],
        "winning_main_class": (
            None if winner is None else ontology.classes[winner].name
        ),
        "automatically_resolved_pixel_count": 0,
        "manual_review_count": 0,
    }


def preflight(
    coco_path: Path,
    images_root: Path,
    metadata_path: Path | None = None,
    policy_path: Path = POLICY_PATH,
) -> dict[str, Any]:
    """Perform the complete read-only validation used by audit and convert."""
    data = _load_coco(coco_path)
    ontology = load_ontology(DEFAULT_ONTOLOGY)
    policy = load_policy(policy_path, ontology)
    images = _unique(data["images"], "id", "image")
    annotations_by_id = _unique(data["annotations"], "id", "annotation")
    categories = _unique(data["categories"], "id", "category")

    file_names = [image.get("file_name") for image in data["images"]]
    if any(not isinstance(name, str) or not name for name in file_names):
        raise ConversionError("image file_name values must be non-empty strings")
    if len(file_names) != len(set(file_names)):
        raise ConversionError("image file_name values must be unique")
    category_names = [category.get("name") for category in data["categories"]]
    if any(not isinstance(name, str) or not name for name in category_names):
        raise ConversionError("category names must be non-empty strings")
    if len(category_names) != len(set(category_names)):
        raise ConversionError("source category names must be unique")

    semantic_by_category: dict[int, int] = {}
    source_names: dict[int, str] = {}
    for category_id, category in categories.items():
        name = category["name"]
        canonical = policy["source_names"].get(name)
        if canonical is None:
            raise ConversionError(
                f"unknown source category {name!r} "
                f"(COCO category id {category_id!r})"
            )
        semantic_by_category[category_id] = ontology.by_name(canonical).id
        source_names[category_id] = name

    for image_id, image in images.items():
        if (
            type(image.get("width")) is not int
            or type(image.get("height")) is not int
            or image["width"] <= 0
            or image["height"] <= 0
        ):
            raise ConversionError(f"image {image_id!r} has invalid dimensions")
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation_id, annotation in annotations_by_id.items():
        if annotation.get("image_id") not in images:
            raise ConversionError(
                f"orphan annotation {annotation_id!r}: unknown image_id "
                f"{annotation.get('image_id')!r}"
            )
        if annotation.get("category_id") not in categories:
            raise ConversionError(
                f"annotation {annotation_id!r}: unknown category_id "
                f"{annotation.get('category_id')!r}"
            )
        validate_segmentation(annotation.get("segmentation"), annotation_id)
        annotations_by_image[annotation["image_id"]].append(annotation)

    resolutions = resolve_all(data, images_root)
    metadata_by_id: dict[str, dict[str, Any]] = {}
    metadata_by_name: dict[str, dict[str, Any]] = {}
    if metadata_path is not None:
        metadata_by_id, metadata_by_name = _metadata_index(
            load_metadata(metadata_path)
        )

    priority = {
        ontology.by_name(name).id: rank
        for rank, name in enumerate(policy["main_priority_high_to_low"])
    }
    overlaps: dict[tuple[int, int], dict[str, Any]] = {}
    decoded_pixels = 0
    outside_coordinate_annotations = 0
    facade_splits: dict[str, set[str]] = defaultdict(set)
    resolved_metadata: dict[int, dict[str, Any]] = {}

    for image in sorted(data["images"], key=lambda item: item["id"]):
        image_id = image["id"]
        width = image["width"]
        height = image["height"]
        resolution = resolutions[image_id]
        try:
            with Image.open(resolution["path"]) as source_image:
                source_image.load()
                actual_size = source_image.size
        except Exception as exc:
            raise ConversionError(
                f"image {image_id!r} is unreadable: {exc}"
            ) from exc
        if actual_size != (width, height):
            raise ConversionError(
                f"image {image_id!r} dimensions {actual_size} != COCO "
                f"{(width, height)}"
            )

        if metadata_path is not None:
            metadata = metadata_by_id.get(str(image_id)) or metadata_by_name.get(
                resolution["canonical_file_name"]
            )
            if metadata is None:
                raise ConversionError(
                    f"no metadata for image_id {image_id!r} / "
                    f"{resolution['canonical_file_name']!r}"
                )
            resolved_metadata[image_id] = metadata
            facade_splits[str(metadata["facade_id"])].add(metadata["split"])

        category_masks: dict[int, np.ndarray] = {}
        for annotation in annotations_by_image[image_id]:
            segmentation = annotation["segmentation"]
            coordinates = [value for polygon in segmentation for value in polygon]
            x_values = coordinates[0::2]
            y_values = coordinates[1::2]
            if (
                min(x_values) < 0
                or max(x_values) > width
                or min(y_values) < 0
                or max(y_values) > height
            ):
                outside_coordinate_annotations += 1
            decoded = polygon_mask(
                segmentation, width, height, annotation["id"]
            )
            decoded_pixels += int(np.count_nonzero(decoded))
            category_id = annotation["category_id"]
            category_masks.setdefault(
                category_id, np.zeros((height, width), dtype=bool)
            )
            category_masks[category_id] |= decoded

        category_ids = sorted(category_masks)
        for position, left in enumerate(category_ids):
            for right in category_ids[position + 1 :]:
                count = int(
                    np.count_nonzero(category_masks[left] & category_masks[right])
                )
                if not count:
                    continue
                pair = (left, right)
                record = overlaps.setdefault(
                    pair,
                    _overlap_record(
                        left,
                        right,
                        semantic_by_category,
                        source_names,
                        priority,
                        policy,
                        ontology,
                    ),
                )
                record["overlapping_pixel_count"] += count
                record["image_count"] += 1
                record["image_ids"].append(image_id)
                record["representative_filenames"].append(image["file_name"])
                if record["overlap_type"].startswith("automatically"):
                    record["automatically_resolved_pixel_count"] += count

    leaking = {
        facade: sorted(splits)
        for facade, splits in facade_splits.items()
        if len(splits) > 1
    }
    if leaking:
        raise ConversionError(f"facade leakage across splits: {leaking}")

    resolution_report = {
        "images": [
            {key: value for key, value in resolutions[image["id"]].items()
             if key != "path"}
            for image in sorted(data["images"], key=lambda item: item["id"])
        ]
    }
    overlap_report = {
        "policy_version": policy["version"],
        "overlaps": list(overlaps.values()),
        "unknown_categories": [],
        "malformed_annotations": [],
    }
    geometry_report = {
        "annotation_count": len(data["annotations"]),
        "decoded_annotation_pixels": decoded_pixels,
        "annotations_with_outside_coordinates": outside_coordinate_annotations,
        "outside_coordinate_policy": (
            "accepted and clipped to the COCO image grid; geometry decoding "
            "to zero pixels is rejected"
        ),
    }
    return {
        "valid": True,
        "data": data,
        "ontology": ontology,
        "policy": policy,
        "semantic_by_category": semantic_by_category,
        "source_names": source_names,
        "annotations_by_image": annotations_by_image,
        "resolutions": resolutions,
        "metadata": resolved_metadata,
        "priority": priority,
        "resolution_report": resolution_report,
        "overlap_report": overlap_report,
        "geometry_report": geometry_report,
        "source_coco_sha256": _hash(coco_path),
    }


def _write_png(
    path: Path, array: np.ndarray, allowed: frozenset[int]
) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(
        path, optimize=False, compress_level=9
    )
    with Image.open(path) as image:
        restored = np.asarray(image)
    if (
        restored.dtype != np.uint8
        or restored.shape != array.shape
        or not np.array_equal(restored, array)
        or not set(np.unique(restored)) <= allowed
    ):
        raise ConversionError(f"written mask verification failed: {path}")


def _generate_staging(preflight_result: dict[str, Any], staging: Path) -> None:
    data = preflight_result["data"]
    semantic_by_category = preflight_result["semantic_by_category"]
    priority = preflight_result["priority"]
    annotations_by_image = preflight_result["annotations_by_image"]
    resolutions = preflight_result["resolutions"]
    metadata_by_image = preflight_result["metadata"]
    source_hash = preflight_result["source_coco_sha256"]
    ontology = preflight_result["ontology"]
    policy = preflight_result["policy"]

    images_dir = staging / "images"
    main_dir = staging / "main_masks"
    ornament_dir = staging / "ornament_masks"
    images_dir.mkdir()
    main_dir.mkdir()
    ornament_dir.mkdir()
    manifest: list[dict[str, Any]] = []

    for image in sorted(data["images"], key=lambda item: item["id"]):
        image_id = image["id"]
        width = image["width"]
        height = image["height"]
        resolution = resolutions[image_id]
        metadata = metadata_by_image[image_id]
        category_masks: dict[int, np.ndarray] = {}
        for annotation in annotations_by_image[image_id]:
            category_id = annotation["category_id"]
            decoded = polygon_mask(
                annotation["segmentation"], width, height, annotation["id"]
            )
            category_masks.setdefault(
                category_id, np.zeros((height, width), dtype=bool)
            )
            category_masks[category_id] |= decoded

        main = np.zeros((height, width), dtype=np.uint8)
        ornament = np.zeros((height, width), dtype=np.uint8)
        for category_id, mask in category_masks.items():
            if semantic_by_category[category_id] == 8:
                ornament[mask] = 1
        ordered_categories = sorted(
            category_masks,
            key=lambda value: priority.get(semantic_by_category[value], -1),
            reverse=True,
        )
        for category_id in ordered_categories:
            semantic_id = semantic_by_category[category_id]
            if semantic_id != 8:
                main[category_masks[category_id]] = semantic_id

        image_output = images_dir / resolution["canonical_file_name"]
        if image_output.exists():
            raise ConversionError(
                f"portable image name collision: {image_output.name!r}"
            )
        shutil.copyfile(resolution["path"], image_output)
        main_output = main_dir / f"{image_id}.png"
        ornament_output = ornament_dir / f"{image_id}.png"
        _write_png(main_output, main, MAIN_ALLOWED)
        _write_png(ornament_output, ornament, ORNAMENT_ALLOWED)

        row = {
            "sample_id": str(image_id),
            "image_id": image_id,
            **{
                key: value
                for key, value in resolution.items()
                if key != "path"
            },
            "resolved_image_path": str(resolution["path"].resolve()),
            "image_path": image_output.relative_to(staging).as_posix(),
            "main_mask_path": main_output.relative_to(staging).as_posix(),
            "ornament_mask_path": ornament_output.relative_to(staging).as_posix(),
            "facade_id": str(metadata["facade_id"]),
            "building_id": str(metadata.get("building_id", "")),
            "split": metadata["split"],
            "schema_version": SCHEMA_VERSION,
            "ontology_version": V2_VERSION,
            "source_coco_sha256": source_hash,
            "source_annotation_ids": [
                annotation["id"] for annotation in annotations_by_image[image_id]
            ],
            "width": width,
            "height": height,
        }
        for field in ("capture_date", "capture_year"):
            if metadata.get(field) not in (None, ""):
                row[field] = metadata[field]
        manifest.append(row)

    manifest_path = staging / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in manifest
        ),
        encoding="utf-8",
    )
    overlap_path = staging / "overlap_report.json"
    overlap_path.write_text(
        json.dumps(
            preflight_result["overlap_report"], indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    resolution_path = staging / "filename_resolution_report.json"
    resolution_path.write_text(
        json.dumps(
            preflight_result["resolution_report"], indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    geometry_path = staging / "geometry_report.json"
    geometry_path.write_text(
        json.dumps(
            preflight_result["geometry_report"], indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    artifact_paths = (
        [manifest_path, overlap_path, resolution_path, geometry_path]
        + sorted(images_dir.iterdir())
        + sorted(main_dir.iterdir())
        + sorted(ornament_dir.iterdir())
    )
    resolutions_list = list(resolutions.values())
    summary = {
        "source_coco_sha256": source_hash,
        "ontology_version": ontology.version,
        "ontology_hash": ontology.hash,
        "priority_policy_version": policy["version"],
        "priority_policy_hash": _canonical_hash(policy),
        "input_counts": {
            key: len(data[key])
            for key in ("images", "annotations", "categories")
        },
        "output_counts": {
            "samples": len(manifest),
            "main_masks": len(manifest),
            "ornament_masks": len(manifest),
        },
        "filename_resolution_statistics": {
            "exact": sum(
                not item["normalization_applied"]
                and not item["case_insensitive_match"]
                for item in resolutions_list
            ),
            "normalized": sum(
                item["normalization_applied"] for item in resolutions_list
            ),
            "case_insensitive": sum(
                item["case_insensitive_match"] for item in resolutions_list
            ),
        },
        "warnings": [],
        "failures": [],
        "artifacts": [
            {
                "path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _hash(path),
            }
            for path in artifact_paths
        ],
    }
    (staging / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _publish_staging(staging: Path, output_dir: Path, overwrite: bool) -> None:
    parent = output_dir.parent
    backup: Path | None = None
    if output_dir.exists():
        if any(output_dir.iterdir()) and not overwrite:
            raise ConversionError(
                f"output directory is not empty: {output_dir}; pass --overwrite"
            )
        backup = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-", dir=parent)
        )
        backup.rmdir()
        os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        if backup is not None and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def convert(
    coco_path: Path,
    images_root: Path,
    metadata_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    policy_path: Path = POLICY_PATH,
) -> dict[str, Any]:
    result = preflight(coco_path, images_root, metadata_path, policy_path)
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )
    try:
        _generate_staging(result, staging)
        validation = validate_manifest(staging / "manifest.jsonl")
        if not validation["valid"]:
            raise ConversionError(
                f"staged conversion failed validation: {validation['errors']}"
            )
        summary = json.loads(
            (staging / "conversion_summary.json").read_text(encoding="utf-8")
        )
        _publish_staging(staging, output_dir, overwrite)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_manifest(path: Path) -> dict[str, Any]:
    try:
        rows = [
            json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError, ConversionError) as exc:
        return {"valid": False, "sample_count": 0, "errors": [str(exc)]}
    required = {
        "sample_id",
        "image_id",
        "source_coco_file_name",
        "canonical_file_name",
        "resolved_image_path",
        "image_path",
        "main_mask_path",
        "ornament_mask_path",
        "facade_id",
        "building_id",
        "split",
        "schema_version",
        "ontology_version",
        "source_coco_sha256",
        "source_annotation_ids",
        "width",
        "height",
    }
    errors: list[str] = []
    if not rows:
        errors.append("manifest must contain at least one sample")
    facade_splits: dict[str, set[str]] = defaultdict(set)
    identifiers: dict[str, set[Any]] = {
        "sample_id": set(),
        "image_id": set(),
    }
    used_paths: set[Path] = set()
    hashes: set[str] = set()

    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            errors.append(f"row {index}: must be an object")
            continue
        missing = required - set(row)
        if missing:
            errors.append(f"row {index}: missing fields {sorted(missing)}")
            continue
        sample_id_valid = (
            isinstance(row["sample_id"], str) and bool(row["sample_id"])
        )
        image_id_valid = type(row["image_id"]) is int
        if not sample_id_valid:
            errors.append(f"row {index}: sample_id must be a non-empty string")
        else:
            if row["sample_id"] in identifiers["sample_id"]:
                errors.append(
                    f"row {index}: duplicate sample_id {row['sample_id']!r}"
                )
            identifiers["sample_id"].add(row["sample_id"])
        if not image_id_valid:
            errors.append(f"row {index}: image_id must be an integer")
        else:
            if row["image_id"] in identifiers["image_id"]:
                errors.append(
                    f"row {index}: duplicate image_id {row['image_id']!r}"
                )
            identifiers["image_id"].add(row["image_id"])
        dimensions_valid = not (
            type(row["width"]) is not int
            or type(row["height"]) is not int
            or row["width"] <= 0
            or row["height"] <= 0
        )
        if not dimensions_valid:
            errors.append(f"row {index}: dimensions must be positive integers")
        annotation_ids = row["source_annotation_ids"]
        if (
            not isinstance(annotation_ids, list)
            or any(type(value) is not int for value in annotation_ids)
            or len(annotation_ids) != len(set(annotation_ids))
        ):
            errors.append(
                f"row {index}: source_annotation_ids must be unique integers"
            )
        source_hash = row["source_coco_sha256"]
        if not isinstance(source_hash, str) or not SHA256_RE.fullmatch(source_hash):
            errors.append(f"row {index}: invalid source_coco_sha256")
        else:
            hashes.add(source_hash)
        if not row["facade_id"] or not isinstance(row["facade_id"], str):
            errors.append(f"row {index}: facade_id must be a non-empty string")
        split_valid = (
            isinstance(row["split"], str)
            and row["split"] in {"train", "validation", "test"}
        )
        if not split_valid:
            errors.append(f"row {index}: invalid split {row['split']!r}")
        if (
            row["schema_version"] != SCHEMA_VERSION
            or row["ontology_version"] != V2_VERSION
        ):
            errors.append(f"row {index}: incompatible schema or ontology version")
        if isinstance(row["facade_id"], str) and row["facade_id"] and split_valid:
            facade_splits[row["facade_id"]].add(row["split"])

        artifact_paths: dict[str, Path] = {}
        for field in ("image_path", "main_mask_path", "ornament_mask_path"):
            value = row[field]
            if not isinstance(value, str) or not value:
                errors.append(f"row {index}: {field} must be non-empty")
                continue
            artifact = Path(value)
            artifact = artifact if artifact.is_absolute() else path.parent / artifact
            resolved = artifact.resolve()
            if resolved in used_paths:
                errors.append(f"row {index}: duplicate artifact path {value!r}")
            used_paths.add(resolved)
            artifact_paths[field] = artifact
        if len(artifact_paths) != 3 or not dimensions_valid:
            continue
        try:
            with Image.open(artifact_paths["image_path"]) as image:
                image.load()
                image_size = image.size
            with Image.open(artifact_paths["main_mask_path"]) as image:
                main = np.asarray(image)
            with Image.open(artifact_paths["ornament_mask_path"]) as image:
                ornament = np.asarray(image)
            expected_shape = (row["height"], row["width"])
            if (
                main.dtype != np.uint8
                or ornament.dtype != np.uint8
                or main.shape != expected_shape
                or ornament.shape != expected_shape
                or image_size != (row["width"], row["height"])
            ):
                errors.append(f"row {index}: image/mask grid or dtype mismatch")
            if not set(np.unique(main)) <= MAIN_ALLOWED:
                errors.append(f"row {index}: invalid main-mask value domain")
            if not set(np.unique(ornament)) <= ORNAMENT_ALLOWED:
                errors.append(f"row {index}: invalid ornament-mask value domain")
        except Exception as exc:
            errors.append(f"row {index}: unreadable artifact: {exc}")

    if len(hashes) > 1:
        errors.append("manifest has inconsistent source_coco_sha256 values")
    for facade, splits in facade_splits.items():
        if len(splits) > 1:
            errors.append(
                f"facade {facade!r} leaks across splits {sorted(splits)}"
            )
    return {"valid": not errors, "sample_count": len(rows), "errors": errors}


def _audit_report(result: dict[str, Any]) -> dict[str, Any]:
    data = result["data"]
    return {
        "valid": True,
        "source_coco_sha256": result["source_coco_sha256"],
        "ontology_version": result["ontology"].version,
        "ontology_hash": result["ontology"].hash,
        "priority_policy_version": result["policy"]["version"],
        "priority_policy_hash": _canonical_hash(result["policy"]),
        "counts": {
            key: len(data[key])
            for key in ("images", "annotations", "categories")
        },
        "filename_resolution": result["resolution_report"],
        "geometry": result["geometry_report"],
        "overlap": result["overlap_report"],
        "metadata_validated": bool(result["metadata"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--coco", type=Path, required=True)
    audit.add_argument("--images-root", type=Path, required=True)
    audit.add_argument("--metadata", type=Path)
    audit.add_argument("--output", type=Path)
    conversion = subparsers.add_parser("convert")
    conversion.add_argument("--coco", type=Path, required=True)
    conversion.add_argument("--images-root", type=Path, required=True)
    conversion.add_argument("--metadata", type=Path, required=True)
    conversion.add_argument("--output-dir", type=Path, required=True)
    conversion.add_argument("--overwrite", action="store_true")
    validation = subparsers.add_parser("validate")
    validation.add_argument("--manifest", type=Path, required=True)
    validation.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "audit":
            report = _audit_report(
                preflight(args.coco, args.images_root, args.metadata)
            )
        elif args.command == "convert":
            report = convert(
                args.coco,
                args.images_root,
                args.metadata,
                args.output_dir,
                overwrite=args.overwrite,
            )
        else:
            report = validate_manifest(args.manifest)
    except (ConversionError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}))
        return 1
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(report, default=str, ensure_ascii=False))
    return 0 if report.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
