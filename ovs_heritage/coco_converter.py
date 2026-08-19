"""Deterministic COCO-polygon to ontology-v2 two-mask conversion."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
from PIL import Image

from .ontology import DEFAULT_ONTOLOGY, V2_VERSION, load_ontology

SCHEMA_VERSION = "heritage_two_map_v2"
POLICY_PATH = Path(__file__).parent / "configs" / "coco_conversion_v1.json"
HASH_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")
MAIN_ALLOWED = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 255})


class ConversionError(ValueError):
    """An input violates a fail-closed conversion contract."""


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {"version", "source_names", "main_priority_high_to_low"}
    if set(policy) != required:
        raise ConversionError(f"invalid conversion policy keys: {sorted(set(policy) ^ required)}")
    return policy


def _load_coco(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read COCO JSON {path}: {exc}") from exc
    if not isinstance(data, dict) or any(not isinstance(data.get(k), list) for k in ("images", "annotations", "categories")):
        raise ConversionError("COCO root requires images, annotations, and categories arrays")
    return data


def _unique(items: list[dict[str, Any]], field: str, label: str) -> dict[Any, dict[str, Any]]:
    result = {}
    for item in items:
        if not isinstance(item, dict) or field not in item:
            raise ConversionError(f"every {label} requires {field}")
        if item[field] in result:
            raise ConversionError(f"duplicate {label} {field}: {item[field]!r}")
        result[item[field]] = item
    return result


def inspect_coco(coco_path: Path, policy_path: Path = POLICY_PATH) -> tuple[dict[str, Any], dict[int, int], dict[int, str]]:
    data, policy, ontology = _load_coco(coco_path), load_policy(policy_path), load_ontology(DEFAULT_ONTOLOGY)
    images = _unique(data["images"], "id", "image")
    annotations = _unique(data["annotations"], "id", "annotation")
    categories = _unique(data["categories"], "id", "category")
    names = [im.get("file_name") for im in data["images"]]
    if any(not isinstance(n, str) or not n for n in names) or len(names) != len(set(names)):
        raise ConversionError("image file_name values must be non-empty and unique")
    semantic_by_category, source_name_by_category = {}, {}
    for category_id, category in categories.items():
        name = category.get("name")
        canonical = policy["source_names"].get(name)
        if canonical is None:
            raise ConversionError(f"unknown source category {name!r} (COCO category id {category_id!r})")
        semantic_by_category[category_id] = ontology.by_name(canonical).id
        source_name_by_category[category_id] = name
    for image_id, image in images.items():
        if type(image.get("width")) is not int or type(image.get("height")) is not int or image["width"] <= 0 or image["height"] <= 0:
            raise ConversionError(f"image {image_id!r} has invalid dimensions")
    for annotation_id, annotation in annotations.items():
        if annotation.get("image_id") not in images:
            raise ConversionError(f"orphan annotation {annotation_id!r}: unknown image_id {annotation.get('image_id')!r}")
        if annotation.get("category_id") not in categories:
            raise ConversionError(f"annotation {annotation_id!r}: unknown category_id {annotation.get('category_id')!r}")
        _validate_segmentation(annotation.get("segmentation"), annotation_id)
    return data, semantic_by_category, source_name_by_category


def _validate_segmentation(segmentation: Any, annotation_id: Any) -> None:
    if not isinstance(segmentation, list) or not segmentation:
        kind = "RLE" if isinstance(segmentation, dict) else type(segmentation).__name__
        raise ConversionError(f"annotation {annotation_id!r}: unsupported or empty segmentation ({kind}); polygons required")
    for index, polygon in enumerate(segmentation):
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            raise ConversionError(f"annotation {annotation_id!r} polygon {index}: requires at least three x/y pairs")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in polygon):
            raise ConversionError(f"annotation {annotation_id!r} polygon {index}: coordinates must be finite numbers")


def polygon_mask(segmentation: list[list[float]], width: int, height: int) -> np.ndarray:
    """Rasterize by even-odd containment at pixel centres; polygon edges count inside."""
    result = np.zeros((height, width), dtype=bool)
    for flat in segmentation:
        points = np.asarray(flat, dtype=np.float64).reshape(-1, 2)
        x0 = max(0, int(math.floor(points[:, 0].min() - 0.5)))
        x1 = min(width, int(math.ceil(points[:, 0].max() - 0.5)) + 1)
        y0 = max(0, int(math.floor(points[:, 1].min() - 0.5)))
        y1 = min(height, int(math.ceil(points[:, 1].max() - 0.5)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        inside = np.zeros(xs.shape, dtype=bool)
        for current, previous in zip(points, np.roll(points, 1, axis=0)):
            xi, yi = current
            xj, yj = previous
            crosses = ((yi > ys) != (yj > ys)) & (xs <= (xj - xi) * (ys - yi) / (yj - yi + np.finfo(float).eps) + xi)
            inside ^= crosses
        result[y0:y1, x0:x1] |= inside
    return result


class ImageResolver:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ConversionError(f"images root is not a directory: {root}")
        self.files = [p for p in self.root.rglob("*") if p.is_file()]
        self.by_name: dict[str, list[Path]] = defaultdict(list)
        self.by_fold: dict[str, list[Path]] = defaultdict(list)
        for path in self.files:
            self.by_name[path.name].append(path)
            self.by_fold[path.name.casefold()].append(path)

    def resolve(self, coco_name: str) -> dict[str, Any]:
        basename = Path(coco_name).name
        canonical = HASH_PREFIX.sub("", basename, count=1)
        candidates = self.by_name[basename]
        normalized = False
        if not candidates and canonical != basename:
            candidates, normalized = self.by_name[canonical], True
        case_insensitive = False
        target = canonical if normalized else basename
        if not candidates:
            candidates = self.by_fold[target.casefold()]
            case_insensitive = bool(candidates)
        if len(candidates) != 1:
            reason = "missing" if not candidates else "ambiguous"
            raise ConversionError(f"{reason} image for COCO file {coco_name!r}: {len(candidates)} matches")
        return {"source_coco_file_name": coco_name, "canonical_file_name": canonical,
                "resolved_image_path": str(candidates[0].resolve()), "normalization_applied": normalized,
                "case_insensitive_match": case_insensitive, "path": candidates[0]}


def resolve_all(data: dict[str, Any], root: Path) -> dict[Any, dict[str, Any]]:
    canonical: dict[str, str] = {}
    result = {}
    resolver = ImageResolver(root)
    for image in data["images"]:
        name = image["file_name"]
        normalized = HASH_PREFIX.sub("", Path(name).name, count=1)
        if normalized in canonical and canonical[normalized] != name:
            raise ConversionError(f"COCO filename normalization collision: {canonical[normalized]!r} and {name!r} -> {normalized!r}")
        canonical[normalized] = name
        result[image["id"]] = resolver.resolve(name)
    return result


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("samples", value) if isinstance(value, dict) else value
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ConversionError("metadata must be a CSV or JSON array of objects")
    return rows


def _metadata_index(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id, by_name = {}, {}
    for row in rows:
        if not row.get("facade_id") or not row.get("split"):
            raise ConversionError("every metadata row requires non-empty facade_id and split")
        split = str(row["split"]).lower()
        split = "validation" if split == "val" else split
        if split not in {"train", "validation", "test"}:
            raise ConversionError(f"invalid metadata split {row['split']!r}")
        row = dict(row, split=split)
        if row.get("image_id") not in (None, ""):
            key = str(row["image_id"])
            if key in by_id: raise ConversionError(f"duplicate metadata image_id {key!r}")
            by_id[key] = row
        elif row.get("canonical_file_name"):
            key = str(row["canonical_file_name"])
            if key in by_name: raise ConversionError(f"duplicate metadata canonical_file_name {key!r}")
            by_name[key] = row
        else:
            raise ConversionError("metadata row requires image_id or canonical_file_name")
    return by_id, by_name


def _write_png(path: Path, array: np.ndarray, allowed: frozenset[int]) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(path, optimize=False, compress_level=9)
    with Image.open(path) as image:
        restored = np.asarray(image)
    if restored.dtype != np.uint8 or restored.shape != array.shape or not np.array_equal(restored, array) or not set(np.unique(restored)) <= allowed:
        raise ConversionError(f"written mask verification failed: {path}")


def convert(coco_path: Path, images_root: Path, metadata_path: Path, output_dir: Path,
            *, overwrite: bool = False, policy_path: Path = POLICY_PATH) -> dict[str, Any]:
    data, semantic_by_category, source_names = inspect_coco(coco_path, policy_path)
    policy, ontology = load_policy(policy_path), load_ontology(DEFAULT_ONTOLOGY)
    resolutions = resolve_all(data, images_root)
    by_id, by_name = _metadata_index(load_metadata(metadata_path))
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite: raise ConversionError(f"output directory is not empty: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(); (output_dir / "main_masks").mkdir(); (output_dir / "ornament_masks").mkdir()
    annotations: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]: annotations[annotation["image_id"]].append(annotation)
    priority = {ontology.by_name(name).id: rank for rank, name in enumerate(policy["main_priority_high_to_low"])}
    overlaps: dict[tuple[int, int], dict[str, Any]] = {}
    manifest = []
    source_hash = _hash(coco_path)
    facade_splits: dict[str, set[str]] = defaultdict(set)
    for image in sorted(data["images"], key=lambda item: str(item["id"])):
        image_id, width, height = image["id"], image["width"], image["height"]
        resolution = resolutions[image_id]
        metadata = by_id.get(str(image_id)) or by_name.get(resolution["canonical_file_name"])
        if metadata is None: raise ConversionError(f"no metadata for image_id {image_id!r} / {resolution['canonical_file_name']!r}")
        facade_splits[str(metadata["facade_id"])].add(metadata["split"])
        with Image.open(resolution["path"]) as source_image:
            source_image.load()
            if source_image.size != (width, height): raise ConversionError(f"image {image_id!r} dimensions {source_image.size} != COCO {(width, height)}")
        category_masks: dict[int, np.ndarray] = {}
        for annotation in annotations[image_id]:
            category_id = annotation["category_id"]
            decoded = polygon_mask(annotation["segmentation"], width, height)
            if decoded.shape != (height, width): raise ConversionError(f"annotation {annotation['id']!r}: decoded mask dimension mismatch")
            category_masks.setdefault(category_id, np.zeros((height, width), bool))
            category_masks[category_id] |= decoded
        ids = sorted(category_masks)
        for pos, left in enumerate(ids):
            for right in ids[pos + 1:]:
                count = int(np.count_nonzero(category_masks[left] & category_masks[right]))
                if not count: continue
                pair = (left, right)
                left_sem, right_sem = semantic_by_category[left], semantic_by_category[right]
                ornament = 8 in {left_sem, right_sem}
                winner = None if ornament else min((left_sem, right_sem), key=priority.get)
                record = overlaps.setdefault(pair, {"source_category_names": [source_names[left], source_names[right]],
                    "source_category_ids": [left, right], "semantic_ids": [left_sem, right_sem], "overlapping_pixel_count": 0,
                    "image_count": 0, "image_ids": [], "representative_filenames": [],
                    "overlap_type": "ornament + main allowed overlap" if ornament else "automatically resolved main + main overlap",
                    "applied_rule": "independent masks" if ornament else policy["version"],
                    "winning_main_class": None if winner is None else ontology.classes[winner].name,
                    "automatically_resolved_pixel_count": 0, "manual_review_count": 0})
                record["overlapping_pixel_count"] += count; record["image_count"] += 1
                record["image_ids"].append(image_id); record["representative_filenames"].append(image["file_name"])
                if not ornament: record["automatically_resolved_pixel_count"] += count
        main = np.zeros((height, width), np.uint8); ornament_mask = np.zeros((height, width), np.uint8)
        for category_id, mask in category_masks.items():
            if semantic_by_category[category_id] == 8: ornament_mask[mask] = 1
        for category_id in sorted(category_masks, key=lambda value: priority.get(semantic_by_category[value], -1), reverse=True):
            semantic = semantic_by_category[category_id]
            if semantic != 8: main[category_masks[category_id]] = semantic
        stem = f"{image_id}"
        image_out = output_dir / "images" / resolution["canonical_file_name"]
        if image_out.exists(): raise ConversionError(f"portable image name collision: {image_out.name!r}")
        shutil.copyfile(resolution["path"], image_out)
        main_out, ornament_out = output_dir / "main_masks" / f"{stem}.png", output_dir / "ornament_masks" / f"{stem}.png"
        _write_png(main_out, main, MAIN_ALLOWED); _write_png(ornament_out, ornament_mask, frozenset({0, 1, 255}))
        row = {"sample_id": str(image_id), "image_id": image_id, **{k: v for k, v in resolution.items() if k != "path"},
               "resolved_image_path": str(resolution["path"].resolve()), "image_path": image_out.relative_to(output_dir).as_posix(),
               "main_mask_path": main_out.relative_to(output_dir).as_posix(), "ornament_mask_path": ornament_out.relative_to(output_dir).as_posix(),
               "facade_id": str(metadata["facade_id"]), "building_id": str(metadata.get("building_id", "")), "split": metadata["split"],
               "schema_version": SCHEMA_VERSION, "ontology_version": V2_VERSION, "source_coco_sha256": source_hash,
               "source_annotation_ids": [a["id"] for a in annotations[image_id]], "width": width, "height": height}
        for field in ("capture_date", "capture_year"):
            if metadata.get(field) not in (None, ""): row[field] = metadata[field]
        manifest.append(row)
    leaking = {facade: sorted(splits) for facade, splits in facade_splits.items() if len(splits) > 1}
    if leaking: raise ConversionError(f"facade leakage across splits: {leaking}")
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest), encoding="utf-8")
    overlap_path = output_dir / "overlap_report.json"
    overlap_path.write_text(json.dumps({"policy_version": policy["version"], "overlaps": list(overlaps.values()),
        "unknown_categories": [], "malformed_annotations": []}, indent=2), encoding="utf-8")
    resolution_path = output_dir / "filename_resolution_report.json"
    resolution_path.write_text(json.dumps({"images": [{k: v for k, v in value.items() if k != "path"} for value in resolutions.values()]}, indent=2), encoding="utf-8")
    artifact_paths = ([manifest_path, overlap_path, resolution_path]
                      + sorted((output_dir / "images").iterdir())
                      + sorted((output_dir / "main_masks").iterdir())
                      + sorted((output_dir / "ornament_masks").iterdir()))
    summary = {"source_coco_sha256": source_hash, "ontology_version": ontology.version, "ontology_hash": ontology.hash,
        "priority_policy_version": policy["version"], "priority_policy_hash": _canonical_hash(policy),
        "input_counts": {k: len(data[k]) for k in ("images", "annotations", "categories")}, "output_counts": {"samples": len(manifest), "main_masks": len(manifest), "ornament_masks": len(manifest)},
        "filename_resolution_statistics": {"exact": sum(not r["normalization_applied"] and not r["case_insensitive_match"] for r in resolutions.values()),
            "normalized": sum(r["normalization_applied"] for r in resolutions.values()), "case_insensitive": sum(r["case_insensitive_match"] for r in resolutions.values())},
        "warnings": [], "failures": [], "artifacts": [{"path": p.relative_to(output_dir).as_posix(), "size_bytes": p.stat().st_size, "sha256": _hash(p)} for p in artifact_paths]}
    summary_path = output_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_manifest(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"sample_id", "image_id", "source_coco_file_name", "canonical_file_name", "resolved_image_path", "image_path",
                "main_mask_path", "ornament_mask_path", "facade_id", "building_id", "split", "schema_version", "ontology_version",
                "source_coco_sha256", "source_annotation_ids", "width", "height"}
    errors, facade_splits = [], defaultdict(set)
    for index, row in enumerate(rows, 1):
        missing = required - set(row)
        if missing: errors.append(f"row {index}: missing fields {sorted(missing)}"); continue
        if not row["facade_id"] or not row["split"]: errors.append(f"row {index}: facade_id and split must be non-empty")
        if row["split"] not in {"train", "validation", "test"}: errors.append(f"row {index}: invalid split {row['split']!r}")
        if row["schema_version"] != SCHEMA_VERSION or row["ontology_version"] != V2_VERSION:
            errors.append(f"row {index}: incompatible schema or ontology version")
        facade_splits[str(row["facade_id"])].add(row["split"])
        try:
            main = np.asarray(Image.open(path.parent / row["main_mask_path"])); ornament = np.asarray(Image.open(path.parent / row["ornament_mask_path"]))
            image = Image.open(path.parent / row["image_path"])
            if main.dtype != np.uint8 or ornament.dtype != np.uint8 or main.shape != ornament.shape or main.shape != (row["height"], row["width"]) or image.size != (row["width"], row["height"]): errors.append(f"row {index}: image/mask grid or dtype mismatch")
            if not set(np.unique(main)) <= MAIN_ALLOWED or not set(np.unique(ornament)) <= {0, 1, 255}: errors.append(f"row {index}: invalid mask value domain")
        except Exception as exc: errors.append(f"row {index}: unreadable artifact: {exc}")
    for facade, splits in facade_splits.items():
        if len(splits) > 1: errors.append(f"facade {facade!r} leaks across splits {sorted(splits)}")
    return {"valid": not errors, "sample_count": len(rows), "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit"); audit.add_argument("--coco", type=Path, required=True); audit.add_argument("--images-root", type=Path, required=True); audit.add_argument("--output", type=Path)
    conversion = sub.add_parser("convert"); conversion.add_argument("--coco", type=Path, required=True); conversion.add_argument("--images-root", type=Path, required=True); conversion.add_argument("--metadata", type=Path, required=True); conversion.add_argument("--output-dir", type=Path, required=True); conversion.add_argument("--overwrite", action="store_true")
    validation = sub.add_parser("validate"); validation.add_argument("--manifest", type=Path, required=True); validation.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            data, _, _ = inspect_coco(args.coco); resolutions = resolve_all(data, args.images_root)
            report = {"valid": True, "source_coco_sha256": _hash(args.coco), "counts": {k: len(data[k]) for k in ("images", "annotations", "categories")}, "filename_resolution": [{k: v for k, v in r.items() if k != "path"} for r in resolutions.values()]}
        elif args.command == "convert": report = convert(args.coco, args.images_root, args.metadata, args.output_dir, overwrite=args.overwrite)
        else: report = validate_manifest(args.manifest)
    except (ConversionError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)})); return 1
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, default=str)); return 0 if report.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
