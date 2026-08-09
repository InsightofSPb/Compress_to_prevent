"""Read-only, strict pre-training validation of facade segmentation masks."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .ontology import DEFAULT_ONTOLOGY, Ontology, load_ontology

MASK_COLUMNS = ("mask_path", "seg_map_path", "annotation", "mask", "label_path")

def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict): data = data.get("samples", data.get("items", data.get("data", [])))
    if not isinstance(data, list): raise ValueError(f"{path}: manifest must contain a list of samples")
    return [dict(x) for x in data]

def _resolve_source(source: str | Path) -> tuple[list[tuple[Path, str | None]], int, str]:
    path = Path(source)
    if path.is_dir():
        masks = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".png", ".tif", ".tiff", ".npy"})
        return [(p, None) for p in masks], len(masks), str(path)
    rows = _manifest_rows(path); result = []
    for index, row in enumerate(rows):
        key = next((k for k in MASK_COLUMNS if row.get(k)), None)
        if key is None: raise ValueError(f"{path}: row {index + 1} has no mask column {MASK_COLUMNS}")
        mask = Path(str(row[key])); mask = mask if mask.is_absolute() else path.parent / mask
        result.append((mask, str(row["facade_id"]) if row.get("facade_id") not in (None, "") else None))
    return result, len(rows), str(path)

def _read_mask(path: Path) -> np.ndarray:
    import numpy as np
    from PIL import Image
    if not path.exists(): raise FileNotFoundError(f"mask does not exist: {path}")
    arr = np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else np.asarray(Image.open(path))
    if arr.ndim != 2: raise ValueError(f"{path}: mask must be single-channel, got shape {arr.shape}")
    return arr

def validate_splits(sources: dict[str, str | Path], ontology: Ontology) -> dict[str, Any]:
    import numpy as np
    report: dict[str, Any] = {"ontology_version": ontology.version, "ontology_hash": ontology.hash,
        "ignore_index": ontology.ignore_index, "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": {k: str(v) for k,v in sources.items()}, "splits": {}, "warnings": [], "errors": []}
    facade_sets: dict[str, set[str]] = {}
    ads_splits = []
    for split, source in sources.items():
        counts, images_with = Counter(), Counter(); unknown_files = []; facades = set()
        try: entries, image_count, checked = _resolve_source(source)
        except Exception as exc:
            report["errors"].append(f"{split}: {exc}"); continue
        for path, facade_id in entries:
            if facade_id is not None: facades.add(facade_id)
            try:
                mask = _read_mask(path); found = {int(x) for x in np.unique(mask)}
                unknown = found - ontology.valid_ids - {ontology.ignore_index}
                if unknown:
                    unknown_files.append({"file": str(path), "ids": sorted(unknown)})
                    report["errors"].append(f"{path}: unknown mask IDs {sorted(unknown)}")
                for value, count in zip(*np.unique(mask, return_counts=True)):
                    counts[int(value)] += int(count); images_with[int(value)] += 1
            except Exception as exc: report["errors"].append(f"{path}: {exc}")
        total = sum(counts.values())
        valid_total = total - counts[ontology.ignore_index]
        missing = sorted(ontology.valid_ids - set(counts))
        if 11 in ontology.valid_ids and 11 not in counts:
            report["warnings"].append(f"{split}: ADVERTISEMENTS (ID 11) is absent")
        if counts[11]: ads_splits.append(split)
        report["splits"][split] = {"image_count": image_count, "mask_count": len(entries),
            "unique_ids": sorted(counts), "pixel_count": {str(i): counts[i] for i in sorted(counts)},
            "pixel_frequency": {str(i): (counts[i] / valid_total if valid_total and i != ontology.ignore_index else 0.0) for i in sorted(counts)},
            "images_with_class": {str(i): images_with[i] for i in sorted(counts)},
            "missing_classes": missing, "unknown_ids": sorted({i for x in unknown_files for i in x["ids"]}),
            "files_with_unknown_ids": unknown_files, "source": checked}
        facade_sets[split] = facades
    if ads_splits and len(ads_splits) != len(report["splits"]):
        report["warnings"].append(f"ADVERTISEMENTS occurs only in splits {ads_splits}")
    names = list(facade_sets)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            overlap = sorted(facade_sets[left] & facade_sets[right])
            if overlap: report["errors"].append(f"facade_id overlap between {left} and {right}: {overlap}")
    report["valid"] = not report["errors"]
    return report

def _dataset_config(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    splits = data.get("splits", data)
    result = {}
    for name in ("train", "val", "validation", "test"):
        if name in splits:
            value = splits[name]; value = value.get("manifest", value.get("mask_dir")) if isinstance(value, dict) else value
            p = Path(value); result["val" if name == "validation" else name] = str(p if p.is_absolute() else path.parent / p)
    return result

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    parser.add_argument("--dataset-config", type=Path, help="JSON/YAML-subset mapping split names to manifests or mask directories")
    for split in ("train", "val", "test"): parser.add_argument(f"--{split}", help=f"{split} manifest or mask directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="return nonzero for validation errors (errors are always reported)")
    args = parser.parse_args(argv); sources = _dataset_config(args.dataset_config) if args.dataset_config else {}
    sources.update({s: getattr(args, s) for s in ("train", "val", "test") if getattr(args, s)})
    if not sources: parser.error("provide --dataset-config or at least one split source")
    try: report = validate_splits(sources, load_ontology(args.ontology))
    except Exception as exc: report = {"valid": False, "errors": [str(exc)], "warnings": [], "sources": sources,
        "timestamp": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "errors": len(report["errors"]), "warnings": len(report["warnings"]), "output": str(args.output)}))
    return 1 if args.strict and not report["valid"] else 0

if __name__ == "__main__": raise SystemExit(main())
