from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from .backends import default_registry
from .io import load_manifest, write_index


def export_semantic_artifacts(
    manifest_csv: Path,
    backends: Iterable[str],
    out_dir: Path,
    tile_size: int,
    force: bool = False,
) -> List[Dict[str, object]]:
    manifest = load_manifest(manifest_csv)
    registry = default_registry()
    rows: List[Dict[str, object]] = []

    for sample in manifest:
        sample_id = sample.get("sample_id") or Path(sample["image_path"]).stem
        split = sample.get("split", "train")
        sample = dict(sample)
        sample["sample_id"] = sample_id

        for backend_name in backends:
            backend = registry.create(backend_name)
            backend_out = out_dir / "artifacts"
            features_hint = backend_out / backend_name / f"{sample_id}_features.json"
            if features_hint.exists() and not force:
                status = "cached"
                payload = {
                    "mask_path": str((backend_out / backend_name / f"{sample_id}_mask.pgm")),
                    "probs_path": str((backend_out / backend_name / f"{sample_id}_probs.json")),
                    "features_path": str(features_hint),
                    "overlay_path": str((backend_out / backend_name / f"{sample_id}_overlay.ppm")),
                    "feature_grid_h": 0,
                    "feature_grid_w": 0,
                    "notes": "reused_cached_artifacts",
                }
            else:
                payload = backend.export_artifacts(sample, backend_out, tile_size=tile_size)
                status = str(payload.get("status", "ok"))

            rows.append(
                {
                    "sample_id": sample_id,
                    "backend": backend_name,
                    "image_path": sample["image_path"],
                    "mask_path": payload.get("mask_path", ""),
                    "probs_path": payload.get("probs_path", ""),
                    "features_path": payload.get("features_path", ""),
                    "overlay_path": payload.get("overlay_path", ""),
                    "feature_grid_h": payload.get("feature_grid_h", 0),
                    "feature_grid_w": payload.get("feature_grid_w", 0),
                    "split": split,
                    "status": status,
                    "notes": payload.get("notes", ""),
                }
            )

    write_index(out_dir / "artifact_index.csv", rows)
    return rows
