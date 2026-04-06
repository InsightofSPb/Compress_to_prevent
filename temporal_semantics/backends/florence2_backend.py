from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..io import save_json
from .base import SemanticBackend


class Florence2Backend(SemanticBackend):
    name = "florence2"

    def export_artifacts(self, sample: Dict[str, str], out_dir: Path, tile_size: int) -> Dict[str, object]:
        sample_id = sample["sample_id"]
        backend_dir = out_dir / self.name
        note_path = backend_dir / f"{sample_id}_experimental.json"
        save_json(note_path, {"status": "experimental", "reason": "Stable dense Florence-2 extraction is not implemented in this lightweight branch."})
        return {
            "mask_path": "",
            "probs_path": "",
            "features_path": "",
            "overlay_path": "",
            "feature_grid_h": 0,
            "feature_grid_w": 0,
            "status": "experimental",
            "notes": "florence2_scaffold_only",
        }
