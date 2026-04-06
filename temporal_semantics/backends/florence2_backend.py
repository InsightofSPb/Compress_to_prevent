from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..io import save_json
from .base import BackendCapabilities, SemanticBackend


class Florence2Backend(SemanticBackend):
    name = "florence2"

    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            backend=cls.name,
            model_id="florence2_experimental_scaffold_v1",
            provides_dense_features=False,
            provides_global_features=False,
            provides_masks=False,
            provides_logits_or_probs=False,
            tile_compatible=False,
            expected_feature_grid_type="none",
            experimental=True,
            notes="Experimental scaffold only; dense extraction not yet implemented.",
        )

    def export_artifacts(self, sample: Dict[str, str], out_dir: Path, tile_size: int) -> Dict[str, object]:
        sample_id = sample["sample_id"]
        backend_dir = out_dir / self.name
        note_path = backend_dir / f"{sample_id}_experimental.json"
        save_json(note_path, {"status": "experimental", "reason": self.capabilities().notes})
        return {
            "mask_path": "",
            "probs_path": "",
            "features_path": "",
            "overlay_path": "",
            "feature_grid_h": 0,
            "feature_grid_w": 0,
            "image_width": 0,
            "image_height": 0,
            "status": "experimental",
            "notes": self.capabilities().notes,
        }
