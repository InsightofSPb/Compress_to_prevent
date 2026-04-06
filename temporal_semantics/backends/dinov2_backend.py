from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..io import load_image, save_json
from .base import BackendCapabilities, SemanticBackend


class DINOv2Backend(SemanticBackend):
    name = "dinov2"

    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            backend=cls.name,
            model_id="dinov2_proxy_patch_v1",
            provides_dense_features=True,
            provides_global_features=False,
            provides_masks=False,
            provides_logits_or_probs=False,
            tile_compatible=True,
            expected_feature_grid_type="tile_grid",
            experimental=False,
            notes="Standalone dense feature baseline compatible with tile extraction.",
        )

    def export_artifacts(self, sample: Dict[str, str], out_dir: Path, tile_size: int) -> Dict[str, object]:
        sample_id = sample["sample_id"]
        width, height, payload = load_image(Path(sample["image_path"]))

        features: List[Dict[str, object]] = []
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                vals_r, vals_g, vals_b = [], [], []
                for yy in range(y0, min(y0 + tile_size, height)):
                    for xx in range(x0, min(x0 + tile_size, width)):
                        idx = (yy * width + xx) * 3
                        vals_r.append(payload[idx] / 255.0)
                        vals_g.append(payload[idx + 1] / 255.0)
                        vals_b.append(payload[idx + 2] / 255.0)
                mean_r = sum(vals_r) / max(len(vals_r), 1)
                mean_g = sum(vals_g) / max(len(vals_g), 1)
                mean_b = sum(vals_b) / max(len(vals_b), 1)
                var = lambda arr, m: sum((v - m) ** 2 for v in arr) / max(len(arr), 1)
                features.append({"x": x0 // tile_size, "y": y0 // tile_size, "vec": [mean_r, mean_g, mean_b, var(vals_r, mean_r), var(vals_g, mean_g), var(vals_b, mean_b)]})

        backend_dir = out_dir / self.name
        features_path = backend_dir / f"{sample_id}_features.json"
        save_json(features_path, {"grid_h": (height + tile_size - 1) // tile_size, "grid_w": (width + tile_size - 1) // tile_size, "tile_size": tile_size, "features": features, "extractor": self.capabilities().model_id})

        return {
            "mask_path": "",
            "probs_path": "",
            "features_path": str(features_path),
            "overlay_path": "",
            "feature_grid_h": (height + tile_size - 1) // tile_size,
            "feature_grid_w": (width + tile_size - 1) // tile_size,
            "image_width": width,
            "image_height": height,
            "status": "ok",
            "notes": self.capabilities().notes,
        }
