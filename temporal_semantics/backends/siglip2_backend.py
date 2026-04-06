from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..io import load_image, save_json
from .base import BackendCapabilities, SemanticBackend


class SigLIP2Backend(SemanticBackend):
    name = "siglip2"

    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            backend=cls.name,
            model_id="siglip2_proxy_patch_v1",
            provides_dense_features=True,
            provides_global_features=True,
            provides_masks=False,
            provides_logits_or_probs=False,
            tile_compatible=True,
            expected_feature_grid_type="tile_grid",
            experimental=False,
            notes="SigLIP2-like dense patch proxy backend.",
        )

    def export_artifacts(self, sample: Dict[str, str], out_dir: Path, tile_size: int) -> Dict[str, object]:
        sample_id = sample["sample_id"]
        width, height, payload = load_image(Path(sample["image_path"]))

        features: List[Dict[str, object]] = []
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                channel_sum = [0.0, 0.0, 0.0]
                channel_sq = [0.0, 0.0, 0.0]
                count = 0
                for yy in range(y0, min(y0 + tile_size, height)):
                    for xx in range(x0, min(x0 + tile_size, width)):
                        idx = (yy * width + xx) * 3
                        vals = [payload[idx] / 255.0, payload[idx + 1] / 255.0, payload[idx + 2] / 255.0]
                        for c in range(3):
                            channel_sum[c] += vals[c]
                            channel_sq[c] += vals[c] * vals[c]
                        count += 1
                mean = [v / max(count, 1) for v in channel_sum]
                std = [((channel_sq[c] / max(count, 1)) - mean[c] ** 2) ** 0.5 for c in range(3)]
                features.append({"x": x0 // tile_size, "y": y0 // tile_size, "vec": mean + std})

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
