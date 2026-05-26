from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..io import load_image, save_json
from .base import SemanticBackend


class CLIPBackend(SemanticBackend):
    name = "clip"

    def export_artifacts(self, sample: Dict[str, str], out_dir: Path, tile_size: int) -> Dict[str, object]:
        sample_id = sample["sample_id"]
        width, height, payload = load_image(Path(sample["image_path"]))

        features: List[Dict[str, object]] = []
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                luma, rg, bg = [], [], []
                edge = 0.0
                for yy in range(y0, min(y0 + tile_size, height)):
                    for xx in range(x0, min(x0 + tile_size, width)):
                        idx = (yy * width + xx) * 3
                        r, g, b = payload[idx], payload[idx + 1], payload[idx + 2]
                        y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
                        luma.append(y)
                        rg.append((r - g) / 255.0)
                        bg.append((b - g) / 255.0)
                        if xx + 1 < width:
                            idx2 = (yy * width + (xx + 1)) * 3
                            edge += abs(payload[idx] - payload[idx2]) / 255.0
                n = max(len(luma), 1)
                features.append(
                    {
                        "x": x0 // tile_size,
                        "y": y0 // tile_size,
                        "vec": [sum(luma) / n, sum(rg) / n, sum(bg) / n, edge / n],
                    }
                )

        backend_dir = out_dir / self.name
        features_path = backend_dir / f"{sample_id}_features.json"
        save_json(features_path, {"grid_h": (height + tile_size - 1) // tile_size, "grid_w": (width + tile_size - 1) // tile_size, "tile_size": tile_size, "features": features, "extractor": "clip_proxy_dense_features"})

        return {
            "mask_path": "",
            "probs_path": "",
            "features_path": str(features_path),
            "overlay_path": "",
            "feature_grid_h": (height + tile_size - 1) // tile_size,
            "feature_grid_w": (width + tile_size - 1) // tile_size,
            "status": "ok",
            "notes": "clip_proxy_dense_features",
        }
