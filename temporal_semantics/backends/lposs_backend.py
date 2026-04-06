from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..io import load_image, make_overlay, save_json, save_mask_as_pgm, save_overlay
from .base import SemanticBackend


def _pixel_probs(r: int, g: int, b: int) -> List[float]:
    total = max(r + g + b, 1)
    p0 = r / total
    p1 = g / total
    p2 = b / total
    p3 = max(0.0, 1.0 - (p0 + p1 + p2))
    return [p0, p1, p2, p3]


class LPOSSBackend(SemanticBackend):
    name = "lposs"

    def export_artifacts(self, sample: Dict[str, str], out_dir: Path, tile_size: int) -> Dict[str, object]:
        sample_id = sample["sample_id"]
        image_path = Path(sample["image_path"])
        width, height, payload = load_image(image_path)

        mask_vals: List[int] = []
        probs: List[List[float]] = []
        for i in range(0, len(payload), 3):
            r, g, b = payload[i], payload[i + 1], payload[i + 2]
            probs_px = _pixel_probs(r, g, b)
            probs.append(probs_px)
            cls = max(range(len(probs_px)), key=lambda idx: probs_px[idx])
            mask_vals.append(cls)

        backend_dir = out_dir / self.name
        mask_path = backend_dir / f"{sample_id}_mask.pgm"
        probs_path = backend_dir / f"{sample_id}_probs.json"
        features_path = backend_dir / f"{sample_id}_features.json"
        overlay_path = backend_dir / f"{sample_id}_overlay.ppm"

        save_mask_as_pgm(mask_path, width, height, mask_vals)
        save_json(probs_path, {"width": width, "height": height, "n_classes": 4, "probs": probs})

        # LPOSS-like patch feature summary (deterministic lightweight proxy).
        features = []
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                acc = [0.0, 0.0, 0.0, 0.0]
                count = 0
                for yy in range(y0, min(y0 + tile_size, height)):
                    for xx in range(x0, min(x0 + tile_size, width)):
                        idx = yy * width + xx
                        px = probs[idx]
                        for c in range(4):
                            acc[c] += px[c]
                        count += 1
                features.append({"x": x0 // tile_size, "y": y0 // tile_size, "vec": [v / max(count, 1) for v in acc]})

        save_json(features_path, {"grid_h": (height + tile_size - 1) // tile_size, "grid_w": (width + tile_size - 1) // tile_size, "tile_size": tile_size, "features": features})

        ow, oh, overlay = make_overlay(mask_vals, width, height)
        save_overlay(overlay_path, ow, oh, overlay)

        return {
            "mask_path": str(mask_path),
            "probs_path": str(probs_path),
            "features_path": str(features_path),
            "overlay_path": str(overlay_path),
            "feature_grid_h": (height + tile_size - 1) // tile_size,
            "feature_grid_w": (width + tile_size - 1) // tile_size,
            "status": "ok",
            "notes": "lposs_proxy_from_rgb",
        }
