from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..io import load_rgb_image, read_csv_rows, write_csv_rows
from ..residuals import load_valid_mask
from .model import ByteNGramEntropyModel, ByteUnigramEntropyModel


def _valid_pixel_mask(row: Dict[str, str], width: int, height: int) -> Optional[bytes]:
    path_value = row.get("valid_mask_path", "")
    if not path_value:
        return None
    threshold = int(float(row.get("valid_threshold", "0") or 0))
    return load_valid_mask(Path(path_value), (width, height), threshold=threshold)


def _filter_valid_rgb_payload(payload: bytes, valid_mask: Optional[bytes]) -> bytes:
    if valid_mask is None:
        return payload
    output = bytearray()
    for pixel_index, is_valid in enumerate(valid_mask):
        if is_valid:
            base = pixel_index * 3
            output.extend(payload[base:base + 3])
    return bytes(output)


def train_entropy_model(
    residual_manifest: Path,
    model_out: Path,
    split: str = "train",
    model_mode: str = "bigram",
    alpha: float = 1.0,
) -> Dict[str, object]:
    rows = read_csv_rows(residual_manifest)
    selected = [row for row in rows if row.get("split", "train") == split]
    payloads: List[bytes] = []
    total_valid_symbols = 0
    for row in selected:
        width, height, payload = load_rgb_image(Path(row["residual_path"]))
        valid_payload = _filter_valid_rgb_payload(payload, _valid_pixel_mask(row, width, height))
        payloads.append(valid_payload)
        total_valid_symbols += len(valid_payload)

    if model_mode == "unigram":
        model = ByteUnigramEntropyModel(alpha=alpha)
    elif model_mode == "bigram":
        model = ByteNGramEntropyModel(order=1, alpha=alpha)
    else:
        raise ValueError(f"Unsupported model_mode: {model_mode}")

    model.fit(payloads)
    model.save(model_out)
    return {
        "n_samples": len(selected),
        "model_mode": model_mode,
        "train_split": split,
        "alpha": alpha,
        "total_valid_symbols": total_valid_symbols,
        "valid_pixel_policy": "only_valid_aligned_pixels_used",
    }


def _load_model(path: Path) -> ByteNGramEntropyModel:
    model = ByteNGramEntropyModel.load(path)
    return model


def _tile_valid_payload(
    width: int,
    height: int,
    payload: bytes,
    valid_mask: Optional[bytes],
    x0: int,
    y0: int,
    tile_size: int,
) -> bytes:
    output = bytearray()
    for yy in range(y0, min(y0 + tile_size, height)):
        for xx in range(x0, min(x0 + tile_size, width)):
            pixel_index = yy * width + xx
            if valid_mask is not None and not valid_mask[pixel_index]:
                continue
            base = pixel_index * 3
            output.extend(payload[base:base + 3])
    return bytes(output)


def _tile_model_bits(
    model: ByteNGramEntropyModel,
    width: int,
    height: int,
    payload: bytes,
    valid_mask: Optional[bytes],
    tile_size: int,
    min_valid_ratio: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            content_width = min(tile_size, width - x0)
            content_height = min(tile_size, height - y0)
            tile_pixels = content_width * content_height
            valid_payload = _tile_valid_payload(width, height, payload, valid_mask, x0, y0, tile_size)
            valid_pixels = len(valid_payload) // 3
            valid_ratio = valid_pixels / max(tile_pixels, 1)
            if valid_pixels == 0 or valid_ratio < min_valid_ratio:
                continue
            bit_length, _ = model.nll_bits_with_components(valid_payload)
            rows.append(
                {
                    "tile_x": x0 // tile_size,
                    "tile_y": y0 // tile_size,
                    "bit_length": bit_length,
                    "bits_per_symbol": bit_length / max(len(valid_payload), 1),
                    "valid_pixel_count": valid_pixels,
                    "valid_ratio": valid_ratio,
                }
            )
    return rows


def eval_entropy_model(
    residual_manifest: Path,
    model_path: Path,
    split: str,
    out_csv: Path,
    tile_size: int | None = None,
    tile_out_csv: Path | None = None,
    min_valid_ratio: float = 0.50,
) -> Dict[str, object]:
    if not 0.0 <= min_valid_ratio <= 1.0:
        raise ValueError("min_valid_ratio must be in [0, 1]")
    model = _load_model(model_path)
    rows = read_csv_rows(residual_manifest)
    selected = [row for row in rows if row.get("split", "train") == split]

    sample_rows: List[Dict[str, object]] = []
    tile_rows: List[Dict[str, object]] = []

    for row in selected:
        width, height, payload = load_rgb_image(Path(row["residual_path"]))
        valid_mask = _valid_pixel_mask(row, width, height)
        valid_payload = _filter_valid_rgb_payload(payload, valid_mask)
        total_bits, _ = model.nll_bits_with_components(valid_payload)
        sample_rows.append(
            {
                "pair_id": row["pair_id"],
                "split": split,
                "score_type": "model_bits",
                "bit_length": total_bits,
                "num_symbols": len(valid_payload),
                "bits_per_symbol": total_bits / max(len(valid_payload), 1),
                "valid_pixel_count": len(valid_payload) // 3,
                "valid_ratio": row.get("valid_ratio", ""),
            }
        )

        if tile_size is not None and tile_out_csv is not None:
            for tile in _tile_model_bits(
                model, width, height, payload, valid_mask,
                tile_size=tile_size, min_valid_ratio=min_valid_ratio,
            ):
                tile_rows.append(
                    {
                        "pair_id": row["pair_id"],
                        "split": split,
                        "score_type": "model_bits",
                        "tile_x": tile["tile_x"],
                        "tile_y": tile["tile_y"],
                        "bit_length": tile["bit_length"],
                        "bits_per_symbol": tile["bits_per_symbol"],
                        "tile_size": tile_size,
                        "valid_pixel_count": tile["valid_pixel_count"],
                        "valid_ratio": tile["valid_ratio"],
                    }
                )

    write_csv_rows(
        out_csv,
        ["pair_id", "split", "score_type", "bit_length", "num_symbols", "bits_per_symbol", "valid_pixel_count", "valid_ratio"],
        sample_rows,
    )

    if tile_size is not None and tile_out_csv is not None:
        write_csv_rows(
            tile_out_csv,
            ["pair_id", "split", "score_type", "tile_x", "tile_y", "bit_length", "bits_per_symbol", "tile_size", "valid_pixel_count", "valid_ratio"],
            tile_rows,
        )

    mean_bps = sum(float(r["bits_per_symbol"]) for r in sample_rows) / max(len(sample_rows), 1)
    return {
        "n_samples": len(sample_rows),
        "split": split,
        "mean_bits_per_symbol": mean_bps,
        "n_tile_scores": len(tile_rows),
        "min_valid_ratio": min_valid_ratio,
        "valid_pixel_policy": "only_valid_aligned_pixels_used",
    }


def merge_entropy_scores(inputs: List[Path], out_csv: Path) -> int:
    merged: Dict[str, Dict[str, float | str]] = {}
    for path in inputs:
        for row in read_csv_rows(path):
            key = row["pair_id"]
            merged.setdefault(key, {"bit_length": 0.0, "num_symbols": 0.0, "split": row.get("split", "")})
            merged[key]["bit_length"] = float(merged[key]["bit_length"]) + float(row["bit_length"])
            merged[key]["num_symbols"] = float(merged[key]["num_symbols"]) + float(row["num_symbols"])
            merged[key]["split"] = row.get("split", str(merged[key]["split"]))

    out_rows = []
    for pair_id, payload in merged.items():
        bit_length = float(payload["bit_length"])
        num_symbols = float(payload["num_symbols"])
        out_rows.append(
            {
                "pair_id": pair_id,
                "split": payload["split"],
                "score_type": "model_bits",
                "bit_length": bit_length,
                "num_symbols": num_symbols,
                "bits_per_symbol": bit_length / max(num_symbols, 1.0),
            }
        )

    write_csv_rows(
        out_csv,
        ["pair_id", "split", "score_type", "bit_length", "num_symbols", "bits_per_symbol"],
        out_rows,
    )
    return len(out_rows)
