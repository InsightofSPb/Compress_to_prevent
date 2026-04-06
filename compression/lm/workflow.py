from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..io import load_rgb_image, read_csv_rows, write_csv_rows
from .model import ByteNGramEntropyModel, ByteUnigramEntropyModel


def train_entropy_model(
    residual_manifest: Path,
    model_out: Path,
    split: str = "train",
    model_mode: str = "bigram",
    alpha: float = 1.0,
) -> Dict[str, object]:
    rows = read_csv_rows(residual_manifest)
    selected = [row for row in rows if row.get("split", "train") == split]
    payloads = [load_rgb_image(Path(row["residual_path"]))[2] for row in selected]

    if model_mode == "unigram":
        model = ByteUnigramEntropyModel(alpha=alpha)
    elif model_mode == "bigram":
        model = ByteNGramEntropyModel(order=1, alpha=alpha)
    else:
        raise ValueError(f"Unsupported model_mode: {model_mode}")

    model.fit(payloads)
    model.save(model_out)
    return {"n_samples": len(selected), "model_mode": model_mode, "train_split": split, "alpha": alpha}


def _load_model(path: Path) -> ByteNGramEntropyModel:
    model = ByteNGramEntropyModel.load(path)
    return model


def _tile_from_symbol_bits(width: int, height: int, per_symbol_bits: List[float], tile_size: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            acc = 0.0
            count = 0
            for yy in range(y0, min(y0 + tile_size, height)):
                for xx in range(x0, min(x0 + tile_size, width)):
                    base = (yy * width + xx) * 3
                    for c in range(3):
                        acc += per_symbol_bits[base + c]
                        count += 1
            rows.append(
                {
                    "tile_x": x0 // tile_size,
                    "tile_y": y0 // tile_size,
                    "bit_length": acc,
                    "bits_per_symbol": acc / max(count, 1),
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
) -> Dict[str, object]:
    model = _load_model(model_path)
    rows = read_csv_rows(residual_manifest)
    selected = [row for row in rows if row.get("split", "train") == split]

    sample_rows: List[Dict[str, object]] = []
    tile_rows: List[Dict[str, object]] = []

    for row in selected:
        width, height, payload = load_rgb_image(Path(row["residual_path"]))
        total_bits, per_symbol_bits = model.nll_bits_with_components(payload)
        sample_rows.append(
            {
                "pair_id": row["pair_id"],
                "split": split,
                "score_type": "model_bits",
                "bit_length": total_bits,
                "num_symbols": len(payload),
                "bits_per_symbol": total_bits / max(len(payload), 1),
            }
        )

        if tile_size is not None and tile_out_csv is not None:
            for tile in _tile_from_symbol_bits(width, height, per_symbol_bits, tile_size=tile_size):
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
                    }
                )

    write_csv_rows(
        out_csv,
        ["pair_id", "split", "score_type", "bit_length", "num_symbols", "bits_per_symbol"],
        sample_rows,
    )

    if tile_size is not None and tile_out_csv is not None:
        write_csv_rows(
            tile_out_csv,
            ["pair_id", "split", "score_type", "tile_x", "tile_y", "bit_length", "bits_per_symbol", "tile_size"],
            tile_rows,
        )

    mean_bps = sum(float(r["bits_per_symbol"]) for r in sample_rows) / max(len(sample_rows), 1)
    return {"n_samples": len(sample_rows), "split": split, "mean_bits_per_symbol": mean_bps}


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
