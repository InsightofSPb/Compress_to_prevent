#!/usr/bin/env python3
"""Render paper-style temporal qualitative grids and visual baseline ablations.

The compact paper-style layout follows the qualitative organisation used in the
facade monitoring manuscript, but makes the temporal signal interpretable on
the facade itself:

    previous RGB | current RGB
    previous semantic mask | current semantic mask
    semantic change map | temporal heatmap overlay on current RGB
    embedded semantic-class legend and primary score colour bar

The script supports two semantic contexts with identical RGB-derived score
maps: manual/GT masks and model-predicted masks. In the RGB-residual
formulation, semantic masks are an interpretation/reference branch; the
compression-derived and image-baseline heatmaps do not depend on whether GT or
predicted semantic masks are displayed.

A second, wider ablation layout renders all supplied temporal score methods for
the same pair:

    previous RGB | current RGB | semantic change | method-1 heatmap | ...
    previous mask| current mask| change overlay  | method-1 overlay | ...

Scores are converted to heatmaps on the current-image coordinate system by
averaging scores covering each pixel. Robust normalisation defaults to one
shared low/high range per method over the selected pairs; it is saved in JSON
so generated figures are reproducible.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm.auto import tqdm


DEFAULT_PALETTE_RGB = [
    [0, 0, 0], [229, 57, 53], [30, 136, 229], [67, 160, 71],
    [251, 140, 0], [142, 36, 170], [253, 216, 53], [0, 172, 193],
    [158, 158, 158], [78, 158, 158], [142, 126, 71],
]
DEFAULT_CLASS_NAMES = [
    "BACKGROUND", "CRACK", "SPALLING", "DELAMINATION", "MISSING_ELEMENT",
    "WATER_STAIN", "EFFLORESCENCE", "CORROSION", "ORNAMENT_INTACT",
    "REPAIRS", "TEXT_OR_IMAGES",
]
REFERENCE_LABELS = {
    "inspection_relevant_change": "inspection-relevant change",
    "damage_or_repair_change": "damage / repair change",
    "damage_type_change": "damage-type change",
    "damage_presence_change": "damage-presence change",
    "intervention_or_content_change": "intervention / content change",
    "any_semantic_change": "any semantic change",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render temporal facade qualitative figures and method ablations.")
    parser.add_argument("--pairs-manifest", type=Path, required=True,
                        help="Aligned RGB temporal pair manifest containing image and valid-mask paths.")
    parser.add_argument("--semantic-manifest", type=Path, required=True,
                        help="Manual or predicted change-map manifest with semantic mask/change paths.")
    parser.add_argument("--semantic-source", choices=("gt", "predicted"), required=True)
    parser.add_argument("--semantic-label", default=None,
                        help="Figure title label; default is derived from semantic-source.")
    parser.add_argument("--reference", default="inspection_relevant_change",
                        choices=tuple(REFERENCE_LABELS))
    parser.add_argument(
        "--score-source", action="append", required=True,
        help=("Temporal score source as LABEL=CSV or LABEL=CSV#RAW_METHOD. The second form selects one method "
              "from a multi-method CSV, e.g. SSIM=/path/basic_tile_scores.csv#ssim_change."),
    )
    parser.add_argument("--primary-method", default="RGB/MSDZip",
                        help="Method label used in paper-style two-column figure.")
    parser.add_argument("--split", default="test", help="Pair split to render; default: test.")
    parser.add_argument("--pair-id", action="append", default=None,
                        help="Specific pair_id to render; repeat. Without this, all selected split pairs are rendered.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Optional limit after pair selection; useful for quick inspection.")
    parser.add_argument("--sort-by", choices=("manifest", "primary_score", "semantic_change_ratio"), default="manifest")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--normalization", choices=("global", "per_pair"), default="global")
    parser.add_argument("--low-percentile", type=float, default=5.0)
    parser.add_argument("--high-percentile", type=float, default=95.0)
    parser.add_argument("--invalid-label", type=int, default=255)
    parser.add_argument("--valid-threshold", type=int, default=0)
    parser.add_argument("--cell-width", type=int, default=460)
    parser.add_argument("--title-height", type=int, default=42)
    parser.add_argument("--overlay-alpha", type=float, default=0.46)
    parser.add_argument("--paper-primary-panel", choices=("overlay", "heatmap"), default="overlay",
                        help="Render the primary temporal score as an interpretable RGB overlay by default.")
    parser.add_argument("--no-embedded-legend", action="store_true",
                        help="Do not append the semantic legend and primary heatmap colour bar to paper figures.")
    parser.add_argument("--no-paper-style", action="store_true")
    parser.add_argument("--no-ablation-style", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{str(k): (v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError("Could not read image: {}".format(path))
    return image


def score_sources(specifications: Sequence[str], split: str) -> Tuple[Dict[str, Dict[str, List[Dict[str, str]]]], Dict[str, str]]:
    by_method: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    paths: Dict[str, str] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("Score source must use LABEL=CSV or LABEL=CSV#RAW_METHOD: {}".format(specification))
        label, value = specification.split("=", 1)
        label = label.strip()
        raw_method: Optional[str] = None
        if "#" in value:
            value, raw_method = value.rsplit("#", 1)
            raw_method = raw_method.strip()
        path = Path(value.strip())
        if not path.is_file():
            raise FileNotFoundError("Score source not found: {}".format(path))
        rows = [row for row in read_csv(path) if row.get("split", "") == split]
        if raw_method:
            rows = [row for row in rows if row.get("method", "") == raw_method]
        detected = sorted({row.get("method", "") for row in rows})
        if not rows:
            raise ValueError("No rows for method label {} in split {} from {}".format(label, split, path))
        if not raw_method and len(detected) != 1:
            raise ValueError("Source {} contains multiple methods {}; select one with #RAW_METHOD".format(path, detected))
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["pair_id"], []).append(row)
        by_method[label] = grouped
        paths[label] = str(path)
    return by_method, paths


def resolve_semantic_paths(row: Dict[str, str], source: str, reference: str) -> Tuple[Path, Path, Path]:
    if source == "gt":
        previous = row.get("prev_manual_mask_aligned_path", "")
        current = row.get("curr_manual_mask_path", "")
    else:
        previous = row.get("prev_predicted_mask_aligned_path", "")
        current = row.get("curr_predicted_mask_in_current_coordinates_path", row.get("curr_predicted_mask_path", ""))
    change = row.get(reference + "_path", "")
    if not previous or not current or not change:
        raise KeyError("Semantic manifest row for {} lacks mask/change paths for {} source".format(row.get("pair_id", ""), source))
    return Path(previous), Path(current), Path(change)


def semantic_colour(mask: np.ndarray, invalid_label: int) -> np.ndarray:
    out = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for idx, rgb in enumerate(DEFAULT_PALETTE_RGB):
        out[mask == idx] = np.asarray(rgb[::-1], dtype=np.uint8)
    out[mask == invalid_label] = (40, 40, 40)
    return out


def change_colour(change: np.ndarray, invalid_label: int) -> np.ndarray:
    out = np.zeros((*change.shape[:2], 3), dtype=np.uint8)
    out[change == 1] = (0, 0, 238)
    out[change == invalid_label] = (55, 55, 55)
    return out


def change_overlay(current: np.ndarray, change: np.ndarray, invalid_label: int, alpha: float) -> np.ndarray:
    colour = change_colour(change, invalid_label)
    out = current.copy()
    valid_change = change == 1
    blended = cv2.addWeighted(current, 1.0 - alpha, colour, alpha, 0.0)
    out[valid_change] = blended[valid_change]
    out[change == invalid_label] = (45, 45, 45)
    return out


def score_map(rows: Sequence[Dict[str, str]], shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape
    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for row in rows:
        tile_size = int(float(row["tile_size"]))
        if row.get("tile_origin_x", "") and row.get("tile_origin_y", ""):
            x0, y0 = int(float(row["tile_origin_x"])), int(float(row["tile_origin_y"]))
        else:
            x0, y0 = int(float(row["tile_x"])) * tile_size, int(float(row["tile_y"])) * tile_size
        x1, y1 = min(width, x0 + tile_size), min(height, y0 + tile_size)
        total[y0:y1, x0:x1] += float(row["tile_score"])
        count[y0:y1, x0:x1] += 1.0
    output = np.zeros((height, width), dtype=np.float32)
    valid = count > 0
    output[valid] = total[valid] / count[valid]
    return output, valid


def method_range(method_rows: Dict[str, List[Dict[str, str]]], selected_pairs: Sequence[str], low: float, high: float) -> Tuple[float, float]:
    values: List[float] = []
    for pair_id in selected_pairs:
        for row in method_rows.get(pair_id, []):
            values.append(float(row["tile_score"]))
    if not values:
        return 0.0, 1.0
    lo, hi = np.percentile(np.asarray(values, dtype=np.float32), [low, high]).tolist()
    if float(hi) <= float(lo):
        hi = float(lo) + 1e-8
    return float(lo), float(hi)


def heatmap_visual(score: np.ndarray, score_valid: np.ndarray, valid: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
    lo, hi = limits
    norm = np.zeros_like(score, dtype=np.float32)
    combined = score_valid & valid
    norm[combined] = np.clip((score[combined] - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colour[~combined] = (32, 32, 32)
    return colour


def heatmap_overlay(current: np.ndarray, heatmap: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    out = cv2.addWeighted(current, 1.0 - alpha, heatmap, alpha, 0.0)
    out[~valid] = (45, 45, 45)
    return out


def fit_panel(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    scale = width / float(image.shape[1])
    return cv2.resize(image, (width, max(1, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)


def fitted_scale(text: str, max_width: int, preferred: float, thickness: int, minimum: float = 0.32) -> float:
    scale = preferred
    while scale > minimum:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if width <= max_width:
            return scale
        scale -= 0.03
    return minimum


def put_fitted_text(canvas: np.ndarray, text: str, position: Tuple[int, int], max_width: int,
                    preferred: float = 0.62, thickness: int = 1, colour: Tuple[int, int, int] = (25, 25, 25)) -> None:
    scale = fitted_scale(text, max_width, preferred, thickness)
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def labelled_panel(image: np.ndarray, title: str, width: int, title_height: int) -> np.ndarray:
    image = fit_panel(image, width)
    band = np.full((title_height, width, 3), 248, dtype=np.uint8)
    put_fitted_text(band, title, (12, int(title_height * 0.68)), width - 24, preferred=0.62, thickness=1)
    return np.concatenate([band, image], axis=0)


def equalise_cells(cells: Sequence[np.ndarray]) -> List[np.ndarray]:
    target_h = max(cell.shape[0] for cell in cells)
    results: List[np.ndarray] = []
    for cell in cells:
        if cell.shape[0] < target_h:
            padding = np.full((target_h - cell.shape[0], cell.shape[1], 3), 255, dtype=np.uint8)
            cell = np.concatenate([cell, padding], axis=0)
        results.append(cell)
    return results


def row_canvas(cells: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(equalise_cells(cells), axis=1)


def add_header(canvas: np.ndarray, facade_id: str, year_prev: str, year_curr: str,
               semantic_label: str, reference: str) -> np.ndarray:
    """Use a two-line compact header so long facade IDs cannot be clipped."""
    band = np.full((84, canvas.shape[1], 3), 255, dtype=np.uint8)
    title = "{} | {} -> {}".format(facade_id, year_prev, year_curr)
    subtitle = "{} | target: {}".format(semantic_label, REFERENCE_LABELS.get(reference, reference))
    put_fitted_text(band, title, (16, 33), canvas.shape[1] - 32, preferred=0.82, thickness=2, colour=(15, 15, 15))
    put_fitted_text(band, subtitle, (16, 67), canvas.shape[1] - 32, preferred=0.56, thickness=1, colour=(55, 55, 55))
    return np.concatenate([band, canvas], axis=0)


def semantic_legend(width: int, primary_method: Optional[str] = None,
                    primary_limits: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Create an embedded semantic-class legend; optionally add a heatmap colour bar."""
    columns = 3 if width < 1300 else 4
    cell_width = width // columns
    entries: List[Tuple[str, Tuple[int, int, int]]] = [
        ("{} {}".format(idx, name), tuple(int(value) for value in rgb[::-1]))
        for idx, (name, rgb) in enumerate(zip(DEFAULT_CLASS_NAMES, DEFAULT_PALETTE_RGB))
    ]
    entries.extend([
        ("semantic change", (0, 0, 238)),
        ("invalid / excluded", (55, 55, 55)),
    ])
    n_rows = int(np.ceil(len(entries) / columns))
    heat_height = 66 if primary_method is not None and primary_limits is not None else 0
    canvas = np.full((42 + n_rows * 33 + heat_height, width, 3), 255, dtype=np.uint8)
    put_fitted_text(canvas, "Semantic class legend", (14, 27), width - 28, preferred=0.60, thickness=1)
    for index, (name, bgr) in enumerate(entries):
        row, col = divmod(index, columns)
        x0 = col * cell_width + 14
        y0 = 42 + row * 33
        canvas[y0 + 5:y0 + 27, x0:x0 + 30] = bgr
        put_fitted_text(canvas, name, (x0 + 40, y0 + 22), cell_width - 54, preferred=0.42, thickness=1)
    if heat_height:
        lo, hi = primary_limits or (0.0, 1.0)
        y0 = 42 + n_rows * 33 + 10
        label = "{} overlay score (global P5-P95: {:.3f} -> {:.3f})".format(primary_method, lo, hi)
        put_fitted_text(canvas, label, (14, y0 + 18), width - 28, preferred=0.45, thickness=1)
        bar_width = min(430, width - 160)
        bar_x, bar_y = 14, y0 + 29
        gradient = np.tile(np.linspace(0, 255, bar_width, dtype=np.uint8), (18, 1))
        bar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        canvas[bar_y:bar_y + 18, bar_x:bar_x + bar_width] = bar
        cv2.putText(canvas, "low", (bar_x + bar_width + 12, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, "high", (bar_x + bar_width + 54, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def write_legend(out_path: Path, width: int, primary_method: Optional[str] = None,
                 primary_limits: Optional[Tuple[float, float]] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), semantic_legend(width, primary_method, primary_limits))


def main() -> None:
    args = parse_args()
    if args.low_percentile < 0 or args.high_percentile > 100 or args.low_percentile >= args.high_percentile:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay-alpha must be in [0, 1]")
    pairs = [row for row in read_csv(args.pairs_manifest) if row.get("split", "") == args.split]
    semantic_rows = {row["pair_id"]: row for row in read_csv(args.semantic_manifest) if row.get("split", "") == args.split}
    scores, score_paths = score_sources(args.score_source, args.split)
    if args.primary_method not in scores:
        raise KeyError("Primary method {} is not one of score sources: {}".format(args.primary_method, sorted(scores)))
    requested = set(args.pair_id or [])
    if requested:
        pairs = [row for row in pairs if row["pair_id"] in requested]
        missing_requested = requested - {row["pair_id"] for row in pairs}
        if missing_requested:
            raise KeyError("Requested pair_ids absent from split {}: {}".format(args.split, sorted(missing_requested)))
    for row in pairs:
        if row["pair_id"] not in semantic_rows:
            raise KeyError("Semantic manifest lacks selected pair {}".format(row["pair_id"]))
        missing_scores = [method for method, grouped in scores.items() if row["pair_id"] not in grouped]
        if missing_scores:
            raise KeyError("Score sources lack pair {} for methods {}".format(row["pair_id"], missing_scores))

    semantic_ratio_key = args.reference + "_ratio"
    if args.sort_by == "semantic_change_ratio":
        pairs.sort(key=lambda row: float(semantic_rows[row["pair_id"]].get(semantic_ratio_key, 0.0)), reverse=True)
    elif args.sort_by == "primary_score":
        pairs.sort(key=lambda row: float(np.mean([float(item["tile_score"]) for item in scores[args.primary_method][row["pair_id"]]])), reverse=True)
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    if not pairs:
        raise ValueError("No pairs selected for figure rendering")
    selected_ids = [row["pair_id"] for row in pairs]
    ranges = {
        method: method_range(grouped, selected_ids, args.low_percentile, args.high_percentile)
        for method, grouped in scores.items()
    }
    semantic_label = args.semantic_label or ("GT semantic masks" if args.semantic_source == "gt" else "Predicted semantic masks")
    paper_dir = args.out_dir / args.semantic_source / "paper_style"
    ablation_dir = args.out_dir / args.semantic_source / "method_ablation"
    paper_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    write_legend(args.out_dir / args.semantic_source / "semantic_legend.png", args.cell_width * 2,
                 args.primary_method, ranges[args.primary_method])
    output_rows: List[Dict[str, object]] = []

    for pair in tqdm(pairs, desc="Rendering temporal grids", unit="pair"):
        pair_id = pair["pair_id"]
        current = read_image(Path(pair["curr_image_path"]), cv2.IMREAD_COLOR)
        previous = read_image(Path(pair["prev_aligned_path"]), cv2.IMREAD_COLOR)
        valid = read_image(Path(pair["valid_mask_path"]), cv2.IMREAD_GRAYSCALE) > args.valid_threshold
        semantic = semantic_rows[pair_id]
        prev_mask_path, curr_mask_path, change_path = resolve_semantic_paths(semantic, args.semantic_source, args.reference)
        prev_mask = read_image(prev_mask_path, cv2.IMREAD_UNCHANGED)
        curr_mask = read_image(curr_mask_path, cv2.IMREAD_UNCHANGED)
        change = read_image(change_path, cv2.IMREAD_UNCHANGED)
        if prev_mask.ndim == 3:
            prev_mask = prev_mask[..., 0]
        if curr_mask.ndim == 3:
            curr_mask = curr_mask[..., 0]
        if change.ndim == 3:
            change = change[..., 0]
        prev_semantic = semantic_colour(prev_mask, args.invalid_label)
        curr_semantic = semantic_colour(curr_mask, args.invalid_label)
        semantic_change = change_colour(change, args.invalid_label)
        semantic_on_rgb = change_overlay(current, change, args.invalid_label, args.overlay_alpha)
        maps: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for method, grouped in scores.items():
            raw, covered = score_map(grouped[pair_id], current.shape[:2])
            limits = (ranges[method] if args.normalization == "global" else
                      method_range({pair_id: grouped[pair_id]}, [pair_id], args.low_percentile, args.high_percentile))
            heat = heatmap_visual(raw, covered, valid, limits)
            overlay = heatmap_overlay(current, heat, covered & valid, args.overlay_alpha)
            maps[method] = (raw, heat, overlay)

        if not args.no_paper_style:
            primary_panel = maps[args.primary_method][2] if args.paper_primary_panel == "overlay" else maps[args.primary_method][1]
            primary_title = "{} heatmap overlay".format(args.primary_method) if args.paper_primary_panel == "overlay" else "{} temporal heatmap".format(args.primary_method)
            cells = [
                [labelled_panel(previous, "Previous aligned RGB ({})".format(pair.get("year_prev", "")), args.cell_width, args.title_height),
                 labelled_panel(current, "Current RGB ({})".format(pair.get("year_curr", "")), args.cell_width, args.title_height)],
                [labelled_panel(prev_semantic, "Previous semantic mask", args.cell_width, args.title_height),
                 labelled_panel(curr_semantic, "Current semantic mask", args.cell_width, args.title_height)],
                [labelled_panel(semantic_change, "Semantic change: {}".format(REFERENCE_LABELS.get(args.reference, args.reference)), args.cell_width, args.title_height),
                 labelled_panel(primary_panel, primary_title, args.cell_width, args.title_height)],
            ]
            canvas = np.concatenate([row_canvas(row) for row in cells], axis=0)
            if not args.no_embedded_legend:
                canvas = np.concatenate([canvas, semantic_legend(canvas.shape[1], args.primary_method, ranges[args.primary_method])], axis=0)
            canvas = add_header(canvas, pair.get("facade_id", pair_id), pair.get("year_prev", ""),
                                pair.get("year_curr", ""), semantic_label, args.reference)
            out_path = paper_dir / (pair_id + "_paper_grid.png")
            cv2.imwrite(str(out_path), canvas)
        else:
            out_path = Path("")

        if not args.no_ablation_style:
            method_labels = list(scores.keys())
            top_cells = [
                labelled_panel(previous, "Previous aligned RGB", args.cell_width, args.title_height),
                labelled_panel(current, "Current RGB", args.cell_width, args.title_height),
                labelled_panel(semantic_change, "Semantic change", args.cell_width, args.title_height),
            ] + [labelled_panel(maps[method][1], method + " heatmap", args.cell_width, args.title_height) for method in method_labels]
            bottom_cells = [
                labelled_panel(prev_semantic, "Previous semantic mask", args.cell_width, args.title_height),
                labelled_panel(curr_semantic, "Current semantic mask", args.cell_width, args.title_height),
                labelled_panel(semantic_on_rgb, "Semantic change overlay", args.cell_width, args.title_height),
            ] + [labelled_panel(maps[method][2], method + " overlay", args.cell_width, args.title_height) for method in method_labels]
            comparison = np.concatenate([row_canvas(top_cells), row_canvas(bottom_cells)], axis=0)
            comparison = add_header(comparison, pair.get("facade_id", pair_id), pair.get("year_prev", ""),
                                    pair.get("year_curr", ""), semantic_label + " | visual method ablation", args.reference)
            comparison_path = ablation_dir / (pair_id + "_methods_grid.png")
            cv2.imwrite(str(comparison_path), comparison)
        else:
            comparison_path = Path("")
        output_rows.append({
            "pair_id": pair_id,
            "facade_id": pair.get("facade_id", ""),
            "year_prev": pair.get("year_prev", ""),
            "year_curr": pair.get("year_curr", ""),
            "split": args.split,
            "semantic_source": args.semantic_source,
            "reference": args.reference,
            "semantic_change_ratio": semantic.get(semantic_ratio_key, ""),
            "paper_grid_path": str(out_path),
            "method_ablation_grid_path": str(comparison_path),
        })

    manifest_path = args.out_dir / args.semantic_source / "rendered_figures_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "pairs_manifest": str(args.pairs_manifest),
        "semantic_manifest": str(args.semantic_manifest),
        "semantic_source": args.semantic_source,
        "semantic_label": semantic_label,
        "reference": args.reference,
        "split": args.split,
        "n_pairs": len(output_rows),
        "methods": list(scores.keys()),
        "score_sources": score_paths,
        "primary_method": args.primary_method,
        "paper_primary_panel": args.paper_primary_panel,
        "embedded_legend": not args.no_embedded_legend,
        "normalization": args.normalization,
        "normalization_percentiles": [args.low_percentile, args.high_percentile],
        "global_score_ranges": {method: [float(value) for value in limits] for method, limits in ranges.items()},
        "layout_note": "Paper-style grids contain RGB observations, semantic maps, semantic change, an RGB heatmap overlay, and an embedded semantic legend.",
        "semantic_branch_note": "In the RGB-residual experiment, semantic source changes contextual/change-map panels only; all temporal score heatmaps are image-derived and held fixed between GT and predicted semantic visualisations.",
        "quantitative_warning": "Predicted-semantic grids are qualitative whole-pipeline views and must not replace GT-based quantitative evaluation.",
        "output_manifest": str(manifest_path),
    }
    report_path = args.out_dir / args.semantic_source / "render_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Rendered {} {} figure sets for split {}".format(len(output_rows), args.semantic_source, args.split))
    print("Paper-style figures:", paper_dir)
    print("Method-ablation figures:", ablation_dir)
    print("Manifest:", manifest_path)
    print("Report:", report_path)
    print("NOTE: RGB-derived heatmaps are identical between GT and predicted semantic contexts; only semantic panels differ.")


if __name__ == "__main__":
    main()
