import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

RADIUS_FACTORS: Dict[str, float] = {
    "strong": 0.06,
    "weak": 0.10,
    "none": 0.15,
}
W_POS = 1.0
W_INT = 0.5
W_STD = 0.2
W_AREA = 0.1
SCORE_MIN = -1.0
MAX_LINES = 200
EPS = 1e-6


@dataclass
class MatchResult:
    a: int
    b: int
    d_pos: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match superpixel objects between years.")
    parser.add_argument("--pairs", required=True, type=Path, help="Path to pairs_consecutive.csv")
    parser.add_argument("--spx-cache", required=True, type=Path, help="Base directory with spx/objs/viz outputs")
    parser.add_argument("--geom-dir", required=True, type=Path, help="Directory with rectification geometry JSON files")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for matches and visualizations")
    parser.add_argument("--facade-id", type=str, default=None, help="Optional facade_id filter")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of pairs to process")
    return parser.parse_args()


def load_pairs(pairs_path: Path, facade_id: Optional[str], limit: Optional[int]) -> pd.DataFrame:
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    df = pd.read_csv(pairs_path)
    required_cols = {"facade_id", "year_a", "year_b"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        raise ValueError(f"Pairs CSV missing required columns: {missing}")
    if facade_id is not None:
        df = df[df["facade_id"] == facade_id]
    if limit is not None:
        df = df.head(limit)
    return df


def load_objects(spx_cache: Path, facade_id: str, year: int) -> pd.DataFrame:
    obj_path = spx_cache / "facades" / str(facade_id) / "spx" / "objs" / f"{year}_spx.parquet"
    if not obj_path.exists():
        raise FileNotFoundError(f"Objects parquet not found: {obj_path}")
    return pd.read_parquet(obj_path)


def load_overlay_image(spx_cache: Path, facade_id: str, year: int) -> Optional[np.ndarray]:
    img_path = spx_cache / "facades" / str(facade_id) / "spx" / "viz" / f"{year}_spx_overlay.png"
    if not img_path.exists():
        LOGGER.warning("Overlay image not found for %s %s: %s", facade_id, year, img_path)
        return None
    return plt.imread(img_path)


def load_geom(geom_dir: Path, facade_id: str, year_a: int, year_b: int) -> Tuple[str, Optional[np.ndarray]]:
    geom_candidates = [
        geom_dir / str(facade_id) / f"{year_a}_{year_b}.json",
        geom_dir / f"{facade_id}_{year_a}_{year_b}.json",
    ]

    geom_path = next((p for p in geom_candidates if p.exists()), None)
    if geom_path is None:
        LOGGER.warning(
            "Geometry file not found for %s. Tried: %s",
            facade_id,
            ", ".join(str(p) for p in geom_candidates),
        )
        return "none", None

    with geom_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    status_quality = data.get("status_quality", "none") or "none"
    H = data.get("H")
    if H is not None:
        H = np.array(H, dtype=float)
        if H.shape != (3, 3):
            LOGGER.warning("Invalid H shape in %s: %s", geom_path, H.shape)
            H = None
    else:
        H = None
    return status_quality, H


def compute_diag(df: pd.DataFrame) -> float:
    width = float(df["bbox_x2"].max())
    height = float(df["bbox_y2"].max())
    return float(np.hypot(width, height))


def warp_points(H: Optional[np.ndarray], coords: np.ndarray) -> np.ndarray:
    if H is None:
        return coords
    pts_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
    warped = (H @ pts_h.T).T
    warped_xy = warped[:, :2] / (warped[:, 2:3] + EPS)
    return warped_xy


def score_pair(row_a: pd.Series, row_b: pd.Series, d_pos: float) -> float:
    mean_diff = abs(float(row_a["mean_intensity"]) - float(row_b["mean_intensity"]))
    std_diff = abs(float(row_a["std_intensity"]) - float(row_b["std_intensity"]))
    area_a = max(float(row_a["area_px"]), EPS)
    area_b = max(float(row_b["area_px"]), EPS)
    area_diff = abs(np.log(area_b / area_a))
    return -W_POS * d_pos - W_INT * mean_diff - W_STD * std_diff - W_AREA * area_diff


def find_best_candidates(
    df_a: pd.DataFrame, df_b: pd.DataFrame, coords_a: np.ndarray, coords_b: np.ndarray, radius: float
) -> Dict[int, MatchResult]:
    tree_b = cKDTree(coords_b)
    best: Dict[int, MatchResult] = {}
    for idx_a, point in enumerate(coords_a):
        candidates = tree_b.query_ball_point(point, radius)
        if not candidates:
            continue
        row_a = df_a.iloc[idx_a]
        best_score = None
        best_res: Optional[MatchResult] = None
        for idx_b in candidates:
            row_b = df_b.iloc[idx_b]
            d_pos = float(np.linalg.norm(point - coords_b[idx_b]))
            score = score_pair(row_a, row_b, d_pos)
            if best_score is None or score > best_score:
                best_score = score
                best_res = MatchResult(a=int(row_a["obj_id"]), b=int(row_b["obj_id"]), d_pos=d_pos, score=score)
        if best_res is not None:
            best[idx_a] = best_res
    return best


def choose_mutual_matches(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    radius: float,
) -> List[MatchResult]:
    best_a_to_b = find_best_candidates(df_a, df_b, coords_a, coords_b, radius)
    best_b_to_a = find_best_candidates(df_b, df_a, coords_b, coords_a, radius)

    obj_id_to_idx_a = {int(row["obj_id"]): idx for idx, row in df_a.iterrows()}
    obj_id_to_idx_b = {int(row["obj_id"]): idx for idx, row in df_b.iterrows()}

    matches: List[MatchResult] = []
    for idx_a, res_ab in best_a_to_b.items():
        idx_b = obj_id_to_idx_b.get(res_ab.b)
        if idx_b is None:
            continue
        res_ba = best_b_to_a.get(idx_b)
        if res_ba is None:
            continue
        if res_ba.b != res_ab.a:
            continue
        if res_ab.d_pos >= radius:
            continue
        if res_ab.score <= SCORE_MIN:
            continue
        matches.append(res_ab)
    return matches


def visualize_matches(
    image_b: np.ndarray,
    coords_a_warp: np.ndarray,
    coords_b: np.ndarray,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    matches: List[MatchResult],
    out_path: Path,
):
    if image_b is None:
        return
    scores = np.array([m.score for m in matches]) if matches else np.array([])
    top_indices = np.argsort(-scores)[:MAX_LINES] if len(scores) else []
    lines = []
    colors = []
    for idx in top_indices:
        match = matches[int(idx)]
        idx_a = df_a.index[df_a["obj_id"] == match.a][0]
        idx_b = df_b.index[df_b["obj_id"] == match.b][0]
        start = coords_a_warp[idx_a]
        end = coords_b[idx_b]
        lines.append([start, end])
        colors.append("yellow")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_b)
    if lines:
        lc = mc.LineCollection(lines, colors=colors, linewidths=1, alpha=0.8)
        ax.add_collection(lc)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def process_pair(
    row: pd.Series,
    spx_cache: Path,
    geom_dir: Path,
    out_dir: Path,
) -> Optional[Dict[str, object]]:
    facade_id = str(row["facade_id"])
    year_a = int(row["year_a"])
    year_b = int(row["year_b"])

    df_a = load_objects(spx_cache, facade_id, year_a)
    df_b = load_objects(spx_cache, facade_id, year_b)

    status_quality, H = load_geom(geom_dir, facade_id, year_a, year_b)
    radius_factor = RADIUS_FACTORS.get(status_quality, RADIUS_FACTORS["none"])
    coords_a = df_a[["cx", "cy"]].to_numpy(dtype=float)
    coords_b = df_b[["cx", "cy"]].to_numpy(dtype=float)
    coords_a_warp = warp_points(H, coords_a)

    diag = compute_diag(df_b)
    radius = radius_factor * diag

    matches = choose_mutual_matches(df_a, df_b, coords_a_warp, coords_b, radius)

    image_b = load_overlay_image(spx_cache, facade_id, year_b)
    viz_path = out_dir / str(facade_id) / f"{year_a}_{year_b}_lines.png"
    visualize_matches(image_b, coords_a_warp, coords_b, df_a, df_b, matches, viz_path)

    match_dicts = [asdict(m) for m in matches]
    summary = {
        "facade_id": facade_id,
        "year_a": year_a,
        "year_b": year_b,
        "status_quality": status_quality,
        "num_A": len(df_a),
        "num_B": len(df_b),
        "num_matches": len(matches),
        "matches": match_dicts,
    }

    out_path = out_dir / str(facade_id) / f"{year_a}_{year_b}_match.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def write_report(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    enriched = []
    for row in rows:
        matches = row.get("matches", [])
        d_positions = [m["d_pos"] for m in matches]
        scores = [m["score"] for m in matches]
        match_ratio = row["num_matches"] / max(min(row["num_A"], row["num_B"]), 1)
        enriched.append(
            {
                "facade_id": row["facade_id"],
                "year_a": row["year_a"],
                "year_b": row["year_b"],
                "status_quality": row["status_quality"],
                "num_A": row["num_A"],
                "num_B": row["num_B"],
                "num_matches": row["num_matches"],
                "match_ratio": match_ratio,
                "mean_d_pos": float(np.mean(d_positions)) if d_positions else None,
                "median_d_pos": float(np.median(d_positions)) if d_positions else None,
                "mean_score": float(np.mean(scores)) if scores else None,
            }
        )
    df = pd.DataFrame(enriched)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    args = parse_args()
    pairs = load_pairs(args.pairs, args.facade_id, args.limit)
    if pairs.empty:
        LOGGER.warning("No pairs to process after filtering.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []

    for idx, row in pairs.iterrows():
        LOGGER.info(
            "Processing pair %s/%s (%d of %d)", row["year_a"], row["year_b"], idx + 1, len(pairs)
        )
        try:
            summary = process_pair(row, args.spx_cache, args.geom_dir, args.out_dir)
            if summary:
                summaries.append(summary)
        except FileNotFoundError as e:
            LOGGER.error("Skipping pair due to missing file: %s", e)
        except Exception:
            LOGGER.exception("Error processing pair %s/%s", row["year_a"], row["year_b"])

    report_path = args.out_dir / "match_report.csv"
    write_report(summaries, report_path)


if __name__ == "__main__":
    main()