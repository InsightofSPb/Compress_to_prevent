from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

from compression.io import read_csv_rows, write_csv_rows


def _topk_tile_ids(rows: List[Dict[str, str]], k: int) -> List[str]:
    return [row["tile_id"] for row in sorted(rows, key=lambda r: float(r["semantic_score_backend"]), reverse=True)[:k]]


def evaluate_temporal_semantic_features(
    features_csv: Path,
    out_summary_csv: Path,
    topk_csv: Path,
    labels_csv: Path | None = None,
    top_k: int = 5,
) -> Dict[str, object]:
    rows = read_csv_rows(features_csv)
    by_backend: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_pair_backend: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    by_pair: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        by_backend[row["backend"]].append(row)
        by_pair_backend[(row["pair_id"], row["backend"])].append(row)
        by_pair[row["pair_id"]].append(row)

    summary_rows = []
    for backend, backend_rows in sorted(by_backend.items()):
        values = [float(r["semantic_score_backend"]) for r in backend_rows]
        agreement_vals = [float(r["backend_agreement_score"]) for r in backend_rows]
        disagreement_vals = [float(r["backend_disagreement_score"]) for r in backend_rows]
        summary_rows.append(
            {
                "backend": backend,
                "n_tiles": len(values),
                "mean_semantic_score_backend": sum(values) / max(len(values), 1),
                "max_semantic_score_backend": max(values) if values else 0.0,
                "mean_backend_agreement_score": sum(agreement_vals) / max(len(agreement_vals), 1),
                "mean_backend_disagreement_score": sum(disagreement_vals) / max(len(disagreement_vals), 1),
            }
        )
    write_csv_rows(
        out_summary_csv,
        [
            "backend",
            "n_tiles",
            "mean_semantic_score_backend",
            "max_semantic_score_backend",
            "mean_backend_agreement_score",
            "mean_backend_disagreement_score",
        ],
        summary_rows,
    )

    top_rows = []
    for pair_id, pair_rows in by_pair.items():
        ordered = sorted(pair_rows, key=lambda r: float(r["semantic_score_fused"]), reverse=True)[:top_k]
        for rank, row in enumerate(ordered, start=1):
            top_rows.append(
                {
                    "pair_id": pair_id,
                    "rank": rank,
                    "tile_id": row["tile_id"],
                    "backend": row["backend"],
                    "semantic_score_fused": row["semantic_score_fused"],
                }
            )
    write_csv_rows(topk_csv, ["pair_id", "rank", "tile_id", "backend", "semantic_score_fused"], top_rows)

    overlap_rows = []
    for pair_id in sorted(by_pair.keys()):
        backend_names = sorted({backend for (pid, backend) in by_pair_backend.keys() if pid == pair_id})
        for b1, b2 in combinations(backend_names, 2):
            t1 = set(_topk_tile_ids(by_pair_backend[(pair_id, b1)], top_k))
            t2 = set(_topk_tile_ids(by_pair_backend[(pair_id, b2)], top_k))
            denom = max(len(t1 | t2), 1)
            jaccard = len(t1 & t2) / denom
            overlap_rows.append({"pair_id": pair_id, "backend_a": b1, "backend_b": b2, "topk_jaccard": jaccard})

    overlap_csv = out_summary_csv.with_name(out_summary_csv.stem + "_overlap.csv")
    write_csv_rows(overlap_csv, ["pair_id", "backend_a", "backend_b", "topk_jaccard"], overlap_rows)

    label_rows = read_csv_rows(labels_csv) if labels_csv is not None and labels_csv.exists() else []
    return {
        "n_rows": len(rows),
        "n_backends": len(by_backend),
        "labels_used": len(label_rows),
        "overlap_csv": str(overlap_csv),
    }
