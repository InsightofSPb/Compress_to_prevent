from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from compression.io import read_csv_rows, write_csv_rows


def evaluate_temporal_semantic_features(
    features_csv: Path,
    out_summary_csv: Path,
    topk_csv: Path,
    labels_csv: Path | None = None,
    top_k: int = 5,
) -> Dict[str, object]:
    rows = read_csv_rows(features_csv)
    by_backend: Dict[str, List[float]] = defaultdict(list)
    by_pair: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(float(row["semantic_score_backend"]))
        by_pair[row["pair_id"]].append(row)

    summary_rows = []
    for backend, values in sorted(by_backend.items()):
        summary_rows.append(
            {
                "backend": backend,
                "n_tiles": len(values),
                "mean_semantic_score_backend": sum(values) / max(len(values), 1),
                "max_semantic_score_backend": max(values) if values else 0.0,
            }
        )
    write_csv_rows(
        out_summary_csv,
        ["backend", "n_tiles", "mean_semantic_score_backend", "max_semantic_score_backend"],
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

    label_rows = read_csv_rows(labels_csv) if labels_csv is not None and labels_csv.exists() else []
    return {
        "n_rows": len(rows),
        "n_backends": len(by_backend),
        "labels_used": len(label_rows),
    }
