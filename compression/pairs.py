from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

from .io import read_csv_rows, write_csv_rows

YEAR_SUFFIX_RE = re.compile(r"^(?P<facade>.+)_(?P<year>\d{4})$")


@dataclass
class Observation:
    facade_id: str
    year: int
    image_path: str
    aligned_path: str
    split: str


def _parse_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _infer_facade_year(path_like: str) -> tuple[str, Optional[int]]:
    stem = Path(path_like).stem
    match = YEAR_SUFFIX_RE.match(stem)
    if not match:
        return stem, None
    return match.group("facade"), int(match.group("year"))


def read_observations(manifest_csv: Path) -> List[Observation]:
    rows = read_csv_rows(manifest_csv)
    observations: List[Observation] = []
    for row in rows:
        image_path = row.get("image_path") or row.get("curr_image_path") or row.get("mask_path")
        if not image_path:
            continue
        facade_id = row.get("facade_id", "").strip()
        year = _parse_int(row.get("year", ""))
        infer_facade, infer_year = _infer_facade_year(image_path)
        if not facade_id:
            facade_id = infer_facade
        if year is None:
            year = infer_year
        if year is None:
            continue
        observations.append(
            Observation(
                facade_id=facade_id,
                year=year,
                image_path=image_path,
                aligned_path=row.get("aligned_image_path", row.get("aligned_path", "")),
                split=row.get("split", "train") or "train",
            )
        )
    return observations


def build_facade_pairs(observations: Iterable[Observation], pair_mode: str = "consecutive") -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Observation]] = {}
    for obs in observations:
        grouped.setdefault((obs.facade_id, obs.split), []).append(obs)

    pair_rows: List[Dict[str, object]] = []
    for (facade_id, split), items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x.year)
        if len(items_sorted) < 2:
            continue
        if pair_mode == "consecutive":
            iterator = zip(items_sorted[:-1], items_sorted[1:])
        elif pair_mode == "all_to_latest":
            latest = items_sorted[-1]
            iterator = ((prev, latest) for prev in items_sorted[:-1])
        else:
            raise ValueError(f"Unsupported pair_mode: {pair_mode}")

        for prev, curr in iterator:
            pair_id = f"{facade_id}_{prev.year}_{curr.year}"
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "facade_id": facade_id,
                    "year_prev": prev.year,
                    "year_curr": curr.year,
                    "prev_image_path": prev.image_path,
                    "curr_image_path": curr.image_path,
                    "prev_aligned_path": prev.aligned_path,
                    "split": split,
                }
            )

    return sorted(pair_rows, key=lambda row: (str(row["split"]), str(row["facade_id"]), int(row["year_prev"])))


def write_split_pair_csvs(pair_rows: List[Dict[str, object]], out_dir: Path) -> None:
    fieldnames = [
        "pair_id",
        "facade_id",
        "year_prev",
        "year_curr",
        "prev_image_path",
        "curr_image_path",
        "prev_aligned_path",
        "split",
    ]
    write_csv_rows(out_dir / "pairs_all.csv", fieldnames, pair_rows)

    split_groups: Dict[str, List[Dict[str, object]]] = {}
    for row in pair_rows:
        split_groups.setdefault(str(row["split"]), []).append(row)
    for split, rows in split_groups.items():
        write_csv_rows(out_dir / f"pairs_{split}.csv", fieldnames, rows)
