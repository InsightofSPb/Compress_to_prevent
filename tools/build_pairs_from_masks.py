import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ManifestRow:
    facade_id: str
    year: Optional[int]
    mask_path: str
    mask_name: str
    parse_rule: str
    width: Optional[int]
    height: Optional[int]
    file_size: Optional[int]

    @property
    def resolution(self) -> Optional[int]:
        if self.width is None or self.height is None:
            return None
        return self.width * self.height


@dataclass
class PairRow:
    pair_id: str
    facade_id: str
    year_a: int
    year_b: int
    mask_a: str
    mask_b: str
    delta_years: int


@dataclass
class DuplicateRow:
    facade_id: str
    year: int
    chosen_mask: str
    other_mask: str
    reason: str


TEMPORAL_FIELDS = ["facade_id", "year", "mask_path", "mask_name", "parse_rule"]
PAIR_FIELDS = ["pair_id", "facade_id", "year_a", "year_b", "mask_a", "mask_b", "delta_years"]
UNKNOWN_FIELDS = ["facade_id", "mask_path", "mask_name", "parse_rule"]
DUPLICATE_FIELDS = ["facade_id", "year", "chosen_mask", "other_mask", "reason"]


def read_summary(summary_path: Path, min_years: int) -> Sequence[str]:
    facade_ids: List[str] = []
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_multi_year = int(row.get("has_multi_year", "0") or 0)
            n_unique_years = int(row.get("n_unique_years", "0") or 0)
            if has_multi_year == 1 and n_unique_years >= min_years:
                facade_ids.append(row["facade_id"])
    return facade_ids


def _parse_optional_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def read_manifest(manifest_path: Path) -> Tuple[List[ManifestRow], List[Dict[str, str]]]:
    parsed_rows: List[ManifestRow] = []
    unknown_rows: List[Dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = _parse_optional_int(row.get("year", ""))
            width = _parse_optional_int(row.get("width", ""))
            height = _parse_optional_int(row.get("height", ""))
            file_size = _parse_optional_int(row.get("file_size", ""))
            manifest_row = ManifestRow(
                facade_id=row.get("facade_id", ""),
                year=year,
                mask_path=row.get("mask_path", ""),
                mask_name=row.get("mask_name", ""),
                parse_rule=row.get("parse_rule", row.get("rule", "")),
                width=width,
                height=height,
                file_size=file_size,
            )
            if manifest_row.year is None:
                unknown_rows.append(
                    {
                        "facade_id": manifest_row.facade_id,
                        "mask_path": manifest_row.mask_path,
                        "mask_name": manifest_row.mask_name,
                        "parse_rule": manifest_row.parse_rule,
                    }
                )
            parsed_rows.append(manifest_row)
    return parsed_rows, unknown_rows


def choose_main_mask(rows: Sequence[ManifestRow]) -> Tuple[ManifestRow, List[DuplicateRow]]:
    def score(row: ManifestRow) -> Tuple[int, int, str]:
        resolution = row.resolution if row.resolution is not None else -1
        file_size = row.file_size if row.file_size is not None else -1
        return (-resolution, -file_size, row.mask_name)

    sorted_rows = sorted(rows, key=score)
    chosen = sorted_rows[0]
    duplicates: List[DuplicateRow] = []
    for other in sorted_rows[1:]:
        reason = "lower_rank"
        duplicates.append(
            DuplicateRow(
                facade_id=other.facade_id,
                year=other.year or -1,
                chosen_mask=chosen.mask_name,
                other_mask=other.mask_name,
                reason=reason,
            )
        )
    return chosen, duplicates


def deduplicate_rows(rows: Sequence[ManifestRow]) -> Tuple[List[ManifestRow], List[DuplicateRow]]:
    grouped: Dict[Tuple[str, int], List[ManifestRow]] = defaultdict(list)
    for row in rows:
        if row.year is None:
            continue
        grouped[(row.facade_id, row.year)].append(row)

    deduped: List[ManifestRow] = []
    duplicates: List[DuplicateRow] = []
    for (facade_id, year), items in grouped.items():
        if len(items) == 1:
            deduped.append(items[0])
            continue
        chosen, dup_rows = choose_main_mask(items)
        deduped.append(chosen)
        duplicates.extend(dup_rows)
    deduped.sort(key=lambda r: (r.facade_id, r.year))
    return deduped, duplicates


def build_pairs(deduped: Sequence[ManifestRow]) -> Tuple[List[PairRow], List[PairRow]]:
    by_facade: Dict[str, Dict[int, ManifestRow]] = defaultdict(dict)
    for row in deduped:
        by_facade[row.facade_id][row.year or -1] = row

    pairs_consecutive: List[PairRow] = []
    pairs_to_ref: List[PairRow] = []
    for facade_id, years_dict in by_facade.items():
        years_sorted = sorted(years_dict.keys())
        if len(years_sorted) < 2:
            continue
        ref_year = years_sorted[0]
        ref_mask = years_dict[ref_year].mask_path
        for idx in range(len(years_sorted) - 1):
            year_a = years_sorted[idx]
            year_b = years_sorted[idx + 1]
            row_a = years_dict[year_a]
            row_b = years_dict[year_b]
            pair_id = f"{facade_id}_{year_a}_{year_b}"
            pairs_consecutive.append(
                PairRow(
                    pair_id=pair_id,
                    facade_id=facade_id,
                    year_a=year_a,
                    year_b=year_b,
                    mask_a=row_a.mask_path,
                    mask_b=row_b.mask_path,
                    delta_years=year_b - year_a,
                )
            )
        for year in years_sorted:
            if year == ref_year:
                continue
            row_y = years_dict[year]
            pair_id = f"{facade_id}_{year}_{ref_year}"
            pairs_to_ref.append(
                PairRow(
                    pair_id=pair_id,
                    facade_id=facade_id,
                    year_a=year,
                    year_b=ref_year,
                    mask_a=row_y.mask_path,
                    mask_b=ref_mask,
                    delta_years=year - ref_year,
                )
            )
    return pairs_consecutive, pairs_to_ref


def write_csv(rows: Sequence[Dict[str, object]], fieldnames: Sequence[str], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal mask pairs")
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary_masks.csv")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest_masks.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--min-years", type=int, default=2, help="Minimum unique years to keep facade")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    facades = set(read_summary(args.summary, args.min_years))
    print(f"Multi-year facades passing filter: {len(facades)}")

    manifest_rows, unknown_rows = read_manifest(args.manifest)
    unknown_path = out_dir / "unknown_year.csv"
    write_csv(unknown_rows, UNKNOWN_FIELDS, unknown_path)
    print(f"Unknown year entries written: {len(unknown_rows)}")

    filtered_rows = [row for row in manifest_rows if row.facade_id in facades and row.year is not None]
    deduped_rows, duplicates = deduplicate_rows(filtered_rows)

    temporal_rows = [
        {
            "facade_id": row.facade_id,
            "year": row.year,
            "mask_path": row.mask_path,
            "mask_name": row.mask_name,
            "parse_rule": row.parse_rule,
        }
        for row in deduped_rows
    ]
    temporal_path = out_dir / "temporal_manifest.csv"
    write_csv(temporal_rows, TEMPORAL_FIELDS, temporal_path)
    print(f"Unique (facade_id, year) rows: {len(temporal_rows)}")

    pairs_consecutive, pairs_to_ref = build_pairs(deduped_rows)
    pairs_consecutive_path = out_dir / "pairs_consecutive.csv"
    pairs_to_ref_path = out_dir / "pairs_to_ref.csv"
    write_csv([pair.__dict__ for pair in pairs_consecutive], PAIR_FIELDS, pairs_consecutive_path)
    write_csv([pair.__dict__ for pair in pairs_to_ref], PAIR_FIELDS, pairs_to_ref_path)

    print(f"Consecutive pairs: {len(pairs_consecutive)}")
    print(f"Pairs to ref: {len(pairs_to_ref)}")

    duplicates_path = out_dir / "duplicates.csv"
    if duplicates:
        write_csv([dup.__dict__ for dup in duplicates], DUPLICATE_FIELDS, duplicates_path)
        print(f"Duplicates recorded: {len(duplicates)}")
    else:
        duplicates_path.write_text("")
        print("Duplicates recorded: 0")

    years_by_facade: Dict[str, int] = defaultdict(int)
    for row in deduped_rows:
        years_by_facade[row.facade_id] += 1
    top_facades = sorted(years_by_facade.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("Top facades by number of years (after dedup):")
    for facade_id, count in top_facades:
        print(f"  {facade_id}: {count}")


if __name__ == "__main__":
    main()
