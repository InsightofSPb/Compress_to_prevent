import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

YEAR_MIN = 1900
YEAR_MAX = 2100


@dataclass
class ParsedMask:
    mask_path: Path
    mask_name: str
    stem: str
    facade_id: str
    year: Optional[int]
    date: str
    time: str
    parse_rule: str

    @property
    def has_year(self) -> int:
        return int(self.year is not None)

    @property
    def has_date(self) -> int:
        return int(bool(self.date))


SUFFIX_YEAR_RE = re.compile(r"^(.*)_(\d{4})$")
PXL_DATETIME_RE = re.compile(r"^PXL_(\d{8})_(\d{6})")
PHOTO_DATETIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")


def _valid_year(year: int) -> bool:
    return YEAR_MIN <= year <= YEAR_MAX


def _format_time(time_digits: str) -> Optional[str]:
    if len(time_digits) != 6:
        return None
    hour, minute, second = time_digits[:2], time_digits[2:4], time_digits[4:]
    return f"{hour}:{minute}:{second}"


def parse_mask(stem: str) -> Tuple[str, Optional[int], str, str, str]:
    """Parse mask stem and return parsing results.

    Returns
    -------
    facade_id, year, date, time, parse_rule
    """
    # Rule A: suffix_year
    match = SUFFIX_YEAR_RE.match(stem)
    if match:
        facade_id, year_str = match.groups()
        year = int(year_str)
        if _valid_year(year):
            return facade_id, year, "", "", "suffix_year"

    # Rule B: pxl_datetime
    match = PXL_DATETIME_RE.match(stem)
    if match:
        date_digits, time_digits = match.groups()
        year = int(date_digits[:4])
        if _valid_year(year):
            date = f"{date_digits[:4]}-{date_digits[4:6]}-{date_digits[6:]}"
            formatted_time = _format_time(time_digits) or ""
            return stem, year, date, formatted_time, "pxl_datetime"

    # Rule C: photo_datetime
    match = PHOTO_DATETIME_RE.search(stem)
    if match:
        date_str, time_str = match.groups()
        year = int(date_str[:4])
        if _valid_year(year):
            time_formatted = time_str.replace("-", ":")
            return stem, year, date_str, time_formatted, "photo_datetime"

    # Rule D: none
    return stem, None, "", "", "none"


def collect_files(masks_dir: Path, extensions: Sequence[str]) -> List[Path]:
    allowed_exts = {ext.lower() for ext in extensions}
    files: List[Path] = []
    for path in masks_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_exts:
            files.append(path)
    return sorted(files)


def build_manifest(masks_dir: Path, extensions: Sequence[str]) -> List[ParsedMask]:
    rows: List[ParsedMask] = []
    for file_path in collect_files(masks_dir, extensions):
        stem = file_path.stem
        facade_id, year, date, time, parse_rule = parse_mask(stem)
        rows.append(
            ParsedMask(
                mask_path=file_path,
                mask_name=file_path.name,
                stem=stem,
                facade_id=facade_id,
                year=year,
                date=date,
                time=time,
                parse_rule=parse_rule,
            )
        )
    return rows


def write_manifest_csv(rows: Sequence[ParsedMask], out_path: Path) -> None:
    fieldnames = [
        "mask_path",
        "mask_name",
        "stem",
        "facade_id",
        "year",
        "date",
        "time",
        "parse_rule",
        "has_year",
        "has_date",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mask_path": str(row.mask_path),
                    "mask_name": row.mask_name,
                    "stem": row.stem,
                    "facade_id": row.facade_id,
                    "year": row.year if row.year is not None else "",
                    "date": row.date,
                    "time": row.time,
                    "parse_rule": row.parse_rule,
                    "has_year": row.has_year,
                    "has_date": row.has_date,
                }
            )


def summarize_facades(rows: Sequence[ParsedMask]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[ParsedMask]] = defaultdict(list)
    for row in rows:
        grouped[row.facade_id].append(row)

    summary_rows: List[Dict[str, object]] = []
    for facade_id, items in grouped.items():
        years = sorted({row.year for row in items if row.year is not None})
        years_str = ";".join(str(y) for y in years) if years else ""
        n_items = len(items)
        n_unique_years = len(years)
        has_multi_year = int(n_unique_years >= 2)
        has_any_year = int(n_unique_years > 0)
        n_unknown_year = sum(1 for row in items if row.year is None)
        summary_rows.append(
            {
                "facade_id": facade_id,
                "n_items": n_items,
                "years": years_str,
                "n_unique_years": n_unique_years,
                "has_multi_year": has_multi_year,
                "has_any_year": has_any_year,
                "n_unknown_year": n_unknown_year,
            }
        )
    summary_rows.sort(key=lambda r: r["facade_id"])
    return summary_rows


def write_summary_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "facade_id",
        "n_items",
        "years",
        "n_unique_years",
        "has_multi_year",
        "has_any_year",
        "n_unknown_year",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_stats(rows: Sequence[ParsedMask], summary_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total_files = len(rows)
    parsed_year_count = sum(row.has_year for row in rows)
    unknown_year_count = total_files - parsed_year_count
    multi_year_facades = sum(row["has_multi_year"] for row in summary_rows)

    rule_counter = Counter(row.parse_rule for row in rows)

    return {
        "total_files": total_files,
        "parsed_year_count": parsed_year_count,
        "unknown_year_count": unknown_year_count,
        "multi_year_facades": multi_year_facades,
        "parse_rule_distribution": dict(rule_counter),
    }


def write_stats_json(stats: Dict[str, object], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def log_report(rows: Sequence[ParsedMask], summary_rows: Sequence[Dict[str, object]]) -> None:
    print(f"N files found: {len(rows)}")
    parsed_year_count = sum(row.has_year for row in rows)
    print(f"parsed year: {parsed_year_count}")
    multi_year_facades = sum(row["has_multi_year"] for row in summary_rows)
    print(f"multi-year facades: {multi_year_facades}")

    top_facades = sorted(summary_rows, key=lambda r: r["n_items"], reverse=True)[:10]
    print("top10 facade_id by n_items:")
    for row in top_facades:
        print(f"  {row['facade_id']}: {row['n_items']}")

    unknown_year_counter = Counter(row.stem for row in rows if row.year is None)
    print("top10 unknown-year stems:")
    for stem, count in unknown_year_counter.most_common(10):
        print(f"  {stem}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest from mask files")
    parser.add_argument("--masks-dir", type=Path, required=True, dest="masks_dir")
    parser.add_argument("--out-dir", type=Path, required=True, dest="out_dir")
    parser.add_argument(
        "--ext",
        type=str,
        default=".png",
        help="Comma-separated list of extensions (e.g., .png,.jpg)",
    )
    return parser.parse_args()


def normalize_extensions(ext_arg: str) -> List[str]:
    extensions = [ext.strip().lower() for ext in ext_arg.split(",") if ext.strip()]
    return extensions if extensions else [".png"]


def main() -> None:
    args = parse_args()
    extensions = normalize_extensions(args.ext)
    masks_dir: Path = args.masks_dir
    out_dir: Path = args.out_dir

    rows = build_manifest(masks_dir, extensions)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_masks.csv"
    summary_path = out_dir / "summary_masks.csv"
    stats_path = out_dir / "stats.json"

    write_manifest_csv(rows, manifest_path)
    summary_rows = summarize_facades(rows)
    write_summary_csv(summary_rows, summary_path)
    stats = build_stats(rows, summary_rows)
    write_stats_json(stats, stats_path)
    log_report(rows, summary_rows)


if __name__ == "__main__":
    main()
