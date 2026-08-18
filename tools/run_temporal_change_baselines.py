#!/usr/bin/env python3
"""Compute tile-level temporal change baseline scores on aligned RGB pairs."""

from __future__ import annotations

import argparse
import csv
from importlib import metadata as package_metadata
import json
import platform
from pathlib import Path
import sys
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression.baselines import compute_baseline_tile_scores  # noqa: E402
from research_ledger import (  # noqa: E402
    ArtifactDescriptor,
    Ledger,
    NewEvent,
    canonical_hash,
    file_descriptor,
    repository_snapshot,
    sanitize_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute valid-region tile-level temporal change baselines."
    )
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument(
        "--methods",
        type=str,
        default="absdiff_l1,ssim_change",
        help="Comma-separated: absdiff_l1,absdiff_l2,grayscale_absdiff,ssim_change,lpips_change,dinov2_patch_cosine",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="",
        help="Optional comma-separated split restriction, e.g. val,test. Empty evaluates all manifest rows.",
    )
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--min-valid-ratio", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--dinov2-model-name", default="dinov2_vitb14")
    parser.add_argument("--dinov2-cache-dir", type=Path, default=None)
    parser.add_argument("--dinov2-weights-path", type=Path, default=None)
    parser.add_argument("--dinov2-repo-dir", type=Path, default=None)
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument(
        "--deep-batch-size",
        type=int,
        default=128,
        help="Batch size for tile-level deep baselines such as LPIPS.",
    )
    parser.add_argument("--skip-deep-baselines", action="store_true")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm progress bars."
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=None,
        help="Optional root for an append-only experiment ledger.",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _validate_and_inventory(
    manifest: Path, requested_splits: Optional[Sequence[str]]
) -> Tuple[List[Mapping[str, Any]], Mapping[str, List[str]], List[str], List[str]]:
    rows = _read_manifest(manifest)
    full_splits: Dict[str, List[str]] = {}
    for index, row in enumerate(rows, 2):
        for field in ("pair_id", "facade_id", "split"):
            if not row.get(field, "").strip():
                raise ValueError("manifest row {} has missing {}".format(index, field))
        full_splits.setdefault(row["split"], []).append(row["facade_id"])
    owners: Dict[str, str] = {}
    for split, facades in full_splits.items():
        for facade in facades:
            previous = owners.setdefault(facade, split)
            if previous != split:
                raise ValueError(
                    "facade {!r} overlaps {!r} and {!r}".format(facade, previous, split)
                )
    full_splits = {
        key: sorted(set(value)) for key, value in sorted(full_splits.items())
    }
    permitted = set(requested_splits or [])
    selected = [row for row in rows if not permitted or row["split"] in permitted]
    seen = set()
    inventory = []
    chronology_fields = (
        "prev_year",
        "curr_year",
        "year_prev",
        "year_curr",
        "previous_year",
        "current_year",
    )
    for row in selected:
        pair_id = row["pair_id"]
        if pair_id in seen:
            raise ValueError("duplicate selected pair_id: {}".format(pair_id))
        seen.add(pair_id)
        item: Dict[str, Any] = {
            "pair_id": pair_id,
            "facade_id": row["facade_id"],
            "split": row["split"],
        }
        for field in ("prev_aligned_path", "curr_image_path", "valid_mask_path"):
            value = row.get(field, "").strip()
            if field != "valid_mask_path" and not value:
                raise ValueError("pair {} has missing {}".format(pair_id, field))
            if value:
                try:
                    item[field] = file_descriptor(value)
                except OSError as exc:
                    raise FileNotFoundError(
                        "missing input for pair {}: {}".format(pair_id, value)
                    ) from exc
        for field in chronology_fields:
            if row.get(field, "").strip():
                item[field] = row[field]
        inventory.append(item)
    inventory.sort(key=lambda item: str(item["pair_id"]))
    return (
        inventory,
        full_splits,
        sorted(seen),
        sorted({str(item["facade_id"]) for item in inventory}),
    )


def _environment_snapshot(
    args: argparse.Namespace, methods: Sequence[str]
) -> Mapping[str, Any]:
    def version(distribution: str) -> Optional[str]:
        try:
            return package_metadata.version(distribution)
        except package_metadata.PackageNotFoundError:
            return None

    dependencies = {"numpy": version("numpy"), "Pillow": version("Pillow")}
    if "ssim_change" in methods:
        dependencies["scikit-image"] = version("scikit-image")
    if not args.skip_deep_baselines and "lpips_change" in methods:
        dependencies.update({"lpips": version("lpips"), "torch": version("torch")})
    if not args.skip_deep_baselines and "dinov2_patch_cosine" in methods:
        dependencies.update(
            {"torch": version("torch"), "torchvision": version("torchvision")}
        )
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": args.device,
        "dependencies": dependencies,
    }


def _append_failure(ledger: Ledger, exc: BaseException) -> None:
    try:
        ledger.append(NewEvent("run.failed", sanitize_error(exc)))
    except Exception as recording_error:
        note = "ledger could not record run failure: {}".format(recording_error)
        if hasattr(exc, "add_note"):
            exc.add_note(note)
        warnings.warn(note, RuntimeWarning)


def _record_snapshots(
    ledger: Ledger,
    args: argparse.Namespace,
    methods: Sequence[str],
    splits: Optional[Sequence[str]],
) -> Tuple[str, List[str]]:
    repo = repository_snapshot(PROJECT_ROOT, exclude_paths=[ledger.run_dir])
    source_ids = []
    source_ids.append(
        ledger.append(
            NewEvent(
                "source.snapshot",
                {
                    "repository": {
                        "git_commit": repo.git_commit,
                        "dirty_tree_fingerprint": repo.dirty_tree_fingerprint,
                    }
                },
            )
        ).event_id
    )
    inventory, definitions, pair_ids, facade_ids = _validate_and_inventory(
        args.residual_manifest, splits
    )
    manifest_info = file_descriptor(args.residual_manifest)
    model_inputs: Dict[str, Any] = {}
    if (
        "dinov2_patch_cosine" in methods
        and not args.skip_deep_baselines
        and args.dinov2_weights_path is not None
    ):
        model_inputs["dinov2_weights"] = file_descriptor(args.dinov2_weights_path)
    if (
        "dinov2_patch_cosine" in methods
        and not args.skip_deep_baselines
        and args.dinov2_repo_dir is not None
    ):
        snapshot = repository_snapshot(args.dinov2_repo_dir)
        model_inputs["dinov2_repository"] = {
            "path": str(args.dinov2_repo_dir.resolve()),
            "git_commit": snapshot.git_commit,
            "dirty_tree_fingerprint": snapshot.dirty_tree_fingerprint,
        }
    inventory_hash = canonical_hash(inventory)
    source_ids.append(
        ledger.append(
            NewEvent(
                "source.snapshot",
                {
                    "manifest": manifest_info,
                    "selected_inventory": inventory,
                    "selected_inventory_hash": inventory_hash,
                    "model_inputs": model_inputs,
                },
            )
        ).event_id
    )
    resolved = {
        key: (str(value.resolve()) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key != "ledger_dir"
    }
    config_hash = canonical_hash(resolved)
    source_ids.append(
        ledger.append(
            NewEvent(
                "config.snapshot",
                {"arguments": resolved, "arguments_hash": config_hash},
            )
        ).event_id
    )
    source_ids.append(
        ledger.append(
            NewEvent(
                "dataset.snapshot",
                {
                    "selected_pair_ids": pair_ids,
                    "selected_facade_ids": facade_ids,
                    "selected_source_inventory_hash": inventory_hash,
                    "fingerprint": canonical_hash(
                        {
                            "pair_ids": pair_ids,
                            "facade_ids": facade_ids,
                            "source_inventory_hash": inventory_hash,
                        }
                    ),
                },
            )
        ).event_id
    )
    source_ids.append(
        ledger.append(
            NewEvent(
                "split.snapshot",
                {
                    "definitions": definitions,
                    "fingerprint": canonical_hash(definitions),
                },
            )
        ).event_id
    )
    source_ids.append(
        ledger.append(
            NewEvent("environment.snapshot", _environment_snapshot(args, methods))
        ).event_id
    )
    return config_hash, source_ids


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")
    if args.deep_batch_size <= 0:
        raise ValueError("deep-batch-size must be positive")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    splits = [item.strip() for item in args.splits.split(",") if item.strip()] or None
    ledger = Ledger(args.ledger_dir, str(uuid4())) if args.ledger_dir else None
    config_hash, source_event_ids = "", []
    if ledger:
        ledger.append(
            NewEvent(
                "run.started",
                {
                    "entrypoint": "tools/run_temporal_change_baselines.py",
                    "invocation": "manual-cli",
                },
            )
        )
        try:
            report_path = args.out_csv.with_suffix(".report.json")
            if args.out_csv.exists() or report_path.exists():
                raise FileExistsError(
                    "ledger-enabled runs require new score and report paths"
                )
            config_hash, source_event_ids = _record_snapshots(
                ledger, args, methods, splits
            )
        except BaseException as exc:
            _append_failure(ledger, exc)
            raise

    def execute():
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        return compute_baseline_tile_scores(
            residual_manifest_csv=args.residual_manifest,
            out_scores_csv=args.out_csv,
            methods=methods,
            tile_size=args.tile_size,
            min_valid_ratio=args.min_valid_ratio,
            device=args.device,
            feature_cache_dir=args.feature_cache_dir,
            dinov2_model_name=args.dinov2_model_name,
            dinov2_cache_dir=args.dinov2_cache_dir,
            dinov2_weights_path=args.dinov2_weights_path,
            dinov2_repo_dir=args.dinov2_repo_dir,
            lpips_net=args.lpips_net,
            skip_deep_baselines=args.skip_deep_baselines,
            include_splits=splits,
            deep_batch_size=args.deep_batch_size,
            show_progress=not args.no_progress,
        )

    try:
        if ledger:
            with ledger.stage("temporal_change_baselines"):
                rows = execute()
            completed_stage = ledger.read()[-1]
        else:
            rows = execute()
            completed_stage = None
    except BaseException as exc:
        if ledger:
            _append_failure(ledger, exc)
        raise
    method_counts, split_counts = {}, {}
    for row in rows:
        method, split = str(row["method"]), str(row["split"])
        method_counts[method] = method_counts.get(method, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1
    report = {
        "residual_manifest": str(args.residual_manifest),
        "out_csv": str(args.out_csv),
        "methods_requested": methods,
        "splits_requested": splits,
        "tile_size": args.tile_size,
        "min_valid_ratio": args.min_valid_ratio,
        "device": args.device,
        "deep_batch_size": args.deep_batch_size,
        "n_score_rows": len(rows),
        "score_rows_by_method": method_counts,
        "score_rows_by_split_all_methods": split_counts,
        "valid_region_policy": "invalid aligned pixels excluded; low-coverage tiles dropped",
        "lpips_edge_tile_policy": "right/bottom partial tiles are edge-padded to fixed tile size before LPIPS inference",
    }
    report_path = args.out_csv.with_suffix(".report.json")
    try:
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if ledger and completed_stage:
            refs = source_event_ids + [completed_stage.event_id]
            for path, role, media_type in (
                (args.out_csv, "score_csv", "text/csv"),
                (report_path, "run_report", "application/json"),
            ):
                artifact = ArtifactDescriptor.from_path(
                    path,
                    role,
                    media_type,
                    "temporal_change_baselines",
                    config_hash,
                    refs,
                )
                ledger.append(NewEvent("artifact.created", artifact.to_dict()))
            ledger.append(NewEvent("run.completed", {}))
    except BaseException as exc:
        if ledger:
            _append_failure(ledger, exc)
        raise
    print("Built tile baseline score rows: {}".format(len(rows)))
    print("Rows by method: {}".format(method_counts))
    print("Rows by split across methods: {}".format(split_counts))
    print("Scores: {}".format(args.out_csv))
    print("Report: {}".format(report_path))
    if ledger:
        print("Run ID: {}".format(ledger.run_id))
        print("Ledger: {}".format(ledger.path.resolve()))


if __name__ == "__main__":
    main()
