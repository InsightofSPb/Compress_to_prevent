from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Optional

import pytest

from research_ledger import Ledger, canonical_hash
from tools.run_temporal_change_baselines import _validate_and_inventory

ROOT = Path(__file__).parents[1]


def _ppm(path: Path, value: int, width: int = 4, height: int = 4) -> None:
    path.write_bytes(
        "P6\n{} {}\n255\n".format(width, height).encode()
        + bytes([value] * width * height * 3)
    )


def _fixture(tmp_path: Path, rows=None) -> Path:
    prev, curr = tmp_path / "prev.ppm", tmp_path / "curr.ppm"
    _ppm(prev, 0)
    _ppm(curr, 10)
    manifest = tmp_path / "manifest.csv"
    fields = [
        "pair_id",
        "facade_id",
        "split",
        "prev_aligned_path",
        "curr_image_path",
        "valid_mask_path",
    ]
    if rows is None:
        rows = [
            {
                "pair_id": "p1",
                "facade_id": "f1",
                "split": "test",
                "prev_aligned_path": prev,
                "curr_image_path": curr,
            }
        ]
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _run(
    manifest: Path, output: Path, ledger: Optional[Path] = None, check: bool = True
):
    command = [
        sys.executable,
        str(ROOT / "tools/run_temporal_change_baselines.py"),
        "--residual-manifest",
        str(manifest),
        "--out-csv",
        str(output),
        "--methods",
        "absdiff_l1",
        "--tile-size",
        "2",
        "--no-progress",
    ]
    if ledger:
        command += ["--ledger-dir", str(ledger)]
    return subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=check
    )


def _ledger(root: Path) -> Ledger:
    run_dir = sorted(root.iterdir())[-1]
    return Ledger(root, run_dir.name)


def test_synthetic_cpu_cli_artifacts_parity_and_prints(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    plain_out, recorded_out = tmp_path / "plain.csv", tmp_path / "recorded.csv"
    _run(manifest, plain_out)
    recorded = _run(manifest, recorded_out, tmp_path / "ledger")
    assert plain_out.read_text() == recorded_out.read_text()
    plain_report = json.loads(plain_out.with_suffix(".report.json").read_text())
    ledger_report = json.loads(recorded_out.with_suffix(".report.json").read_text())
    plain_report.update({"out_csv": "OUT", "residual_manifest": "MANIFEST"})
    ledger_report.update({"out_csv": "OUT", "residual_manifest": "MANIFEST"})
    assert plain_report == ledger_report
    ledger = _ledger(tmp_path / "ledger")
    ledger.verify()
    ledger.verify_artifacts()
    assert ledger.reconstruct().status == "completed"
    artifacts = [
        event.payload
        for event in ledger.read()
        if event.event_type == "artifact.created"
    ]
    assert {item["logical_role"] for item in artifacts} == {"score_csv", "run_report"}
    assert "Run ID: {}".format(ledger.run_id) in recorded.stdout
    assert "Ledger: {}".format(ledger.path.resolve()) in recorded.stdout


def test_selected_inventory_filter_and_image_mutation(tmp_path: Path) -> None:
    paths = []
    for name, value in (("a0", 0), ("a1", 1), ("b0", 2), ("b1", 3)):
        path = tmp_path / (name + ".ppm")
        _ppm(path, value)
        paths.append(path)
    manifest = _fixture(
        tmp_path,
        [
            {
                "pair_id": "train-p",
                "facade_id": "train-f",
                "split": "train",
                "prev_aligned_path": paths[0],
                "curr_image_path": paths[1],
            },
            {
                "pair_id": "test-p",
                "facade_id": "test-f",
                "split": "test",
                "prev_aligned_path": paths[2],
                "curr_image_path": paths[3],
            },
        ],
    )
    inventory, definitions, pairs, facades = _validate_and_inventory(manifest, ["test"])
    assert pairs == ["test-p"] and facades == ["test-f"]
    assert definitions == {"test": ["test-f"], "train": ["train-f"]}
    before = canonical_hash(inventory)
    _ppm(paths[3], 9)
    after = canonical_hash(_validate_and_inventory(manifest, ["test"])[0])
    assert before != after


@pytest.mark.parametrize("field", ["pair_id", "facade_id", "split"])
def test_missing_manifest_identity_rejected(tmp_path: Path, field: str) -> None:
    prev, curr = tmp_path / "p.ppm", tmp_path / "c.ppm"
    _ppm(prev, 0)
    _ppm(curr, 1)
    row = {
        "pair_id": "p",
        "facade_id": "f",
        "split": "test",
        "prev_aligned_path": prev,
        "curr_image_path": curr,
    }
    row[field] = ""
    with pytest.raises(ValueError, match=field):
        _validate_and_inventory(_fixture(tmp_path, [row]), None)


def test_duplicate_pair_leakage_and_missing_file_rejected(tmp_path: Path) -> None:
    prev, curr = tmp_path / "p.ppm", tmp_path / "c.ppm"
    _ppm(prev, 0)
    _ppm(curr, 1)
    common = {"prev_aligned_path": prev, "curr_image_path": curr}
    duplicate = _fixture(
        tmp_path,
        [
            dict(common, pair_id="p", facade_id="f1", split="test"),
            dict(common, pair_id="p", facade_id="f2", split="test"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        _validate_and_inventory(duplicate, None)
    leakage = _fixture(
        tmp_path,
        [
            dict(common, pair_id="p1", facade_id="f", split="train"),
            dict(common, pair_id="p2", facade_id="f", split="test"),
        ],
    )
    with pytest.raises(ValueError, match="overlaps"):
        _validate_and_inventory(leakage, None)
    missing = _fixture(
        tmp_path,
        [
            dict(
                common,
                pair_id="p",
                facade_id="f",
                split="test",
                curr_image_path=tmp_path / "missing.ppm",
            )
        ],
    )
    with pytest.raises(FileNotFoundError):
        _validate_and_inventory(missing, None)


def test_ledger_run_refuses_overwrite(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    output = tmp_path / "scores.csv"
    root = tmp_path / "ledger"
    _run(manifest, output, root)
    failed = _run(manifest, output, root, check=False)
    assert (
        failed.returncode != 0 and "require new score and report paths" in failed.stderr
    )
    failed_ledger = next(
        candidate
        for directory in root.iterdir()
        for candidate in [Ledger(root, directory.name)]
        if [event.event_type for event in candidate.read()]
        == ["run.started", "run.failed"]
    )
    assert [event.event_type for event in failed_ledger.read()] == [
        "run.started",
        "run.failed",
    ]


def test_cli_scoring_failure_records_stage_and_one_run_failure(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    # A model-visible file exists and inventories successfully, but its shape mismatches during scoring.
    _ppm(tmp_path / "curr.ppm", 1, width=3, height=3)
    result = _run(manifest, tmp_path / "scores.csv", tmp_path / "ledger", check=False)
    assert result.returncode != 0
    ledger = _ledger(tmp_path / "ledger")
    types = [event.event_type for event in ledger.read()]
    assert types.count("stage.failed") == 1 and types.count("run.failed") == 1
    assert types[-1] == "run.failed"
