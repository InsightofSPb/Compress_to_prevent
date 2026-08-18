from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from research_ledger import Ledger

ROOT = Path(__file__).parents[1]


def _fixture(tmp_path: Path) -> Path:
    prev, curr = tmp_path / "prev.ppm", tmp_path / "curr.ppm"
    header = b"P6\n4 4\n255\n"
    prev.write_bytes(header + bytes([0, 0, 0]) * 16)
    curr.write_bytes(header + bytes([10, 10, 10]) * 16)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "pair_id",
                "facade_id",
                "split",
                "prev_aligned_path",
                "curr_image_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "p1",
                "facade_id": "f1",
                "split": "test",
                "prev_aligned_path": prev,
                "curr_image_path": curr,
            }
        )
    return manifest


def _run(
    manifest: Path, output: Path, ledger: Path | None = None
) -> subprocess.CompletedProcess[str]:
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
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)


def test_synthetic_cpu_cli_ledger_and_unchanged_output(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    plain_out, recorded_out = tmp_path / "plain.csv", tmp_path / "recorded.csv"
    plain = _run(manifest, plain_out)
    recorded = _run(manifest, recorded_out, tmp_path / "ledger")
    assert plain.stdout.replace(str(plain_out), "OUT") == recorded.stdout.replace(
        str(recorded_out), "OUT"
    )
    assert plain_out.read_text() == recorded_out.read_text()
    run_dir = next((tmp_path / "ledger").iterdir())
    ledger = Ledger(tmp_path / "ledger", run_dir.name)
    ledger.verify()
    assert ledger.reconstruct().status == "completed"
    artifact = next(
        event for event in ledger.read() if event.event_type == "artifact.created"
    )
    assert artifact.payload["sha256"] == sha256(recorded_out.read_bytes()).hexdigest()
    assert (
        json.loads(recorded_out.with_suffix(".report.json").read_text())["n_score_rows"]
        == 4
    )
