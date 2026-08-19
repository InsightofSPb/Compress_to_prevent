"""Generate pinned-official LPOSS references and compare them with this repository.

This is deliberately an opt-in CUDA check.  It never accepts an unprovenanced tensor file.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import torch

from ovs_heritage.stock_features import compatible_torch_load

UPSTREAM_COMMIT = "e489a7445528922ddfe4e39631ef2fe34827c873"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_score_mappings(reference, local, *, atol: float, rtol: float):
    """Contract shared by fake CPU tests and the numerical CUDA workflow."""
    differences = {}
    for stage in ("seed_scores", "propagated_scores"):
        if stage not in reference or stage not in local:
            raise AssertionError(f"missing execution stage {stage}")
        expected, observed = reference[stage], local[stage]
        if expected.shape != observed.shape:
            raise AssertionError(f"{stage} shape {tuple(observed.shape)} != {tuple(expected.shape)}")
        if reference.get("channel_names") != local.get("channel_names"):
            raise AssertionError("channel order differs")
        if reference.get("semantic_ids") != local.get("semantic_ids"):
            raise AssertionError("semantic IDs differ")
        torch.testing.assert_close(observed, expected, atol=atol, rtol=rtol)
        differences[stage] = float((observed - expected).abs().max())
    return differences


def main(argv=None):
    parser = argparse.ArgumentParser(description="real pinned-upstream LPOSS CUDA parity")
    parser.add_argument("--image", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    upstream = Path(args.upstream_root).resolve()
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream checkout is {commit}, expected {UPSTREAM_COMMIT}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise SystemExit("official upstream checkout must be clean; modified semantics are not accepted")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: CPU contract tests are not numerical upstream parity")
    image = Path(args.image).resolve()
    root = Path(args.work_dir).resolve()
    if root.exists():
        raise SystemExit(f"work directory already exists: {root}")
    root.mkdir(parents=True)
    # The adapter lives in this repository, but imports feature extraction and graph
    # propagation from the verified official checkout rather than accepting tensors.
    reference = root / "official"
    subprocess.run([sys.executable, "tools/run_official_lposs_parity.py",
        "--upstream-root", str(upstream), "--image", str(image), "--device", args.device,
        "--output-dir", str(reference)], check=True)
    local = root / "local"
    subprocess.run([sys.executable, "-m", "ovs_heritage.infer_ovs", "--image", str(image),
        "--model-config", "configs/stock_lposs.yaml", "--mode", "lposs", "--device", args.device,
        "--vocabulary", "ovs_heritage/configs/heritage_vocab.yaml", "--ornament-threshold", "0.5",
        "--output-dir", str(local), "--save-scores"], check=True)
    official_scores = compatible_torch_load(reference / "scores.pt")
    local_scores_path = next(local.glob("*/scores.pt"))
    local_scores = compatible_torch_load(local_scores_path)
    differences = compare_score_mappings(official_scores, local_scores, atol=args.atol, rtol=args.rtol)
    local_manifest = json.loads((local / "run_manifest.json").read_text())
    manifest = {"schema_version": "lposs-upstream-parity-v1", "real_gpu_parity": True,
        "upstream_commit": commit, "input_sha256": sha256_file(image), "device": args.device,
        "atol": args.atol, "rtol": args.rtol, "max_absolute_differences": differences,
        "resolved_configuration": local_manifest["config"], "weights": local_manifest["weights"],
        "channel_names": local_scores["channel_names"], "semantic_ids": local_scores["semantic_ids"],
        "stages": ["seed_scores", "propagated_scores"]}
    (root / "parity_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
