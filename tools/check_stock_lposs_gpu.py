"""Opt-in real-GPU contract and pinned-upstream tensor parity check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

UPSTREAM_COMMIT = "e489a7445528922ddfe4e39631ef2fe34827c873"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--upstream-scores", required=True,
                        help="Pinned-upstream seed_scores and propagated_scores tensor mapping")
    parser.add_argument("--device", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "-C", args.upstream_root, "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream checkout is {commit}, expected {UPSTREAM_COMMIT}")
    if not torch.cuda.is_available():
        raise SystemExit("SKIP: CUDA unavailable")
    root = Path(args.work_dir)
    if root.exists():
        raise SystemExit(f"work directory already exists: {root}")
    reference = torch.load(args.upstream_scores, map_location="cpu", weights_only=True)
    outputs = {}
    for mode, config in (("maskclip_raw", "configs/stock_lposs.yaml"),
                         ("lposs", "configs/stock_lposs.yaml"),
                         ("lposs_plus", "configs/stock_lposs_plus.yaml")):
        target = root / mode
        subprocess.run([sys.executable, "-m", "ovs_heritage.infer_ovs", "--image", args.image,
            "--model-config", config, "--mode", mode, "--device", args.device,
            "--vocabulary", "ovs_heritage/configs/heritage_vocab.yaml",
            "--ornament-threshold", "0.5", "--output-dir", str(target), "--save-scores"], check=True)
        manifest = json.loads((target / "run_manifest.json").read_text())
        execution = manifest["records"][0]["execution"]
        expected = (mode != "maskclip_raw", mode != "maskclip_raw", mode == "lposs_plus")
        observed = (execution["dino_executed"], execution["patch_propagation_executed"],
                    execution["pixel_refinement_executed"])
        if observed != expected:
            raise AssertionError(f"{mode} stages {observed} != {expected}")
        score_path = next(target.glob("*/scores.pt"))
        scores = torch.load(score_path, map_location="cpu", weights_only=True)
        for key in ("seed_scores", "propagated_scores"):
            tensor = scores[key]
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"{mode} {key} is non-finite")
        outputs[mode] = scores
    for mode in ("maskclip_raw", "lposs", "lposs_plus"):
        for key in ("seed_scores", "propagated_scores"):
            expected = reference[mode][key]
            torch.testing.assert_close(outputs[mode][key], expected,
                                       atol=args.atol, rtol=args.rtol)
    print("Pinned upstream tensor parity passed.")


if __name__ == "__main__":
    main()
