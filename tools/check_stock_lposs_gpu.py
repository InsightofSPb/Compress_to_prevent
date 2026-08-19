"""Auditable pinned-upstream versus local LPOSS CUDA parity workflow."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from ovs_heritage.stock_features import compatible_torch_load

UPSTREAM_COMMIT = "e489a7445528922ddfe4e39631ef2fe34827c873"
UPSTREAM_REPOSITORY = "https://github.com/vladan-stojnic/LPOSS"
MODES = ("maskclip_raw", "lposs", "lposs_plus")
REQUIRED_STAGES = {"maskclip_raw": {"seed_scores"},
                   "lposs": {"seed_scores", "propagated_scores"},
                   "lposs_plus": {"seed_scores", "propagated_scores", "pixel_refinement"}}


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def official_adapter_command(*, python: str, adapter: Path, request: Path,
                             output: Path) -> list[str]:
    return [python, str(adapter), "--request", str(request), "--output-dir", str(output)]


def verified_checkout_identity(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    commit = git("rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT or git("status", "--porcelain"):
        raise ValueError("upstream checkout must be at the pinned commit and clean")
    if "github.com/vladan-stojnic/LPOSS" not in git("remote", "-v"):
        raise ValueError("upstream checkout has no authoritative LPOSS remote")
    return {"repository": UPSTREAM_REPOSITORY, "commit": commit,
            "tree": git("rev-parse", "HEAD^{tree}")}


def validate_official_artifact(manifest_path: Path, *, expected: dict) -> tuple[dict, np.lib.npyio.NpzFile]:
    """Fail closed on fabricated, stale, partial, or provenance-mismatched output."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "official-lposs-parity-artifact-v1" or manifest.get("producer") != "official-upstream":
        raise ValueError("not a versioned official LPOSS parity artifact")
    upstream = manifest.get("upstream", {})
    if upstream != expected.get("upstream"):
        raise ValueError("official artifact has wrong repository, commit, or source-tree identity")
    checks = ("input_sha256", "prototype_artifact_sha256", "prototypes_sha256",
              "configurations", "configuration_sha256", "channel_names", "semantic_ids", "ontology_hash",
              "model_hashes", "device", "seed")
    for key in checks:
        if manifest.get(key) != expected.get(key):
            raise ValueError(f"official artifact provenance mismatch for {key}")
    if canonical_hash(manifest["configurations"]) != manifest["configuration_sha256"]:
        raise ValueError("official artifact configuration content does not match its fingerprint")
    dino_loads = manifest.get("dino_hub_loads", [])
    if len(dino_loads) != 1 or dino_loads[0].get("requested_repository") != "facebookresearch/dino:main":
        raise ValueError("official artifact lacks the single expected DINO interception")
    if (dino_loads[0].get("resolved_repository") !=
            "facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a"
            or dino_loads[0].get("model") != "dino_vitb16"):
        raise ValueError("official artifact did not resolve DINO to the pinned model revision")
    for mode, required in REQUIRED_STAGES.items():
        if not required.issubset(set(manifest.get("stages", {}).get(mode, []))):
            raise ValueError(f"official artifact is incomplete for {mode}")
    artifact = manifest_path.parent / manifest.get("stage_artifact", "")
    if not artifact.is_file() or file_hash(artifact) != manifest.get("stage_artifact_sha256"):
        raise ValueError("official stage artifact is missing or fingerprint-mismatched")
    arrays = np.load(artifact, allow_pickle=False)
    for mode in MODES:
        for stage in ("seed_scores", "propagated_scores"):
            key = f"{mode}.{stage}"
            if key not in arrays or arrays[key].ndim != 4 or not np.isfinite(arrays[key]).all():
                raise ValueError(f"official artifact lacks valid {key}")
    for feature in ("clip_features", "dino_features"):
        if feature not in arrays or arrays[feature].ndim != 2 or not np.isfinite(arrays[feature]).all():
            raise ValueError(f"official artifact lacks valid {feature}")
    if list(arrays["dino_features"].shape[:1]) != [manifest["patch_grid"][0] * manifest["patch_grid"][1]]:
        raise ValueError("official DINO feature shape disagrees with patch grid")
    return manifest, arrays


def compare_scores(reference: np.ndarray, observed: torch.Tensor, *, atol: float, rtol: float) -> float:
    expected = torch.from_numpy(reference)
    if expected.shape != observed.shape:
        raise AssertionError(f"shape mismatch: local {tuple(observed.shape)}, official {tuple(expected.shape)}")
    torch.testing.assert_close(observed.cpu(), expected, atol=atol, rtol=rtol)
    return float((observed.cpu() - expected).abs().max())


def main(argv=None):
    parser = argparse.ArgumentParser(description="real pinned-upstream LPOSS CUDA parity")
    parser.add_argument("--image", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: CPU contracts are not real numerical parity")
    local_root = Path(__file__).resolve().parents[1]
    image, upstream, work = Path(args.image).resolve(), Path(args.upstream_root).resolve(), Path(args.work_dir).resolve()
    if work.exists():
        raise SystemExit(f"work directory already exists: {work}")
    work.mkdir(parents=True)
    upstream_identity = verified_checkout_identity(upstream)

    common = ["--image", str(image), "--device", args.device,
        "--vocabulary", str(local_root / "ovs_heritage/configs/heritage_vocab.yaml"),
        "--ornament-threshold", "0.5", "--save-scores"]
    bootstrap = work / "local-maskclip_raw"
    subprocess.run([sys.executable, "-m", "ovs_heritage.infer_ovs", *common,
        "--model-config", str(local_root / "configs/stock_lposs.yaml"), "--mode", "maskclip_raw",
        "--output-dir", str(bootstrap)], cwd=local_root, check=True)
    bootstrap_manifest = json.loads((bootstrap / "run_manifest.json").read_text())
    bootstrap_scores = compatible_torch_load(next(bootstrap.glob("*/scores.pt")))
    prototypes_np = bootstrap_scores["prototypes"].numpy()
    prototype_meta = {"schema_version": "lposs-prototypes-v1",
        "channel_names": list(bootstrap_scores["channel_names"]),
        "semantic_ids": list(bootstrap_scores["semantic_ids"]),
        "ontology_hash": bootstrap_manifest["ontology_hash"],
        "vocabulary_specification_hash": bootstrap_scores["vocabulary_hash"],
        "prompt_settings": bootstrap_scores["prompt_settings"],
        "prototypes_sha256": sha256(prototypes_np.tobytes(order="C")).hexdigest()}
    prototype_artifact = work / "prototypes.npz"
    np.savez(prototype_artifact, prototypes=prototypes_np,
             metadata_json=np.asarray(json.dumps(prototype_meta, sort_keys=True)))
    config = bootstrap_manifest["config"]
    plus_config = deepcopy(config)
    plus_config["pixel_refine"] = True
    configurations = {"maskclip_raw": config, "lposs": config, "lposs_plus": plus_config}
    request = {"schema_version": "official-lposs-parity-request-v1", "local_root": str(local_root),
        "upstream_root": str(upstream), "image_path": str(image), "input_sha256": file_hash(image),
        "prototype_artifact": str(prototype_artifact), "prototype_artifact_sha256": file_hash(prototype_artifact),
        "prototypes_sha256": prototype_meta["prototypes_sha256"], "configurations": configurations,
        "configuration_sha256": canonical_hash(configurations), "channel_names": prototype_meta["channel_names"],
        "semantic_ids": prototype_meta["semantic_ids"], "ontology_hash": prototype_meta["ontology_hash"],
        "model_hashes": {k: bootstrap_manifest["weights"][k]
                         for k in ("clip_state_sha256", "dino_state_sha256")},
        "device": args.device, "seed": 0, "upstream": upstream_identity}
    if request["model_hashes"]["dino_state_sha256"] is None:
        # Bootstrap raw mode intentionally omits DINO; run lposs once with the exact prototypes.
        dino_bootstrap = work / "local-lposs"
        subprocess.run([sys.executable, "-m", "ovs_heritage.infer_ovs", *common,
            "--prototype-artifact", str(prototype_artifact), "--model-config", str(local_root / "configs/stock_lposs.yaml"),
            "--parity-feature-artifact", "--mode", "lposs", "--output-dir", str(dino_bootstrap)],
            cwd=local_root, check=True)
        dino_manifest = json.loads((dino_bootstrap / "run_manifest.json").read_text())
        request["model_hashes"] = {k: dino_manifest["weights"][k]
                                   for k in ("clip_state_sha256", "dino_state_sha256")}
    request_path = work / "request.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    official_dir = work / "official"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(upstream)
    command = official_adapter_command(python=sys.executable,
        adapter=local_root / "tools/run_official_lposs_parity.py", request=request_path, output=official_dir)
    subprocess.run(command, cwd=upstream, env=env, check=True)
    expected = {**request, "prototype_artifact_sha256": file_hash(prototype_artifact)}
    official_manifest, official = validate_official_artifact(official_dir / "manifest.json", expected=expected)
    local_features = np.load(work / "local-lposs/parity_features.npz", allow_pickle=False)
    if local_features["patch_grid"].tolist() != official_manifest["patch_grid"]:
        raise ValueError("local and official patch grids differ")
    if local_features["image_grid"].tolist() != official_manifest["image_grid"]:
        raise ValueError("local and official image grids differ")
    differences = {}
    shapes = {}
    for feature in ("clip_features", "dino_features"):
        observed = torch.from_numpy(local_features[feature])
        differences[feature] = compare_scores(official[feature], observed, atol=args.atol, rtol=args.rtol)
        shapes[feature] = {"official": list(official[feature].shape),
                           "local": list(local_features[feature].shape)}

    local_outputs = {"maskclip_raw": (bootstrap_manifest, bootstrap_scores)}
    for mode, config_name in (("lposs", "stock_lposs.yaml"), ("lposs_plus", "stock_lposs_plus.yaml")):
        target = work / f"local-{mode}"
        if target.exists():  # reuse the DINO bootstrap
            manifest = json.loads((target / "run_manifest.json").read_text())
            scores = compatible_torch_load(next(target.glob("*/scores.pt")))
        else:
            subprocess.run([sys.executable, "-m", "ovs_heritage.infer_ovs", *common,
                "--prototype-artifact", str(prototype_artifact), "--model-config", str(local_root / f"configs/{config_name}"),
                "--mode", mode, "--output-dir", str(target)], cwd=local_root, check=True)
            manifest = json.loads((target / "run_manifest.json").read_text())
            scores = compatible_torch_load(next(target.glob("*/scores.pt")))
        local_outputs[mode] = (manifest, scores)

    for mode, (manifest, scores) in local_outputs.items():
        if manifest["config"] != configurations[mode]:
            raise ValueError(f"local {mode} did not execute the requested resolved configuration")
        if list(scores["channel_names"]) != request["channel_names"] or list(scores["semantic_ids"]) != request["semantic_ids"]:
            raise ValueError(f"local {mode} channel/semantic identity mismatch")
        execution = manifest["records"][0]["execution"]
        expected_path = (mode != "maskclip_raw", mode != "maskclip_raw", mode == "lposs_plus")
        actual_path = (execution["dino_executed"], execution["patch_propagation_executed"], execution["pixel_refinement_executed"])
        if actual_path != expected_path:
            raise ValueError(f"local {mode} execution path {actual_path} != {expected_path}")
        if mode != "maskclip_raw" and (execution["requested_k"] != config["graph"]["k"]
                                       or execution["effective_k"] != config["graph"]["k"]):
            raise ValueError(f"local {mode} silently changed stock neighbor count")
        for stage in ("seed_scores", "propagated_scores"):
            key = f"{mode}.{stage}"
            differences[key] = compare_scores(
                official[f"{mode}.{stage}"], scores[stage], atol=args.atol, rtol=args.rtol)
            shapes[key] = {"official": list(official[key].shape),
                           "local": list(scores[stage].shape)}
    parity = {"schema_version": "lposs-upstream-parity-v2", "real_gpu_parity": True,
        "passed": True, "upstream": official_manifest["upstream"], "input_sha256": request["input_sha256"],
        "configurations": configurations, "configuration_sha256": request["configuration_sha256"],
        "prototype_artifact_sha256": request["prototype_artifact_sha256"],
        "prototypes_sha256": request["prototypes_sha256"], "model_hashes": request["model_hashes"],
        "channel_names": request["channel_names"], "semantic_ids": request["semantic_ids"],
        "device": args.device, "seed": 0, "atol": args.atol, "rtol": args.rtol,
        "maximum_absolute_differences": differences, "tensor_shapes": shapes,
        "patch_grid": official_manifest["patch_grid"], "image_grid": official_manifest["image_grid"],
        "modes": list(MODES),
        "artifacts": {"request": str(request_path), "prototypes": str(prototype_artifact),
                      "official_manifest": str(official_dir / "manifest.json")}}
    (work / "parity_manifest.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")
    print(json.dumps(parity, sort_keys=True))


if __name__ == "__main__":
    main()
