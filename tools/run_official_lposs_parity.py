"""Isolated driver for unmodified LPOSS at the reviewed upstream commit.

Run this with the official checkout first on ``PYTHONPATH``.  Communication with
the parent checker is exclusively through versioned JSON/NPZ files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

EXPECTED_COMMIT = "e489a7445528922ddfe4e39631ef2fe34827c873"
EXPECTED_REPOSITORY = "https://github.com/vladan-stojnic/LPOSS"
DINO_REQUESTED_REPOSITORY = "facebookresearch/dino:main"
DINO_RESOLVED_REPOSITORY = "facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a"
DINO_MODEL = "dino_vitb16"
LPOSS_CONSTRUCTOR_PARAMETERS = (
    "clip_backbone", "class_names", "vit_arch", "vit_patch_size", "enc_type_feats")


@contextmanager
def pinned_dino_hub_load(torch_module, calls):
    """Narrowly redirect the single hard-coded official DINO load, then restore it."""
    original = torch_module.hub.load

    def load(repository, model, *args, **kwargs):
        if repository != DINO_REQUESTED_REPOSITORY or model != DINO_MODEL:
            raise RuntimeError(f"unexpected official torch.hub.load request: {repository!r}, {model!r}")
        call = {"requested_repository": repository, "resolved_repository": DINO_RESOLVED_REPOSITORY,
                "model": model, "args": list(args), "kwargs": dict(kwargs)}
        calls.append(call)
        return original(DINO_RESOLVED_REPOSITORY, model, *args, **kwargs)

    torch_module.hub.load = load
    try:
        yield
    finally:
        torch_module.hub.load = original


def construct_official_lposs(lposs_class, torch_module, *, class_names, config):
    """Validate and call the exact constructor exposed by pinned official LPOSS."""
    parameters = tuple(inspect.signature(lposs_class).parameters)
    if parameters != LPOSS_CONSTRUCTOR_PARAMETERS:
        raise RuntimeError("pinned official LPOSS constructor drift: "
                           f"expected {LPOSS_CONSTRUCTOR_PARAMETERS}, observed {parameters}")
    calls = []
    with pinned_dino_hub_load(torch_module, calls):
        model = lposs_class(clip_backbone="maskclip", class_names=class_names,
            vit_arch=config["dino"]["architecture"], vit_patch_size=config["dino"]["patch_size"],
            enc_type_feats=config["dino"]["feature_type"])
    if len(calls) != 1:
        raise RuntimeError(f"official LPOSS must issue exactly one pinned DINO load; observed {len(calls)}")
    return model, calls


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkout_identity(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    commit = git("rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"official LPOSS commit mismatch: {commit}")
    if git("status", "--porcelain"):
        raise ValueError("official LPOSS checkout is dirty")
    remotes = git("remote", "-v")
    if "github.com/vladan-stojnic/LPOSS" not in remotes:
        raise ValueError(f"official LPOSS repository URL is not present in remotes: {remotes}")
    tree = git("rev-parse", "HEAD^{tree}")
    return {"repository": EXPECTED_REPOSITORY, "commit": commit, "tree": tree}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    request_path = Path(args.request).resolve()
    request = json.loads(request_path.read_text())
    upstream = Path(request["upstream_root"]).resolve()
    identity = checkout_identity(upstream)
    if identity != request.get("upstream"):
        raise ValueError("upstream checkout identity changed after request construction")
    if file_hash(Path(request["image_path"])) != request["input_sha256"]:
        raise ValueError("input fingerprint changed after parity request construction")
    if file_hash(Path(request["prototype_artifact"])) != request["prototype_artifact_sha256"]:
        raise ValueError("prototype artifact fingerprint changed after request construction")
    if canonical_hash(request["configurations"]) != request["configuration_sha256"]:
        raise ValueError("resolved configuration fingerprint mismatch")
    # Fail before importing a same-named local package. The verified upstream root
    # is the first import location and is also the working directory for configs.
    local_root = Path(request["local_root"]).resolve()
    sys.path[:] = [str(upstream)] + [p for p in sys.path
                                    if Path(p or ".").resolve() not in (local_root, local_root / "tools")]
    os.chdir(upstream)

    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from models.lposs.lposs import LPOSS
    from segmentation.evaluation.lposs_eval import (
        get_lposs_laplacian, get_lposs_plus_laplacian, perform_lp)

    def state_hash(module):
        digest = sha256()
        for key, value in sorted(module.state_dict().items()):
            cpu = value.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(tuple(cpu.shape)).encode())
            digest.update(str(cpu.dtype).encode())
            digest.update(cpu.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    torch.manual_seed(request["seed"])
    torch.cuda.manual_seed_all(request["seed"])
    torch.cuda.set_device(torch.device(request["device"]))
    proto_file = Path(request["prototype_artifact"])
    proto_npz = np.load(proto_file, allow_pickle=False)
    proto_meta = json.loads(str(proto_npz["metadata_json"].item()))
    prototypes_np = proto_npz["prototypes"]
    if sha256(prototypes_np.tobytes(order="C")).hexdigest() != request["prototypes_sha256"]:
        raise ValueError("prototype payload does not match request fingerprint")
    for key in ("channel_names", "semantic_ids", "ontology_hash"):
        if proto_meta.get(key) != request.get(key):
            raise ValueError(f"prototype metadata disagrees with request {key}")
    prototypes = torch.from_numpy(prototypes_np).to(request["device"])
    configurations = request["configurations"]
    config = configurations["lposs"]
    if configurations["maskclip_raw"] != config:
        raise ValueError("raw and LPOSS configurations unexpectedly differ")
    plus_expected = dict(config)
    plus_expected["pixel_refine"] = True
    if configurations["lposs_plus"] != plus_expected:
        raise ValueError("LPOSS+ configuration differs by more than pixel_refine")
    graph = config["graph"]
    model, dino_loads = construct_official_lposs(
        LPOSS, torch, class_names=proto_meta["channel_names"], config=config)
    model = model.to(request["device"]).eval()
    model.clip_backbone.decode_head.class_embeddings = prototypes
    model_hashes = {"clip_state_sha256": state_hash(model.clip_backbone.backbone),
                    "dino_state_sha256": state_hash(model.dino_encoder)}
    if model_hashes != request["model_hashes"]:
        raise ValueError(f"official/local loaded model fingerprints differ: {model_hashes} != {request['model_hashes']}")

    rgb = np.asarray(Image.open(request["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(request["device"])
    with torch.no_grad():
        dino, clip, classifier = model(image)
        if tuple(dino.shape[1:3]) != tuple(clip.shape[1:3]):
            clip = F.interpolate(clip.permute(0, 3, 1, 2), dino.shape[1:3],
                                 mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        h, w = dino.shape[1:3]
        if h * w < graph["k"]:
            raise RuntimeError(f"official stock k={graph['k']} requires at least {graph['k']} patch nodes; got {h*w}")
        dino_nodes = F.normalize(dino[0].reshape(h * w, -1), dim=-1)
        clip_nodes = F.normalize(clip[0].reshape(h * w, -1), dim=-1)
        seed_nodes = clip_nodes @ prototypes.T
        locations = torch.tensor([[0, image.shape[-2], 0, image.shape[-1]]], device=image.device)
        laplacian = get_lposs_laplacian(dino_nodes, locations, [(h, w)],
            sigma=graph["sigma"], pix_dist_pow=graph["pix_dist_pow"], k=graph["k"],
            gamma=graph["gamma"], alpha=graph["alpha"], patch_size=config["dino"]["patch_size"])
        propagated_nodes = perform_lp(laplacian, seed_nodes)
        seed_full = F.interpolate(seed_nodes.T.reshape(1, -1, h, w), image.shape[-2:],
                                  mode="bilinear", align_corners=False)
        propagated_full = F.interpolate(propagated_nodes.T.reshape(1, -1, h, w), image.shape[-2:],
                                        mode="bilinear", align_corners=False)
        flat = propagated_full[0].permute(1, 2, 0).reshape(-1, prototypes.shape[0])
        pixel_laplacian = get_lposs_plus_laplacian(image, flat, tau=graph["tau"],
            neigh=graph["r"] // 2, alpha=graph["alpha"])
        refined = perform_lp(pixel_laplacian, flat).reshape(
            image.shape[-2], image.shape[-1], -1).permute(2, 0, 1).unsqueeze(0)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True)
    arrays = {"clip_features": clip_nodes.detach().cpu().numpy(),
        "dino_features": dino_nodes.detach().cpu().numpy(),
        "maskclip_raw.seed_scores": seed_full.detach().cpu().numpy(),
        "maskclip_raw.propagated_scores": seed_full.detach().cpu().numpy(),
        "lposs.seed_scores": seed_full.detach().cpu().numpy(),
        "lposs.propagated_scores": propagated_full.detach().cpu().numpy(),
        "lposs_plus.seed_scores": seed_full.detach().cpu().numpy(),
        "lposs_plus.propagated_scores": refined.detach().cpu().numpy()}
    np.savez(output / "stages.npz", **arrays)
    manifest = {"schema_version": "official-lposs-parity-artifact-v1", "producer": "official-upstream",
        "upstream": identity, "input_sha256": file_hash(Path(request["image_path"])),
        "prototype_artifact_sha256": file_hash(proto_file), "prototypes_sha256": request["prototypes_sha256"],
        "configurations": configurations, "configuration_sha256": request["configuration_sha256"],
        "channel_names": proto_meta["channel_names"], "semantic_ids": proto_meta["semantic_ids"],
        "ontology_hash": proto_meta["ontology_hash"], "patch_grid": [h, w],
        "image_grid": list(image.shape[-2:]), "device": request["device"], "seed": request["seed"],
        "model_hashes": model_hashes,
        "dino_hub_loads": dino_loads,
        "stages": {"maskclip_raw": ["seed_scores"],
                   "lposs": ["seed_scores", "propagated_scores"],
                   "lposs_plus": ["seed_scores", "propagated_scores", "pixel_refinement"]},
        "stage_artifact": "stages.npz", "stage_artifact_sha256": file_hash(output / "stages.npz")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
