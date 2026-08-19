"""Dataset-independent stock MaskCLIP/LPOSS inference CLI."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata as package_metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image
import torch
import yaml

from .metadata import make_metadata
from .ontology import load_ontology
from .projection import MAIN_SEMANTIC_IDS, OntologyProjection
from .stock_features import (StockFeatureModel, compatible_torch_load,
                             DINO_REPOSITORY, model_state_sha256, optional_weight_hash)
from .stock_lposs import (
    IMPLEMENTATION_ID,
    MODES,
    UPSTREAM_COMMIT,
    StockOVSEngine,
    UpstreamGraphPropagator,
    graph_preflight,
    metadata_dict,
    resolve_device,
)
from .vocabulary import PrototypeSet, RuntimeClass, build_prototypes, heritage_runtime_vocabulary

UPSTREAM_REPOSITORY = "https://github.com/vladan-stojnic/LPOSS"
SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})


@dataclass(frozen=True)
class InputSample:
    sample_id: str
    image_path: Path
    provenance: dict[str, Any] | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Genuine stock open-vocabulary LPOSS inference")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--image-dir")
    source.add_argument("--manifest")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--vocabulary", default=str(Path(__file__).parent / "configs/heritage_vocab.yaml"))
    parser.add_argument("--extra-concepts")
    parser.add_argument("--ornament-contrast", default=str(Path(__file__).parent / "configs/ornament_contrast_v1.yaml"))
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--inference", choices=("whole", "slide"), default="whole")
    parser.add_argument("--crop-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--stride", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--ornament-threshold", type=float)
    parser.add_argument("--ledger-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--prototype-artifact", help=argparse.SUPPRESS)
    parser.add_argument("--parity-feature-artifact", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def installed_versions() -> dict[str, str | None]:
    result = {}
    for distribution in ("open-clip-torch", "torch", "faiss-gpu", "faiss-cpu", "cupy-cuda11x", "cupy-cuda12x"):
        try:
            result[distribution] = package_metadata.version(distribution)
        except package_metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read stock config {path}: {exc}") from exc
    required = {"stock_implementation", "upstream_repository", "upstream_commit",
                "clip", "dino", "graph", "pixel_refine", "slide"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"{path} is not a complete dedicated stock LPOSS config")
    if config["stock_implementation"] != IMPLEMENTATION_ID:
        raise ValueError("incompatible stock implementation identifier")
    if config["upstream_repository"] != UPSTREAM_REPOSITORY or config["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("stock config has an incompatible upstream LPOSS reference")
    expected = {
        "clip": {"model": "ViT-B-16", "pretrained": "laion2b_s34b_b88k", "patch_size": 16, "image_size": 224},
        "dino": {"repository": DINO_REPOSITORY,
                 "model": "dino_vitb16", "source": "github", "weights": None,
                 "architecture": "vit_base", "patch_size": 16, "feature_type": "v"},
        "graph": {"alpha": .95, "gamma": 3.0, "k": 400, "sigma": .01,
                  "pix_dist_pow": 1.0, "tau": .01, "r": 13},
        "slide": {"crop_size": [512, 512], "stride": [341, 341]},
    }
    for section, values in expected.items():
        if not isinstance(config.get(section), dict) or any(config[section].get(k) != v for k, v in values.items()):
            raise ValueError(f"stock config changes scientifically meaningful {section} parameters")
    if not isinstance(config["pixel_refine"], bool):
        raise ValueError("pixel_refine must be boolean")
    return config


def _sample_id(value: Any, path: Path, line: int | None = None) -> str:
    sample = path.stem if value in (None, "") else value
    where = f" at manifest line {line}" if line else ""
    if not isinstance(sample, str) or not SAFE_SAMPLE_ID.fullmatch(sample) or sample in (".", ".."):
        raise ValueError(f"unsafe sample_id {sample!r}{where}")
    return sample


def discover_inputs(args, ontology=None) -> list[InputSample]:
    samples = []
    if args.image:
        path = Path(args.image).expanduser().resolve()
        samples.append(InputSample(_sample_id(None, path), path,
            {"input_kind": "image", "dataset_metadata_available": False}))
    elif args.image_dir:
        directory = Path(args.image_dir).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"image directory does not exist: {directory}")
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(InputSample(_sample_id(None, path), path.resolve(),
                    {"input_kind": "image_dir", "dataset_metadata_available": False}))
    else:
        manifest = Path(args.manifest).expanduser().resolve()
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"cannot read manifest {manifest}: {exc}") from exc
        manifest_sha256 = file_hash(manifest)
        facade_splits: dict[str, set[str]] = {}
        from .ontology import V2_VERSION
        from .validate_dataset import V2_DATASET_SCHEMA
        canonical_ontology = ontology or load_ontology()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest}: malformed JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(item, dict) or not isinstance(item.get("image_path"), str) or not item["image_path"].strip():
                raise ValueError(f"{manifest}: line {line_number} requires canonical image_path")
            path = Path(item["image_path"]).expanduser()
            path = path if path.is_absolute() else manifest.parent / path
            sample = item.get("sample_id", item.get("source_id", item.get("image_id")))
            canonical_fields = {
                "schema_version": item.get("schema_version"),
                "ontology_version": item.get("ontology_version"),
                "facade_id": item.get("facade_id"), "split": item.get("split"),
                "image_path": item.get("image_path"),
                "main_mask_path": item.get("main_mask_path"),
                "ornament_mask_path": item.get("ornament_mask_path"),
            }
            metadata_available = (canonical_fields["schema_version"] == V2_DATASET_SCHEMA
                and canonical_fields["ontology_version"] == V2_VERSION
                and all(isinstance(canonical_fields[k], str) and canonical_fields[k].strip()
                        for k in ("facade_id", "split", "image_path")))
            canonical = (metadata_available
                and all(isinstance(canonical_fields[k], str) and canonical_fields[k].strip()
                        for k in ("main_mask_path", "ornament_mask_path"))
                and item.get("ontology_hash", canonical_ontology.hash) == canonical_ontology.hash)
            if canonical_fields["schema_version"] not in (None, V2_DATASET_SCHEMA):
                raise ValueError(f"{manifest}: line {line_number} has incompatible dataset schema")
            if canonical_fields["ontology_version"] not in (None, V2_VERSION):
                raise ValueError(f"{manifest}: line {line_number} has incompatible ontology revision")
            if item.get("ontology_hash", canonical_ontology.hash) != canonical_ontology.hash:
                raise ValueError(f"{manifest}: line {line_number} has incompatible ontology hash")
            if (isinstance(canonical_fields["facade_id"], str)
                    and canonical_fields["facade_id"].strip()
                    and isinstance(canonical_fields["split"], str)
                    and canonical_fields["split"].strip()):
                facade = canonical_fields["facade_id"].strip()
                identity = canonical_fields["split"].strip()
                facade_splits.setdefault(facade, set()).add(identity)
            samples.append(InputSample(_sample_id(sample, path, line_number), path.resolve(), {
                "input_kind": "manifest", "dataset_metadata_available": metadata_available,
                "canonical_dataset_v2": canonical, "canonical_fields": canonical_fields,
                "facade_disjoint_split_verified": False,
                "facade_disjoint_split_evidence": "unavailable from a single inference manifest",
                "source_manifest_path": str(manifest), "source_manifest_sha256": manifest_sha256,
                "source_manifest_line_number": line_number, "source_record": item}))
        conflicts = {facade: sorted(values) for facade, values in facade_splits.items() if len(values) > 1}
        if conflicts:
            raise ValueError(f"facades assigned inconsistently across split identities: {conflicts}")
    if not samples:
        raise ValueError("input contains no supported images")
    ids = [item.sample_id for item in samples]
    paths = [item.image_path for item in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs are not allowed")
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate input paths are not allowed")
    for item in samples:
        try:
            with Image.open(item.image_path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"image is missing or unreadable: {item.image_path}: {exc}") from exc
    return samples


def prepare_output(path: Path, samples: list[InputSample]) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    targets = [output / item.sample_id for item in samples]
    if len(targets) != len(set(targets)) or any(output not in target.parents for target in targets):
        raise ValueError("sample output paths collide or escape the output directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def dataset_snapshot(samples: list[InputSample]) -> dict[str, Any]:
    """Payload for the existing research ledger; no parallel provenance store."""
    return {"inputs": [{"sample_id": item.sample_id, "path": str(item.image_path),
                        "sha256": file_hash(item.image_path), "provenance": item.provenance}
                       for item in samples]}


def load_runtime_classes(ontology, extra_path: str | None, contrast_path: Path):
    classes = list(heritage_runtime_vocabulary(ontology))
    if extra_path:
        try:
            raw = yaml.safe_load(Path(extra_path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot read extra concepts: {exc}") from exc
        if not isinstance(raw, list):
            raise ValueError("extra concepts must be a YAML list")
        for index, item in enumerate(raw):
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ValueError(f"extra concept {index} requires a name")
            prompts = item.get("prompts")
            if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p for p in prompts):
                raise ValueError(f"extra concept {index} requires nonempty string prompts")
            semantic_id = item.get("semantic_id")
            if semantic_id is not None:
                raise ValueError(
                    f"extra concept {index} semantic_id must be null; canonical IDs come from ontology v2")
            classes.append(RuntimeClass(item["name"], tuple(prompts), semantic_id=semantic_id))
    try:
        contrast = yaml.safe_load(contrast_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read ornament contrast: {exc}") from exc
    if not isinstance(contrast, dict) or contrast.get("semantic_id") is not None:
        raise ValueError("ornament contrast must be a semantic_id:null mapping")
    prompts = contrast.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("ornament contrast requires prompts")
    classes.append(RuntimeClass(contrast["name"], tuple(prompts), semantic_id=None))
    names, semantic = [item.name for item in classes], [item.semantic_id for item in classes if item.semantic_id is not None]
    if len(names) != len(set(names)) or len(semantic) != len(set(semantic)):
        raise ValueError("runtime classes contain duplicate names or semantic IDs")
    return classes, contrast


def validate_threshold(value):
    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("ornament_threshold must be finite and within [0, 1]")


def _atomic_png(path: Path, array: np.ndarray):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    Image.fromarray(array).save(temporary, format="PNG")
    os.replace(temporary, path)
    readback = np.asarray(Image.open(path))
    if not np.array_equal(readback, array):
        raise RuntimeError(f"PNG read-back mismatch: {path}")


def _atomic_torch(path: Path, payload):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    compatible_torch_load(path, map_location="cpu")


def export_sample(output: Path, sample: InputSample, result, prototypes,
                  save_scores: bool, visualize: bool) -> list[tuple[Path, str, str]]:
    directory = output / sample.sample_id
    directory.mkdir()
    if result.main_mask is None or result.ornament_mask is None:
        raise ValueError("canonical export requires complete main semantics and an ornament threshold")
    main = result.main_mask[0].cpu().numpy().astype(np.uint8)
    ornament = result.ornament_mask[0, 0].cpu().numpy().astype(np.uint8)
    if set(np.unique(main)) - set(MAIN_SEMANTIC_IDS):
        raise ValueError("main mask contains invalid semantic IDs")
    if set(np.unique(ornament)) - {0, 1}:
        raise ValueError("ornament mask must contain binary values 0/1")
    artifacts = []
    main_path, ornament_path = directory / "main_semantic.png", directory / "ornament_mask.png"
    _atomic_png(main_path, main)
    _atomic_png(ornament_path, ornament)
    artifacts.extend(((main_path, "main_semantic_mask", "image/png"),
                      (ornament_path, "ornament_binary_mask", "image/png")))
    if visualize:
        visual = directory / "ornament_visualization.png"
        _atomic_png(visual, ornament * 255)
        artifacts.append((visual, "ornament_visualization", "image/png"))
    if save_scores:
        score_path = directory / "scores.pt"
        payload = {"seed_scores": result.seed_scores.cpu(),
                   "propagated_scores": result.propagated_scores.cpu(),
                   "ornament_score": result.ornament_score.cpu(),
                   "ornament_probability": result.ornament_probability.cpu(),
                   "extra_scores": {key: value.cpu() for key, value in result.extra_scores.items()},
                   "channel_names": prototypes.channel_names,
                   "semantic_ids": prototypes.semantic_ids,
                   "vocabulary_hash": prototypes.vocabulary_hash,
                   "prompt_settings": dict(prototypes.prompt_settings)}
        payload["prototypes"] = prototypes.prototypes.cpu()
        _atomic_torch(score_path, payload)
        artifacts.append((score_path, "raw_score_tensors", "application/x-pytorch"))
    return artifacts


def _ledger_stage(ledger, name, function):
    if ledger is None:
        return function(), None
    with ledger.stage(name):
        result = function()
    return result, ledger.read()[-1]


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(0)
    validate_threshold(args.ornament_threshold)
    if args.ornament_threshold is None:
        raise ValueError("--ornament-threshold is required for the canonical binary output")
    config_path = Path(args.model_config).expanduser().resolve()
    config = load_config(config_path)
    if (args.mode == "lposs_plus") != bool(config["pixel_refine"]):
        raise ValueError("mode/config mismatch: use stock_lposs_plus.yaml only with lposs_plus")
    if args.mode == "lposs" and config["pixel_refine"]:
        raise ValueError("lposs requires stock_lposs.yaml with pixel_refine:false")
    ontology = load_ontology(args.vocabulary)
    samples = discover_inputs(args, ontology)
    if args.parity_feature_artifact and (len(samples) != 1 or args.inference != "whole"
                                         or args.mode == "maskclip_raw"):
        raise ValueError("parity feature capture requires one whole-inference graph mode sample")
    output = prepare_output(Path(args.output_dir), samples)
    projection = OntologyProjection.from_ontology(ontology)
    classes, contrast = load_runtime_classes(
        ontology, args.extra_concepts, Path(args.ornament_contrast).resolve())
    device = resolve_device(args.device)
    run_id = args.run_id or f"stock-lposs-{uuid4().hex}"
    ledger = None
    source_events = []
    if args.ledger_dir:
        from research_ledger import Ledger, NewEvent
        ledger = Ledger(args.ledger_dir, run_id)
        started = ledger.append(NewEvent("run.started", {"implementation": IMPLEMENTATION_ID,
            "mode": args.mode, "inference": args.inference, "output": str(output)}))
        source_events.append(started.event_id)
    try:
        config_hash = canonical_hash(config)
        if ledger:
            from research_ledger import NewEvent, repository_snapshot
            snapshot = repository_snapshot(Path(__file__).resolve().parents[1], exclude_paths=[output])
            for event_type, payload in (
                ("source.snapshot", {"git_commit": snapshot.git_commit,
                                     "dirty_tree_fingerprint": snapshot.dirty_tree_fingerprint,
                                     "upstream_repository": UPSTREAM_REPOSITORY,
                                     "upstream_commit": UPSTREAM_COMMIT}),
                ("config.snapshot", {"config": config, "canonical_hash": config_hash,
                                     "path": str(config_path), "sha256": file_hash(config_path)}),
                ("dataset.snapshot", dataset_snapshot(samples)),
                ("environment.snapshot", {"device": asdict(device), "torch": torch.__version__,
                                          "cuda": torch.version.cuda,
                                          "graph_dependencies": "pending mode-specific preflight"}),
            ):
                source_events.append(ledger.append(NewEvent(event_type, payload)).event_id)
            metadata = make_metadata(component_name=IMPLEMENTATION_ID, component_version="1",
                ontology_version=ontology.version, ontology_hash=ontology.hash,
                mapping=projection.as_dict(), vocabulary_specification_hash=None,
                ornament_threshold=args.ornament_threshold)
            source_events.append(ledger.append(NewEvent("ontology.snapshot", {"metadata": metadata.to_dict()})).event_id)
        dependencies, _ = _ledger_stage(
            ledger, "preflight", lambda: graph_preflight(args.mode, device))
        if ledger:
            source_events.append(ledger.append(NewEvent("environment.snapshot", {
                "device": asdict(device), "torch": torch.__version__,
                "cuda": torch.version.cuda, "graph_dependencies": dependencies})).event_id)

        def build_feature_model():
            clip = config["clip"]
            model = StockFeatureModel(clip_model=clip["model"], clip_pretrained=clip["pretrained"],
                                      patch_size=clip["patch_size"], image_size=clip["image_size"])
            model.parity_capture_enabled = args.parity_feature_artifact
            model.to(device.resolved_device).eval()
            if args.mode != "maskclip_raw":
                dino = config["dino"]
                model.configure_dino(repository=dino["repository"], model=dino["model"],
                    patch_size=dino["patch_size"], feature_type=dino["feature_type"],
                    source=dino["source"], weights=dino["weights"])
                model.to(device.resolved_device).eval()
            return model
        model, _ = _ledger_stage(ledger, "model_loading", build_feature_model)
        model_fingerprints = {"clip_state_sha256": model_state_sha256(model.clip),
            "dino_state_sha256": (model_state_sha256(model.dino_encoder)
                                  if model.dino_encoder is not None else None)}
        def construct_prototypes():
            if not args.prototype_artifact:
                return build_prototypes(classes, model.encode_text,
                    device=device.resolved_device, ontology_hash=ontology.hash)
            artifact = np.load(args.prototype_artifact, allow_pickle=False)
            metadata = json.loads(str(artifact["metadata_json"].item()))
            expected_names = [item.name for item in classes]
            expected_ids = [item.semantic_id for item in classes]
            if metadata.get("schema_version") != "lposs-prototypes-v1" or metadata.get("channel_names") != expected_names:
                raise ValueError("prototype artifact channel ordering is incompatible")
            if metadata.get("semantic_ids") != expected_ids or metadata.get("ontology_hash") != ontology.hash:
                raise ValueError("prototype artifact semantic IDs or ontology hash are incompatible")
            tensor = torch.from_numpy(artifact["prototypes"]).to(device.resolved_device)
            digest = sha256(artifact["prototypes"].tobytes(order="C")).hexdigest()
            if digest != metadata.get("prototypes_sha256"):
                raise ValueError("prototype artifact fingerprint mismatch")
            return PrototypeSet(tensor, tuple(expected_names), tuple(expected_ids),
                metadata["vocabulary_specification_hash"], ontology.hash, metadata["prompt_settings"])
        prototypes, _ = _ledger_stage(ledger, "prototype_construction", construct_prototypes)
        graph = dict(config["graph"])
        graph["vit_patch_size"] = config["dino"]["patch_size"]
        engine = StockOVSEngine(model, UpstreamGraphPropagator() if args.mode != "maskclip_raw" else None,
                                device=device, cosine_scale=1.0, graph_parameters=graph)
        crop = tuple(args.crop_size or config["slide"]["crop_size"])
        stride = tuple(args.stride or config["slide"]["stride"])
        records, artifact_rows, completed = [], [], []

        def run_inference():
            for sample in samples:
                rgb = np.asarray(Image.open(sample.image_path).convert("RGB"), dtype=np.float32) / 255.0
                image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
                if device.logical_index is not None:
                    torch.cuda.reset_peak_memory_stats(device.logical_index)
                    torch.cuda.synchronize(device.logical_index)
                    free_before, total_memory = torch.cuda.mem_get_info(device.logical_index)
                else:
                    free_before = total_memory = None
                started = time.perf_counter()
                result = engine.run(image, prototypes, mode=args.mode, inference=args.inference,
                    crop_size=crop, stride=stride, ornament_negative_index=len(classes) - 1,
                    ornament_threshold=args.ornament_threshold)
                if device.logical_index is not None:
                    torch.cuda.synchronize(device.logical_index)
                    free_after, _ = torch.cuda.mem_get_info(device.logical_index)
                else:
                    free_after = None
                peak = (torch.cuda.max_memory_allocated(device.logical_index)
                        if device.logical_index is not None else None)
                completed.append((sample, result, rgb.shape[:2],
                                  time.perf_counter() - started, peak, free_before, free_after, total_memory))

        _, inference_event = _ledger_stage(ledger, "inference_and_propagation", run_inference)

        def run_export():
            for sample, result, original_size, elapsed, peak, free_before, free_after, total in completed:
                artifacts = export_sample(
                    output, sample, result, prototypes, args.save_scores, args.visualize)
                artifact_rows.extend(artifacts)
                records.append({"sample_id": sample.sample_id, "image_path": str(sample.image_path),
                    "image_sha256": file_hash(sample.image_path), "original_size": list(original_size),
                    "output_grid": list(result.propagated_scores.shape[-2:]),
                    "execution": metadata_dict(result), "elapsed_seconds": elapsed,
                    "provenance": sample.provenance,
                    "peak_pytorch_allocated_bytes": peak,
                    "device_memory": {"free_before_bytes": free_before, "free_after_bytes": free_after,
                        "total_bytes": total, "limitation": "snapshots are not total peak GPU memory; PyTorch peak excludes CuPy and FAISS"},
                    "artifacts": [{"path": str(path), "byte_size": path.stat().st_size,
                                   "sha256": file_hash(path)} for path, _, _ in artifacts]})
            if args.parity_feature_artifact:
                clip_nodes, dino_nodes, patch_grid = model.parity_features()
                feature_path = output / "parity_features.npz"
                temporary = output / f".parity-features.{uuid4().hex}.tmp"
                with temporary.open("wb") as stream:
                    np.savez(stream, clip_features=clip_nodes.cpu().numpy(),
                             dino_features=dino_nodes.cpu().numpy(),
                             patch_grid=np.asarray(patch_grid, dtype=np.int64),
                             image_grid=np.asarray(completed[0][2], dtype=np.int64))
                os.replace(temporary, feature_path)
                artifact_rows.append((feature_path, "parity_normalized_dense_features",
                                      "application/x-npz"))
        _, export_event = _ledger_stage(ledger, "export", run_export)
        manifest = {"schema_version": "stock-ovs-lposs-run-v1", "run_id": run_id,
            "stock_implementation": IMPLEMENTATION_ID, "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT, "config": config, "config_hash": config_hash,
            "weights": {"clip_model": config["clip"]["model"], "clip_pretrained": config["clip"]["pretrained"],
                "clip_weight_sha256": optional_weight_hash(config["clip"]["pretrained"]),
                "dino_repository": config["dino"]["repository"], "dino_model": config["dino"]["model"],
                "dino_weights": config["dino"]["weights"],
                "dino_weight_sha256": optional_weight_hash(config["dino"]["weights"]),
                **model_fingerprints},
            "device": asdict(device), "dependencies": dependencies,
            "installed_versions": installed_versions(), "graph_parameters": graph,
            "ontology_hash": ontology.hash, "projection": projection.as_dict(),
            "projection_hash": canonical_hash(projection.as_dict()),
            "vocabulary_hash": prototypes.vocabulary_hash, "channel_names": list(prototypes.channel_names),
            "semantic_ids": list(prototypes.semantic_ids), "prompt_settings": dict(prototypes.prompt_settings),
            "runtime_vocabulary": [{"name": item.name, "semantic_id": item.semantic_id,
                                    "prompts": list(item.prompts), "aliases": list(item.aliases)}
                                   for item in classes],
            "ornament_contrast": contrast, "ornament_threshold": args.ornament_threshold,
            "determinism": {"seed": 0, "policy": "inference-only; fixed seed; no stochastic augmentation",
                            "torch_deterministic_algorithms": False},
            "records": records}
        manifest_path = output / "run_manifest.json"
        temporary = output / f".manifest.{uuid4().hex}.tmp"
        temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, manifest_path)
        json.loads(manifest_path.read_text())
        artifact_rows.append((manifest_path, "run_manifest", "application/json"))
        if ledger:
            from research_ledger import ArtifactDescriptor, NewEvent
            refs = source_events + [inference_event.event_id, export_event.event_id]
            for path, role, media in artifact_rows:
                descriptor = ArtifactDescriptor.from_path(path, role, media,
                    "export", config_hash, refs)
                ledger.append(NewEvent("artifact.created", descriptor.to_dict()))
            ledger.verify_artifacts()
            ledger.append(NewEvent("run.completed", {"status": "completed",
                                                      "artifact_count": len(artifact_rows)}))
    except BaseException as exc:
        if ledger:
            from research_ledger import NewEvent, sanitize_error
            try:
                ledger.append(NewEvent("run.failed", {"status": "failed", **sanitize_error(exc)}))
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main(sys.argv[1:])
