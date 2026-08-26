import argparse
import builtins
from hashlib import sha256
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from ovs_heritage.infer_ovs import (
    InputSample,
    dataset_snapshot,
    discover_inputs,
    export_sample,
    load_config,
    prepare_output,
    validate_threshold,
    _ledger_stage,
)
from ovs_heritage.projection import MAIN_SEMANTIC_IDS
from ovs_heritage.ontology import load_ontology
from ovs_heritage.stock_features import (
    DINO_REPOSITORY, StockFeatureModel, model_state_sha256, module_state_fingerprint,
    validate_pinned_hub_repository,
)
from ovs_heritage.stock_lposs import (
    DeviceInfo,
    StockOVSEngine,
    graph_preflight,
    patch_graph_preflight,
    pixel_graph_preflight,
    resolve_device,
)
from ovs_heritage.vocabulary import PrototypeSet


class FakeModel:
    def __init__(self):
        self.clip_calls = 0
        self.dino_calls = 0

    def clip_dense(self, image):
        self.clip_calls += 1
        return image[:, :2]

    def dino_dense(self, image):
        self.dino_calls += 1
        return torch.ones(image.shape[0], 2, 2, 2, device=image.device)


def test_scalar_state_hashing_is_deterministic_and_sensitive():
    module = nn.Module()
    module.register_parameter("scalar", nn.Parameter(torch.tensor(1.0)))
    module.register_buffer("matrix", torch.arange(6).reshape(2, 3))

    fingerprint = module_state_fingerprint(module)
    state_hash = model_state_sha256(module)
    assert module_state_fingerprint(module) == fingerprint
    assert model_state_sha256(module) == state_hash

    module.scalar.data.fill_(2.0)
    assert module_state_fingerprint(module) != fingerprint
    assert model_state_sha256(module) != state_hash


class FakePropagation:
    def __init__(self):
        self.patch_calls = 0
        self.pixel_calls = 0
        self.locations = None

    def patch_nodes(self, dino, seeds, **kwargs):
        self.patch_calls += 1
        self.locations = kwargs["locations"].cpu()
        return seeds + 1, min(kwargs["parameters"]["k"], dino.shape[0])

    def pixel(self, image, scores, **kwargs):
        self.pixel_calls += 1
        return scores + 2


def prototype_set(include_extra=True):
    ids = list(MAIN_SEMANTIC_IDS) + [8, None]
    names = [f"class_{i}" for i in MAIN_SEMANTIC_IDS] + ["ornament_region", "negative"]
    if include_extra:
        ids.append(None)
        names.append("extra")
    vectors = torch.tensor([[1.0, 0.0] if i % 2 else [0.0, 1.0]
                            for i in range(len(ids))])
    return PrototypeSet(vectors, tuple(names), tuple(ids), "vocab", "ontology", {})


def engine(mode):
    model, propagation = FakeModel(), FakePropagation()
    configured = None if mode == "maskclip_raw" else propagation
    instance = StockOVSEngine(configured and model or model, configured,
        device=DeviceInfo("cpu", "cpu", None, None) if mode == "maskclip_raw"
        else DeviceInfo("cuda:0", "cpu", 0, "fake"),
        graph_parameters={"k": 99, "r": 13, "available_gpu_bytes": 10**12,
                          "gpu_memory_reserve_bytes": 0})
    return instance, model, propagation


@pytest.mark.parametrize("mode,expected", [
    ("maskclip_raw", (0, 0, 0)),
    ("lposs", (1, 1, 0)),
    ("lposs_plus", (1, 1, 1)),
])
def test_exact_stage_routing_and_truthful_metadata(monkeypatch, mode, expected):
    monkeypatch.setattr(torch.cuda, "device", lambda _index: __import__("contextlib").nullcontext())
    instance, model, propagation = engine(mode)
    result = instance.run(torch.rand(1, 3, 5, 7), prototype_set(), mode=mode,
                          ornament_negative_index=12, ornament_threshold=0.5)
    assert (model.dino_calls, propagation.patch_calls, propagation.pixel_calls) == expected
    assert result.propagated_scores.shape == (1, 14, 5, 7)
    assert result.main_scores.shape == (1, 11, 5, 7)
    assert result.main_mask.shape == (1, 5, 7)
    assert result.ornament_mask.shape == (1, 1, 5, 7)
    assert set(result.extra_scores) == {"extra"}
    assert result.metadata.dino_executed == bool(expected[0])
    assert result.metadata.patch_propagation_executed == bool(expected[1])
    assert result.metadata.pixel_refinement_executed == bool(expected[2])


def test_true_slide_builds_one_location_aware_graph_and_overlap_average(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda _index: __import__("contextlib").nullcontext())
    instance, model, propagation = engine("lposs")
    result = instance.run(torch.rand(1, 3, 6, 7), prototype_set(), mode="lposs",
        inference="slide", crop_size=(4, 4), stride=(3, 3),
        ornament_negative_index=12, ornament_threshold=0.5)
    assert model.clip_calls == model.dino_calls == propagation.patch_calls == 1
    assert propagation.locations.tolist() == [[0, 4, 0, 4], [0, 4, 3, 7],
                                               [2, 6, 0, 4], [2, 6, 3, 7]]
    assert result.propagated_scores.shape[-2:] == (6, 7)
    assert result.metadata.window_count == 4
    assert result.metadata.graph_nodes == 16


def test_raw_preflight_never_imports_graph_dependencies(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith(("cupy", "cupyx", "faiss")):
            raise AssertionError(f"unexpected graph import: {name}")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert graph_preflight("maskclip_raw", DeviceInfo("cpu", "cpu", None, None)) == {
        "cupy": None, "faiss": None}


def test_requested_cuda_index_is_validated_and_resolved(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    resolved = resolve_device("cuda:1")
    assert resolved == DeviceInfo("cuda:1", "cuda:1", 1, "GPU-1")
    with pytest.raises(RuntimeError, match="only 2"):
        resolve_device("cuda:2")


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4.0).reshape(2, 2))


class FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeBlock()
        self.visual = types.SimpleNamespace(
            positional_embedding=nn.Parameter(torch.ones(5, 2)),
            transformer=types.SimpleNamespace(resblocks=[FakeBlock(), FakeBlock()]),
        )


def test_actual_feature_model_state_is_runtime_vocabulary_invariant(monkeypatch):
    fake = FakeClip()
    module = types.ModuleType("open_clip")
    module.create_model_from_pretrained = lambda *_args, **_kwargs: (fake, None)
    module.get_tokenizer = lambda _name: lambda text: torch.tensor([[len(text)]])
    monkeypatch.setitem(sys.modules, "open_clip", module)
    model = StockFeatureModel(clip_model="fake", clip_pretrained="fake",
                              patch_size=2, image_size=4)
    before = module_state_fingerprint(model)
    for classes in (("a", "b"), ("b",), ("novel", "a", "b")):
        prototypes = PrototypeSet(torch.eye(len(classes), 2), classes,
            tuple(None for _ in classes), sha256(repr(classes).encode()).hexdigest(), None, {})
        assert prototypes.channel_names == classes
        assert module_state_fingerprint(model) == before


def _args(**kwargs):
    defaults = {"image": None, "image_dir": None, "manifest": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_dataset_v2_jsonl_resolves_relative_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(image_dir / "a.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('\n{"sample_id":"sample-a","image_path":"images/a.png"}\n')
    result = discover_inputs(_args(manifest=str(manifest)))
    assert result[0].sample_id == "sample-a"
    assert result[0].image_path == (image_dir / "a.png").resolve()
    assert result[0].provenance["source_record"]["image_path"] == "images/a.png"
    assert result[0].provenance["source_manifest_sha256"] == sha256(manifest.read_bytes()).hexdigest()
    assert result[0].provenance["dataset_metadata_available"] is False


def test_source_manifest_hash_is_computed_once(tmp_path, monkeypatch):
    import ovs_heritage.infer_ovs as inference
    for name in ("a.png", "b.png"):
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tmp_path / name)
    manifest = tmp_path / "generic.jsonl"
    manifest.write_text("\n".join(json.dumps({"sample_id": name[0], "image_path": name})
                                  for name in ("a.png", "b.png")))
    calls = []
    original = inference.file_hash
    monkeypatch.setattr(inference, "file_hash", lambda path: calls.append(Path(path)) or original(path))
    samples = discover_inputs(_args(manifest=str(manifest)))
    assert calls == [manifest.resolve()]
    assert {sample.provenance["source_manifest_sha256"] for sample in samples} == {
        sha256(manifest.read_bytes()).hexdigest()}
    snapshot = dataset_snapshot(samples)
    assert snapshot["inputs"][0]["provenance"]["source_record"]["image_path"] == "a.png"


def _converter_manifest_fixture(tmp_path):
    ontology = load_ontology()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), "RGB").save(tmp_path / "a.png")
    Image.fromarray(np.array([[0, 1], [2, 11]], dtype=np.uint8)).save(tmp_path / "main.png")
    Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).save(tmp_path / "ornament.png")
    row = {"sample_id": "a", "image_id": 1, "source_coco_file_name": "a.png",
        "canonical_file_name": "a.png", "resolved_image_path": str((tmp_path / "a.png").resolve()),
        "image_path": "a.png", "main_mask_path": "main.png", "ornament_mask_path": "ornament.png",
        "facade_id": "facade-a", "building_id": "building-a", "split": "test",
        "schema_version": "heritage_two_map_v2", "ontology_version": ontology.version,
        "source_coco_sha256": "a" * 64, "source_annotation_ids": [1], "width": 2, "height": 2}
    manifest = tmp_path / "canonical.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return manifest, row


def test_authoritatively_valid_converter_manifest_is_canonical(tmp_path, monkeypatch):
    import ovs_heritage.coco_converter as converter
    manifest, _ = _converter_manifest_fixture(tmp_path)
    calls = []
    original = converter.validate_manifest
    monkeypatch.setattr(converter, "validate_manifest",
                        lambda path: calls.append(path) or original(path))
    sample = discover_inputs(_args(manifest=str(manifest)))[0]
    assert calls == [manifest.resolve()]
    assert sample.provenance["dataset_metadata_available"] is True
    assert sample.provenance["canonical_dataset_v2"] is True
    assert sample.provenance["facade_disjoint_split_verified"] is False
    assert sample.provenance["canonical_fields"]["facade_id"] == "facade-a"


@pytest.mark.parametrize("defect,expected", [
    ("missing_mask", "unreadable artifact"),
    ("invalid_split", "invalid split"),
    ("invalid_domain", "invalid main-mask value domain"),
    ("grid_mismatch", "image/mask grid or dtype mismatch"),
])
def test_declared_canonical_manifest_defects_fail_closed(tmp_path, defect, expected):
    manifest, row = _converter_manifest_fixture(tmp_path)
    if defect == "missing_mask":
        (tmp_path / "main.png").unlink()
    elif defect == "invalid_split":
        row["split"] = "holdout"
        manifest.write_text(json.dumps(row) + "\n")
    elif defect == "invalid_domain":
        Image.fromarray(np.full((2, 2), 8, dtype=np.uint8)).save(tmp_path / "main.png")
    else:
        Image.fromarray(np.zeros((1, 2), dtype=np.uint8)).save(tmp_path / "main.png")
    with pytest.raises(ValueError, match=f"canonical dataset-v2 manifest validation failed:.*{expected}"):
        discover_inputs(_args(manifest=str(manifest)))


def test_partial_dataset_identity_is_explicitly_noncanonical(tmp_path):
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    manifest = tmp_path / "partial.jsonl"
    manifest.write_text(json.dumps({"image_path": "a.png", "facade_id": "f"}))
    sample = discover_inputs(_args(manifest=str(manifest)))[0]
    assert sample.provenance["dataset_metadata_available"] is False
    assert sample.provenance["canonical_dataset_v2"] is False


@pytest.mark.parametrize("name,pixel", [("stock_lposs.yaml", False),
                                         ("stock_lposs_plus.yaml", True)])
def test_dedicated_stock_configs_pin_upstream_dino(name, pixel):
    config = load_config(Path(__file__).resolve().parents[2] / "configs" / name)
    assert config["upstream_commit"] == "e489a7445528922ddfe4e39631ef2fe34827c873"
    assert config["dino"]["model"] == "dino_vitb16"
    assert config["dino"]["patch_size"] == 16
    assert config["dino"]["feature_type"] == "v"
    assert config["dino"]["repository"] == DINO_REPOSITORY
    assert config["pixel_refine"] is pixel


@pytest.mark.parametrize("rows,match", [
    ([{"sample_id": "../bad", "image_path": "images/a.png"}], "unsafe"),
    ([{"sample_id": "same", "image_path": "images/a.png"},
      {"sample_id": "same", "image_path": "images/b.png"}], "duplicate sample"),
])
def test_manifest_rejects_unsafe_or_duplicate_ids(tmp_path, rows, match):
    (tmp_path / "images").mkdir()
    for name in ("a.png", "b.png"):
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tmp_path / "images" / name)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match=match):
        discover_inputs(_args(manifest=str(manifest)))


def test_output_is_immutable_and_ornament_png_is_binary(tmp_path):
    sample = InputSample("sample", tmp_path / "input.png")
    output = prepare_output(tmp_path / "out", [sample])
    instance, _, _ = engine("maskclip_raw")
    result = instance.run(torch.rand(1, 3, 2, 3), prototype_set(), mode="maskclip_raw",
        ornament_negative_index=12, ornament_threshold=0.5)
    export_sample(output, sample, result, prototype_set(), False, False)
    ornament = np.asarray(Image.open(output / "sample" / "ornament_mask.png"))
    assert ornament.dtype == np.uint8
    assert set(np.unique(ornament)) <= {0, 1}
    with pytest.raises(FileExistsError):
        prepare_output(output, [sample])


def test_scores_are_saved_and_reloaded_with_112_compatible_loader(tmp_path, monkeypatch):
    from ovs_heritage.stock_features import compatible_torch_load
    sample = InputSample("sample", tmp_path / "input.png")
    output = prepare_output(tmp_path / "out", [sample])
    instance, _, _ = engine("maskclip_raw")
    result = instance.run(torch.rand(1, 3, 2, 3), prototype_set(), mode="maskclip_raw",
        ornament_negative_index=12, ornament_threshold=0.5)
    original = torch.load
    def old_load(path, **kwargs):
        if "weights_only" in kwargs:
            raise TypeError("load() got an unexpected keyword argument 'weights_only'")
        return original(path, **kwargs)
    monkeypatch.setattr(torch, "load", old_load)
    export_sample(output, sample, result, prototype_set(), True, False)
    payload = compatible_torch_load(output / "sample" / "scores.pt")
    assert torch.equal(payload["seed_scores"], result.seed_scores.cpu())


def test_graph_preflights_safe_and_reject_before_propagation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda _index: __import__("contextlib").nullcontext())
    device = DeviceInfo("cuda:0", "cpu", 0, "fake")
    safe = {"r": 13, "available_gpu_bytes": 10**9, "gpu_memory_reserve_bytes": 0,
            "max_graph_nodes": 100, "max_dense_graph_bytes": 10**9,
            "max_pixel_nodes": 100, "max_pixel_edges": 10000, "max_pixel_graph_bytes": 10**9}
    assert patch_graph_preflight(4, 2, 4, device, safe) > 0
    assert pixel_graph_preflight(2, 2, 2, 4, device, safe)[0] == 4 * 168
    instance, _, propagation = engine("lposs")
    instance.parameters["max_graph_nodes"] = 1
    with pytest.raises(RuntimeError, match="patch graph preflight rejected"):
        instance.run(torch.rand(1, 3, 5, 7), prototype_set(), mode="lposs")
    assert propagation.patch_calls == 0
    instance, _, propagation = engine("lposs_plus")
    instance.parameters["max_pixel_edges"] = 1
    with pytest.raises(RuntimeError, match="pixel graph preflight rejected"):
        instance.run(torch.rand(1, 3, 5, 7), prototype_set(), mode="lposs_plus")
    assert propagation.pixel_calls == 0


def test_official_adapter_command_is_isolated_and_explicit(tmp_path):
    from tools.check_stock_lposs_gpu import official_adapter_command
    command = official_adapter_command(python="python", adapter=tmp_path / "adapter.py",
        request=tmp_path / "request.json", output=tmp_path / "official")
    assert command == ["python", str(tmp_path / "adapter.py"), "--request",
                       str(tmp_path / "request.json"), "--output-dir", str(tmp_path / "official")]


def _official_artifact(tmp_path):
    from tools.check_stock_lposs_gpu import canonical_hash, file_hash
    arrays = {f"{mode}.{stage}": np.ones((1, 2, 3, 4), dtype=np.float32)
              for mode in ("maskclip_raw", "lposs", "lposs_plus")
              for stage in ("seed_scores", "propagated_scores")}
    arrays.update(clip_features=np.ones((4, 2), np.float32),
                  dino_features=np.ones((4, 2), np.float32))
    np.savez(tmp_path / "stages.npz", **arrays)
    configurations = {"maskclip_raw": {"pixel_refine": False},
                      "lposs": {"pixel_refine": False}, "lposs_plus": {"pixel_refine": True}}
    expected = {"input_sha256": "input", "prototype_artifact_sha256": "file-proto",
        "prototypes_sha256": "proto", "configurations": configurations,
        "configuration_sha256": canonical_hash(configurations),
        "channel_names": ["a", "b"], "semantic_ids": [1, None], "ontology_hash": "ontology",
        "model_hashes": {"clip_state_sha256": "clip", "dino_state_sha256": "dino"},
        "device": "cuda:0", "seed": 0, "upstream": {
            "repository": "https://github.com/vladan-stojnic/LPOSS",
            "commit": "e489a7445528922ddfe4e39631ef2fe34827c873", "tree": "tree"}}
    manifest = {**expected, "schema_version": "official-lposs-parity-artifact-v1",
        "producer": "official-upstream", "upstream": expected["upstream"],
        "dino_hub_loads": [{"requested_repository": "facebookresearch/dino:main",
            "resolved_repository": DINO_REPOSITORY, "model": "dino_vitb16",
            "args": [], "kwargs": {}}],
        "patch_grid": [2, 2], "image_grid": [3, 4],
        "stages": {"maskclip_raw": ["seed_scores"],
            "lposs": ["seed_scores", "propagated_scores"],
            "lposs_plus": ["seed_scores", "propagated_scores", "pixel_refinement"]},
        "stage_artifact": "stages.npz", "stage_artifact_sha256": file_hash(tmp_path / "stages.npz")}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, expected, manifest


def test_official_artifact_validation_and_rejections(tmp_path):
    from tools.check_stock_lposs_gpu import validate_official_artifact
    path, expected, manifest = _official_artifact(tmp_path)
    validated, arrays = validate_official_artifact(path, expected=expected)
    assert validated["upstream"]["commit"] == "e489a7445528922ddfe4e39631ef2fe34827c873"
    arrays.close()
    for key, value in (("input_sha256", "wrong"), ("prototypes_sha256", "wrong")):
        changed = {**manifest, key: value}
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="provenance mismatch"):
            validate_official_artifact(path, expected=expected)
    changed = {**manifest, "stages": {**manifest["stages"], "lposs_plus": ["seed_scores"]}}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="incomplete"):
        validate_official_artifact(path, expected=expected)
    changed = {**manifest, "upstream": {**manifest["upstream"], "commit": "0" * 40}}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="wrong repository"):
        validate_official_artifact(path, expected=expected)
    changed = {**manifest, "channel_names": ["b", "a"]}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="channel_names"):
        validate_official_artifact(path, expected=expected)


def test_correct_dino_commit_is_passed_to_torch_hub(monkeypatch):
    validate_pinned_hub_repository(DINO_REPOSITORY)
    with pytest.raises(ValueError, match="immutable"):
        validate_pinned_hub_repository("facebookresearch/dino:main")
    calls = []
    attention = types.SimpleNamespace(qkv=nn.Linear(2, 6), num_heads=1)
    encoder = nn.Module()
    encoder.blocks = [types.SimpleNamespace(attn=attention)]
    monkeypatch.setattr(torch.hub, "load", lambda repository, model, **kwargs:
                        calls.append((repository, model, kwargs)) or encoder)
    model = object.__new__(StockFeatureModel)
    nn.Module.__init__(model)
    model._dino_hook = {}
    StockFeatureModel.configure_dino(model, repository=DINO_REPOSITORY, model="dino_vitb16",
        patch_size=16, feature_type="v")
    assert calls == [(DINO_REPOSITORY, "dino_vitb16", {"source": "github"})]


def test_official_constructor_boundary_and_scoped_dino_redirect(monkeypatch):
    from tools.run_official_lposs_parity import construct_official_lposs
    constructed = []
    downloads = []

    class FakeOfficialLPOSS:
        def __init__(self, clip_backbone, class_names, vit_arch="vit_base",
                     vit_patch_size=16, enc_type_feats="k"):
            constructed.append((clip_backbone, class_names, vit_arch,
                                vit_patch_size, enc_type_feats))
            self.encoder = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")

    def original(repository, model, *args, **kwargs):
        downloads.append((repository, model, args, kwargs))
        return object()
    monkeypatch.setattr(torch.hub, "load", original)
    config = {"dino": {"architecture": "vit_base", "patch_size": 16, "feature_type": "v"}}
    model, intercepted = construct_official_lposs(
        FakeOfficialLPOSS, torch, class_names=["wall", "window"], config=config)
    assert isinstance(model, FakeOfficialLPOSS)
    assert constructed == [("maskclip", ["wall", "window"], "vit_base", 16, "v")]
    assert downloads == [(DINO_REPOSITORY, "dino_vitb16", (), {})]
    assert intercepted[0]["requested_repository"] == "facebookresearch/dino:main"
    assert intercepted[0]["resolved_repository"] == DINO_REPOSITORY
    assert torch.hub.load is original


@pytest.mark.parametrize("repository,model", [
    ("facebookresearch/dino:other", "dino_vitb16"),
    ("facebookresearch/dino:main", "dino_vits16"),
])
def test_official_dino_redirect_rejects_unexpected_calls_and_restores(monkeypatch, repository, model):
    from tools.run_official_lposs_parity import pinned_dino_hub_load
    def original(*_args, **_kwargs):
        return object()
    monkeypatch.setattr(torch.hub, "load", original)
    with pytest.raises(RuntimeError, match="unexpected official"):
        with pinned_dino_hub_load(torch, []):
            torch.hub.load(repository, model)
    assert torch.hub.load is original


def test_stock_config_rejects_changed_scientific_parameter(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = (root / "configs/stock_lposs.yaml").read_text().replace("alpha: 0.95", "alpha: 0.90")
    path = tmp_path / "changed.yaml"
    path.write_text(config)
    with pytest.raises(ValueError, match="scientifically meaningful graph"):
        load_config(path)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_threshold_validation(threshold):
    with pytest.raises(ValueError, match="finite"):
        validate_threshold(threshold)


def test_nonfinite_scores_fail_before_export():
    instance, model, _ = engine("maskclip_raw")
    model.clip_dense = lambda image: torch.full((1, 2, *image.shape[-2:]), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        instance.run(torch.rand(1, 3, 2, 2), prototype_set(), mode="maskclip_raw")


def test_ledger_success_and_failure_lifecycles(tmp_path):
    from research_ledger import Ledger, NewEvent, sanitize_error

    success = Ledger(tmp_path / "ledger", "success")
    success.append(NewEvent("run.started", {"implementation": "test"}))
    value, completed = _ledger_stage(success, "preflight", lambda: 7)
    assert value == 7
    assert completed.event_type == "stage.completed"
    success.append(NewEvent("run.completed", {"status": "completed"}))
    assert success.reconstruct().status == "completed"

    failure = Ledger(tmp_path / "ledger", "failure")
    failure.append(NewEvent("run.started", {"implementation": "test"}))
    error = RuntimeError("deliberate")
    with pytest.raises(RuntimeError, match="deliberate"):
        _ledger_stage(failure, "inference", lambda: (_ for _ in ()).throw(error))
    failure.append(NewEvent("run.failed", {"status": "failed", **sanitize_error(error)}))
    projection = failure.reconstruct()
    assert projection.status == "failed"
    assert projection.stages["inference"] == "failed"
