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
    discover_inputs,
    export_sample,
    load_config,
    prepare_output,
    validate_threshold,
    _ledger_stage,
)
from ovs_heritage.projection import MAIN_SEMANTIC_IDS
from ovs_heritage.stock_features import StockFeatureModel, module_state_fingerprint
from ovs_heritage.stock_lposs import (
    DeviceInfo,
    StockOVSEngine,
    graph_preflight,
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
        graph_parameters={"k": 99})
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
    assert result == [InputSample("sample-a", (image_dir / "a.png").resolve())]


@pytest.mark.parametrize("name,pixel", [("stock_lposs.yaml", False),
                                         ("stock_lposs_plus.yaml", True)])
def test_dedicated_stock_configs_pin_upstream_dino(name, pixel):
    config = load_config(Path(__file__).resolve().parents[2] / "configs" / name)
    assert config["upstream_commit"] == "e489a7445528922ddfe4e39631ef2fe34827c873"
    assert config["dino"]["model"] == "dino_vitb16"
    assert config["dino"]["patch_size"] == 16
    assert config["dino"]["feature_type"] == "v"
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
