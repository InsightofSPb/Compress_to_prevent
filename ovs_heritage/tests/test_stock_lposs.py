import torch
import pytest

from ovs_heritage.projection import MAIN_SEMANTIC_IDS
from ovs_heritage.stock_lposs import StockOVSEngine, preflight
from ovs_heritage.vocabulary import PrototypeSet


class FakeModel:
    def __init__(self): self.clip_calls = self.dino_calls = 0
    def clip_dense(self, image):
        self.clip_calls += 1
        return torch.stack((image[:, 0], image[:, 1]), dim=1).squeeze(2)
    def dino_dense(self, image):
        self.dino_calls += 1
        return torch.ones(image.shape[0], 2, 2, 2, device=image.device)


class FakePropagation:
    def __init__(self): self.patch_calls = self.pixel_calls = 0
    def patch(self, dino, seeds, **kwargs):
        self.patch_calls += 1
        return seeds + 1, min(kwargs["parameters"]["k"], dino.shape[-1] * dino.shape[-2])
    def pixel(self, image, scores, **kwargs):
        self.pixel_calls += 1
        return scores + 2


def prototypes(extra=True):
    ids = list(MAIN_SEMANTIC_IDS) + [8, None]
    names = [f"class_{i}" for i in MAIN_SEMANTIC_IDS] + ["ornament_region", "negative"]
    if extra: ids.append(None); names.append("extra")
    vectors = torch.tensor([[1., 0.] if i % 2 else [0., 1.] for i in range(len(ids))])
    return PrototypeSet(vectors, tuple(names), tuple(ids), "vocab", "ontology", {})


@pytest.mark.parametrize("mode,expected", [("maskclip_raw", (0, 0)), ("lposs", (1, 0)), ("lposs_plus", (1, 1))])
def test_explicit_stage_execution_and_two_map_contract(mode, expected):
    model, propagation = FakeModel(), FakePropagation()
    engine = StockOVSEngine(model, propagation, device="cpu", cosine_scale=1,
                            graph_parameters={"k": 99}, enforce_preflight=False)
    out = engine.run(torch.rand(1, 3, 5, 7), prototypes(), mode=mode,
                     ornament_negative_index=12, ornament_threshold=.5)
    assert (model.dino_calls, propagation.pixel_calls) == expected
    assert propagation.patch_calls == expected[0]
    assert out.seed_scores.shape[-2:] == (5, 7)
    assert out.propagated_scores.shape[-2:] == (5, 7)
    assert out.main_scores.shape == (1, 11, 5, 7)
    assert out.main_mask.shape == (1, 5, 7)
    assert out.ornament_score.shape == (1, 1, 5, 7)
    assert set(out.extra_scores) == {"extra"}
    assert out.metadata.dino_executed is (mode != "maskclip_raw")
    assert out.metadata.patch_propagation_executed is (mode != "maskclip_raw")
    assert out.metadata.pixel_refinement_executed is (mode == "lposs_plus")


def test_runtime_order_subset_and_nonpersistent_prototypes():
    model, propagation = FakeModel(), FakePropagation()
    engine = StockOVSEngine(model, propagation, device="cpu", enforce_preflight=False)
    before = engine.scorer.state_dict()
    subset = PrototypeSet(torch.eye(2), ("novel", "crack"), (None, 1), "x", None, {})
    out = engine.run(torch.rand(1, 3, 3, 4), subset, mode="maskclip_raw")
    assert out.propagated_scores.shape == (1, 2, 3, 4)
    assert list(out.extra_scores) == ["novel"]
    assert out.main_mask is None
    assert engine.scorer.state_dict() == before == {}


def test_graph_mode_fails_closed_on_cpu_but_raw_does_not():
    preflight("maskclip_raw", "cpu")
    with pytest.raises(RuntimeError, match="genuine GPU graph propagation"):
        preflight("lposs", "cpu")


def test_raw_cosine_has_no_softmax():
    engine = StockOVSEngine(FakeModel(), None, device="cpu", cosine_scale=1, enforce_preflight=False)
    p = PrototypeSet(torch.eye(2), ("a", "b"), (None, None), "x", None, {})
    image = torch.zeros(1, 3, 2, 2); image[:, 0] = 1
    scores = engine.run(image, p, mode="maskclip_raw").seed_scores
    assert torch.allclose(scores[:, 0], torch.ones_like(scores[:, 0]))
    assert torch.allclose(scores[:, 1], torch.zeros_like(scores[:, 1]))
    # Cosines are returned directly (a softmax would make the zero channel non-zero).
    assert torch.count_nonzero(scores[:, 1]) == 0


def test_duplicated_conditional_expression_is_absent():
    source = open("segmentation/evaluation/lposs_eval.py", encoding="utf-8").read()
    assert "else i\n            (i.float()" not in source
    assert 'device="cuda"' not in source
