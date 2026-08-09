import pytest
import torch

from ovs_heritage.ontology import load_ontology
from ovs_heritage.scoring import RawCosineScorer
from ovs_heritage.vocabulary import RuntimeClass, build_prototypes, heritage_runtime_vocabulary


def encoder(prompts):
    return torch.tensor(
        [[len(prompt), sum(map(ord, prompt)) % 19 + 1, 1.0] for prompt in prompts],
        dtype=torch.float32,
    )


def test_prompt_ensemble_and_aliases_make_one_channel_per_class():
    vocabulary = (
        RuntimeClass("mixed", ("first", "second"), ("alias one", "alias two")),
        RuntimeClass("new", ("third",)),
    )
    result = build_prototypes(vocabulary, encoder, include_alias_prompts=True)
    assert result.prototypes.dtype == torch.float32
    assert result.prototypes.shape == (2, 3)
    assert result.channel_names == ("mixed", "new")
    assert torch.allclose(result.prototypes.norm(dim=1), torch.ones(2))


def test_heritage_mixed_unseen_and_arbitrary_order():
    ontology = load_ontology()
    mixed = heritage_runtime_vocabulary(ontology, ["advertisements", "crack"]) + (
        RuntimeClass("unseen", ("an unseen thing",)),
    )
    assert build_prototypes(mixed, encoder).channel_names == ("advertisements", "crack", "unseen")
    assert build_prototypes((RuntimeClass("only_new", ("new",)),), encoder).prototypes.shape == (1, 3)


def test_prototype_and_scorer_cpu_smoke_has_no_persistent_cache():
    prototypes = build_prototypes(
        (RuntimeClass("one", ("first",)), RuntimeClass("two", ("second", "another"))),
        encoder,
    )
    scorer = RawCosineScorer(scale=10.0)
    logits = scorer(torch.randn(1, 3, 4, 5), prototypes.prototypes)
    assert logits.shape == (1, 2, 4, 5)
    assert scorer.state_dict() == {}


def test_runtime_validation_remains_independent_of_heritage_invariants():
    with pytest.raises(ValueError, match="duplicate"):
        build_prototypes((RuntimeClass("x", ("a",)), RuntimeClass("x", ("b",))), encoder)
    with pytest.raises(ValueError, match="no prompts"):
        build_prototypes((RuntimeClass("x", ()),), encoder)
