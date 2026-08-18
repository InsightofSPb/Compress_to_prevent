import torch

from ovs_heritage.ontology import load_ontology
from ovs_heritage.scoring import RawCosineScorer
from ovs_heritage.vocabulary import RuntimeClass, build_prototypes, heritage_runtime_vocabulary


def encoder(prompts):
    return torch.tensor([[len(prompt), sum(map(ord, prompt)) % 19 + 1, 1.0] for prompt in prompts], dtype=torch.float32)


def test_runtime_orders_subset_extended_mixed_and_unseen():
    ontology = load_ontology()
    mixed = heritage_runtime_vocabulary(ontology, ["advertisements", "crack"]) + (
        RuntimeClass("unseen", ("an unseen thing",), semantic_id=None),
    )
    result = build_prototypes(mixed, encoder, ontology_hash=ontology.hash)
    assert result.channel_names == ("advertisements", "crack", "unseen")
    assert result.semantic_ids == (11, 1, None)
    assert result.prototypes.shape == (3, 3)
    assert result.ontology_hash == ontology.hash
    assert RawCosineScorer()(torch.randn(1, 3, 2, 2), result.prototypes).shape == (1, 3, 2, 2)


def test_prompt_settings_change_specification_hash_without_persistent_state():
    classes = (RuntimeClass("one", ("first",), ("alias",), None),)
    plain = build_prototypes(classes, encoder, include_alias_prompts=False)
    aliases = build_prototypes(classes, encoder, include_alias_prompts=True)
    assert plain.vocabulary_specification_hash != aliases.vocabulary_specification_hash
    assert plain.prompt_settings["include_alias_prompts"] is False
    assert RawCosineScorer().state_dict() == {}
