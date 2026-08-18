import pytest
import torch

from ovs_heritage.ontology import load_ontology
from ovs_heritage.projection import MAIN_SEMANTIC_IDS, MappingEntry, OntologyProjection


def test_exact_two_head_mapping_and_round_trip():
    projection = OntologyProjection.from_ontology(load_ontology())
    assert projection.main_channel_count == 11
    assert projection.for_semantic_id(8).output_head == "ornament"
    assert projection.for_semantic_id(8).channel_index == 0
    for semantic_id in MAIN_SEMANTIC_IDS:
        entry = projection.for_semantic_id(semantic_id)
        assert projection.for_channel("main", entry.channel_index).semantic_id == semantic_id
    semantic = torch.tensor([list(MAIN_SEMANTIC_IDS) + [255]])
    channels = projection.semantic_main_to_channels(semantic)
    assert channels[0, -1].item() == 255
    assert torch.equal(projection.main_channels_to_semantic(channels), semantic)


def test_projection_rejects_ornament_and_unknown_in_main():
    projection = OntologyProjection.from_ontology(load_ontology())
    with pytest.raises(ValueError, match="semantic ID 8"):
        projection.semantic_main_to_channels(torch.tensor([[[8]]]))
    with pytest.raises(ValueError, match="99"):
        projection.semantic_main_to_channels(torch.tensor([[[99]]]))


def test_duplicate_head_channel_is_ambiguous():
    entries = (
        MappingEntry(0, "a", "main", 0, "multiclass_softmax"),
        MappingEntry(1, "b", "main", 0, "multiclass_softmax"),
    )
    with pytest.raises(ValueError, match="duplicate channel"):
        OntologyProjection(entries)


def test_projection_rejects_ontology_name_drift_and_invalid_predictions():
    from dataclasses import replace

    ontology = load_ontology()
    changed_classes = tuple(
        replace(item, name="ornament_changed") if item.id == 8 else item
        for item in ontology.classes
    )
    with pytest.raises(ValueError, match="ornament_region"):
        OntologyProjection.from_ontology(replace(ontology, classes=changed_classes))
    projection = OntologyProjection.from_ontology(ontology)
    with pytest.raises(ValueError, match="non-finite"):
        projection.main_logits_to_semantic(torch.full((1, 11, 1, 1), float("nan")))
    with pytest.raises(ValueError, match="threshold"):
        projection.ornament_logits_to_binary(torch.zeros(1, 1, 1, 1), threshold=float("nan"))


def test_incomplete_and_noncontiguous_projections_fail_closed():
    canonical = OntologyProjection.from_ontology(load_ontology()).entries
    with pytest.raises(ValueError, match="main semantic IDs"):
        OntologyProjection(canonical[:-2] + canonical[-1:])
    changed = list(canonical)
    changed[2] = MappingEntry(2, "spalling", "main", 9, "multiclass_softmax")
    with pytest.raises(ValueError, match="duplicate channel|contiguous"):
        OntologyProjection(tuple(changed))


def test_unmapped_values_never_become_ignore():
    projection = OntologyProjection.from_ontology(load_ontology())
    with pytest.raises(ValueError, match="99"):
        projection.semantic_main_to_channels(torch.tensor([[[99]]]))
    with pytest.raises(ValueError, match="99"):
        projection.main_channels_to_semantic(torch.tensor([[[99]]]))
    assert projection.semantic_main_to_channels(torch.tensor([[[255]]])).item() == 255
    assert projection.main_channels_to_semantic(torch.tensor([[[255]]])).item() == 255
