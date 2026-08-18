import pytest
import torch

from ovs_heritage.projection import MAIN_SEMANTIC_IDS, MappingEntry, OntologyProjection


def test_exact_two_head_mapping_and_round_trip():
    projection = OntologyProjection.canonical_v2()
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
    projection = OntologyProjection.canonical_v2()
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
