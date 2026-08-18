import torch
import torch.nn.functional as F
import pytest

from ovs_heritage.losses import combined_two_head_loss, main_segmentation_loss, ornament_region_loss
from ovs_heritage.projection import OntologyProjection


def test_main_loss_maps_semantic_ids_and_uses_raw_logits():
    logits = torch.randn(1, 11, 1, 2, requires_grad=True)
    semantic = torch.tensor([[[9, 255]]])
    channels = OntologyProjection.canonical_v2().semantic_main_to_channels(semantic)
    got = main_segmentation_loss(logits, semantic)
    assert torch.allclose(got, F.cross_entropy(logits, channels, ignore_index=255))


def test_main_all_ignore_is_differentiable_zero_and_id8_rejected():
    logits = torch.randn(1, 11, 2, 2, requires_grad=True)
    loss = main_segmentation_loss(logits, torch.full((1, 2, 2), 255))
    assert loss.item() == 0 and loss.requires_grad
    with pytest.raises(ValueError, match="semantic ID 8"):
        main_segmentation_loss(logits, torch.full((1, 2, 2), 8))


def test_ornament_ignore_mask_and_all_ignore():
    logits = torch.tensor([[[[0.0, 10.0, -2.0]]]], requires_grad=True)
    target = torch.tensor([[[[1, 255, 0]]]])
    got = ornament_region_loss(logits, target)
    expected = F.binary_cross_entropy_with_logits(logits[..., [0, 2]], torch.tensor([[[[1.0, 0.0]]]]))
    assert torch.allclose(got, expected)
    all_ignore = ornament_region_loss(logits, torch.full_like(target, 255))
    assert all_ignore.item() == 0 and all_ignore.requires_grad


def test_combined_loss_settings_and_overlap_targets():
    main = torch.randn(1, 11, 1, 2)
    ornament = torch.randn(1, 1, 1, 2)
    y_main = torch.tensor([[[7, 5]]])
    y_ornament = torch.tensor([[[1, 1]]])
    result = combined_two_head_loss(main, ornament, y_main, y_ornament, lambda_ornament=0.25)
    assert torch.allclose(result.total, result.main + 0.25 * result.ornament)
    assert result.metadata == {"lambda_ornament": 0.25, "pos_weight": None}
    with pytest.raises(ValueError, match="non-negative"):
        combined_two_head_loss(main, ornament, y_main, y_ornament, lambda_ornament=-1)


def test_probability_like_main_input_is_rejected():
    probabilities = torch.softmax(torch.randn(1, 11, 2, 2), dim=1)
    with pytest.raises(ValueError, match="normalized probabilities"):
        main_segmentation_loss(probabilities, torch.zeros(1, 2, 2, dtype=torch.long))
