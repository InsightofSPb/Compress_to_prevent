import pytest
import torch
import torch.nn.functional as F

from ovs_heritage.losses import supervised_cross_entropy


def test_loss_is_raw_ce_and_ignore():
    logits = torch.tensor([[[[3.0, 1.0]], [[1.0, 3.0]], [[0.0, 0.0]]]])
    target = torch.tensor([[[0, 255]]])
    got = supervised_cross_entropy(logits, target)
    assert torch.allclose(got, F.cross_entropy(logits, target, ignore_index=255))
    assert not torch.allclose(got, F.cross_entropy(logits.softmax(1), target, ignore_index=255))


def test_float_and_boolean_targets_are_rejected_before_long_conversion():
    logits = torch.randn(1, 12, 1, 2)
    for target in (torch.tensor([[[11.0, 255.0]]]), torch.tensor([[[True, False]]])):
        with pytest.raises(ValueError, match=r"integer dtype.*found IDs"):
            supervised_cross_entropy(logits, target)


def test_id_11_valid_for_12_but_error_for_11():
    target = torch.tensor([[[11]]])
    assert torch.isfinite(supervised_cross_entropy(torch.randn(1, 12, 1, 1), target))
    with pytest.raises(ValueError, match=r"unknown target IDs \[11\]"):
        supervised_cross_entropy(torch.randn(1, 11, 1, 1), target)
    with pytest.raises(ValueError, match=r"unknown target IDs \[99\]"):
        supervised_cross_entropy(torch.randn(1, 12, 1, 1), torch.tensor([[[99]]]))
