import pytest
import torch

from ovs_heritage.scoring import RawCosineScorer


def test_cpu_raw_scorer_dynamic_channels_and_no_state():
    scorer = RawCosineScorer(scale=2)
    features = torch.randn(2, 4, 3, 5)
    assert scorer(features, torch.randn(7, 4)).shape == (2, 7, 3, 5)
    assert scorer(features, torch.randn(2, 4)).shape == (2, 2, 3, 5)
    assert scorer.state_dict() == {}


def test_unbatched_per_class_parameters():
    out = RawCosineScorer()(
        torch.randn(4, 2, 3),
        torch.randn(3, 4),
        scale=torch.ones(3),
        bias=torch.arange(3.0),
    )
    assert out.shape == (3, 2, 3)


def test_dimension_and_shape_errors():
    with pytest.raises(ValueError, match="dimension mismatch"):
        RawCosineScorer()(torch.randn(1, 3, 2, 2), torch.randn(2, 4))
    with pytest.raises(ValueError, match="prototypes"):
        RawCosineScorer()(torch.randn(3, 2, 2), torch.randn(3))


def test_scorer_rejects_non_floating_nonfinite_zero_norm_and_bad_eps():
    with pytest.raises(ValueError, match="eps"):
        RawCosineScorer(eps=0)
    scorer = RawCosineScorer()
    with pytest.raises(ValueError, match="floating-point"):
        scorer(torch.ones(1, 3, 2, 2, dtype=torch.int64), torch.ones(2, 3))
    with pytest.raises(ValueError, match="finite"):
        scorer(torch.full((1, 3, 2, 2), float("nan")), torch.ones(2, 3))
    with pytest.raises(ValueError, match="non-zero"):
        scorer(torch.ones(1, 3, 2, 2), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="scale must be finite"):
        scorer(torch.ones(1, 3, 2, 2), torch.ones(2, 3), scale=float("inf"))
