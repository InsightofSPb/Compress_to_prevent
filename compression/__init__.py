"""Facade compression and change-detection toolkit."""

from .pairs import build_facade_pairs
from .residuals import build_residual_dataset

__all__ = ["build_facade_pairs", "build_residual_dataset"]
