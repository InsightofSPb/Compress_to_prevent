from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class BackendCapabilities:
    backend: str
    model_id: str
    provides_dense_features: bool
    provides_global_features: bool
    provides_masks: bool
    provides_logits_or_probs: bool
    tile_compatible: bool
    expected_feature_grid_type: str
    experimental: bool
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class SemanticBackend(ABC):
    name: str = "base"

    @classmethod
    @abstractmethod
    def capabilities(cls) -> BackendCapabilities:
        raise NotImplementedError

    @abstractmethod
    def export_artifacts(self, sample: Dict[str, str], out_dir: Path, tile_size: int) -> Dict[str, object]:
        raise NotImplementedError


class BackendRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, type[SemanticBackend]] = {}

    def register(self, backend_cls: type[SemanticBackend]) -> None:
        self._registry[backend_cls.name] = backend_cls

    def create(self, name: str) -> SemanticBackend:
        if name not in self._registry:
            raise KeyError(f"Unknown backend: {name}")
        return self._registry[name]()

    def names(self) -> List[str]:
        return sorted(self._registry.keys())

    def capabilities(self, name: str) -> BackendCapabilities:
        if name not in self._registry:
            raise KeyError(f"Unknown backend: {name}")
        return self._registry[name].capabilities()

    def capability_table(self) -> List[Dict[str, object]]:
        return [self.capabilities(name).to_dict() for name in self.names()]
