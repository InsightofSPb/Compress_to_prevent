from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict


class SemanticBackend(ABC):
    name: str = "base"

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

    def names(self) -> list[str]:
        return sorted(self._registry.keys())
