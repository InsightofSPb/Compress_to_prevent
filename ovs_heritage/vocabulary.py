"""Runtime logical vocabularies and one-prototype-per-class construction."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F

from .ontology import Ontology


def _freeze_settings(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_settings(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_settings(item) for item in value)
    return value


@dataclass(frozen=True)
class RuntimeClass:
    name: str
    prompts: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    semantic_id: int | None = None


@dataclass(frozen=True)
class PrototypeSet:
    prototypes: torch.Tensor
    channel_names: tuple[str, ...]
    semantic_ids: tuple[int | None, ...]
    vocabulary_specification_hash: str
    ontology_hash: str | None
    prompt_settings: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_settings", _freeze_settings(dict(self.prompt_settings)))

    @property
    def vocabulary_hash(self) -> str:
        """Backward-compatible name for the specification hash."""
        return self.vocabulary_specification_hash


def heritage_runtime_vocabulary(
    ontology: Ontology, names: Iterable[str] | None = None,
) -> tuple[RuntimeClass, ...]:
    wanted = ontology.class_names if names is None else tuple(names)
    if len(wanted) != len(set(wanted)):
        raise ValueError("runtime vocabulary contains duplicate classes")
    return tuple(
        RuntimeClass(item.name, item.prompts, item.aliases, item.id)
        for name in wanted
        for item in (ontology.by_name(name),)
    )


def _validate(classes: tuple[RuntimeClass, ...]) -> None:
    names = [item.name for item in classes]
    if len(names) != len(set(names)):
        raise ValueError("runtime vocabulary contains duplicate class names")
    ids = [item.semantic_id for item in classes if item.semantic_id is not None]
    if len(ids) != len(set(ids)):
        raise ValueError("runtime vocabulary contains duplicate semantic IDs")
    aliases = [alias.casefold() for item in classes for alias in item.aliases]
    if len(aliases) != len(set(aliases)):
        raise ValueError("runtime vocabulary contains conflicting aliases")
    for item in classes:
        if not item.prompts:
            raise ValueError(f"runtime class {item.name!r} has no prompts")


def vocabulary_specification_hash(
    classes: Iterable[RuntimeClass], *, include_alias_prompts: bool,
    prompt_method: str = "normalize_mean_normalize",
) -> str:
    payload = {
        "classes": [
            {
                "name": item.name,
                "semantic_id": item.semantic_id,
                "prompts": list(item.prompts),
                "aliases": list(item.aliases),
            }
            for item in classes
        ],
        "prompt_settings": {
            "include_alias_prompts": include_alias_prompts,
            "method": prompt_method,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def build_prototypes(
    classes: Iterable[RuntimeClass], text_encoder: Callable[[list[str]], torch.Tensor],
    *, device=None, dtype=None, include_alias_prompts: bool = False,
    ontology_hash: str | None = None, eps: float = 1e-12,
) -> PrototypeSet:
    classes = tuple(classes)
    _validate(classes)
    prototypes = []
    for item in classes:
        prompts = list(item.prompts)
        if include_alias_prompts:
            prompts.extend(f"a {alias}" for alias in item.aliases)
        encoded = text_encoder(prompts)
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2:
            raise ValueError("text_encoder must return [number_of_prompts, embedding_dim]")
        if encoded.shape[0] != len(prompts) or not encoded.is_floating_point():
            raise ValueError("text_encoder must return floating embeddings for every prompt")
        if device is not None or dtype is not None:
            encoded = encoded.to(device=device, dtype=dtype)
        normalized = F.normalize(encoded, dim=-1, eps=eps)
        mean = normalized.mean(dim=0)
        if not torch.isfinite(mean).all() or mean.norm() <= eps:
            raise ValueError(f"prototype for {item.name!r} is zero or non-finite")
        prototypes.append(F.normalize(mean, dim=0, eps=eps))
    if not prototypes:
        raise ValueError("runtime vocabulary is empty")
    settings = {"include_alias_prompts": include_alias_prompts, "method": "normalize_mean_normalize"}
    return PrototypeSet(
        torch.stack(prototypes),
        tuple(item.name for item in classes),
        tuple(item.semantic_id for item in classes),
        vocabulary_specification_hash(classes, include_alias_prompts=include_alias_prompts),
        ontology_hash,
        settings,
    )
