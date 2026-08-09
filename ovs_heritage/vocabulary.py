"""Runtime logical vocabularies and prompt-ensemble prototype construction."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Iterable
import torch
import torch.nn.functional as F

from .ontology import Ontology

@dataclass(frozen=True)
class RuntimeClass:
    name: str
    prompts: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    id: int | None = None

@dataclass(frozen=True)
class PrototypeSet:
    prototypes: torch.Tensor
    channel_names: tuple[str, ...]
    vocabulary_hash: str

def heritage_runtime_vocabulary(ontology: Ontology, names: Iterable[str] | None = None) -> tuple[RuntimeClass, ...]:
    wanted = ontology.class_names if names is None else tuple(names)
    if len(wanted) != len(set(wanted)): raise ValueError("runtime vocabulary contains duplicate classes")
    return tuple(RuntimeClass(c.name, c.prompts, c.aliases, c.id) for name in wanted
                 for c in (ontology.by_name(name),))

def _validate(classes: tuple[RuntimeClass, ...]) -> None:
    names = [c.name for c in classes]
    if len(names) != len(set(names)): raise ValueError("runtime vocabulary contains duplicate class names")
    ids = [c.id for c in classes if c.id is not None]
    if len(ids) != len(set(ids)): raise ValueError("runtime vocabulary contains duplicate class IDs")
    aliases = [a.casefold() for c in classes for a in c.aliases]
    if len(aliases) != len(set(aliases)): raise ValueError("runtime vocabulary contains conflicting aliases")
    for c in classes:
        if not c.prompts: raise ValueError(f"runtime class {c.name!r} has no prompts")

def vocabulary_hash(classes: Iterable[RuntimeClass]) -> str:
    payload = [{"name": c.name, "id": c.id, "prompts": list(c.prompts), "aliases": list(c.aliases)} for c in classes]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def build_prototypes(classes: Iterable[RuntimeClass], text_encoder: Callable[[list[str]], torch.Tensor],
                     *, device=None, dtype=None, include_alias_prompts: bool = False,
                     eps: float = 1e-12) -> PrototypeSet:
    classes = tuple(classes); _validate(classes)
    prototypes = []
    for cls in classes:
        prompts = list(cls.prompts)
        if include_alias_prompts:
            prompts.extend(f"a {alias}" for alias in cls.aliases)
        encoded = text_encoder(prompts)
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2 or encoded.shape[0] != len(prompts):
            raise ValueError("text_encoder must return [number_of_prompts, embedding_dim]")
        if device is not None or dtype is not None: encoded = encoded.to(device=device, dtype=dtype)
        normalized = F.normalize(encoded, dim=-1, eps=eps)
        mean = normalized.mean(dim=0)
        if not torch.isfinite(mean).all() or mean.norm() <= eps:
            raise ValueError(f"prototype for {cls.name!r} is zero or non-finite")
        prototypes.append(F.normalize(mean, dim=0, eps=eps))
    if not prototypes: raise ValueError("runtime vocabulary is empty")
    return PrototypeSet(torch.stack(prototypes), tuple(c.name for c in classes), vocabulary_hash(classes))
