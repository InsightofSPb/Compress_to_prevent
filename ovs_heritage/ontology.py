"""Typed, strictly validated ontology loaded from the single YAML source."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


IGNORE_INDEX = 255
DEFAULT_ONTOLOGY = Path(__file__).parent / "configs" / "heritage_vocab.yaml"


class OntologyError(ValueError):
    pass


@dataclass(frozen=True)
class OntologyClass:
    id: int
    name: str
    display_name: str
    description: str
    prompts: tuple[str, ...]
    aliases: tuple[str, ...]
    role: str
    is_heritage: bool
    evaluation_groups: tuple[str, ...]
    color: tuple[int, int, int]


@dataclass(frozen=True)
class Ontology:
    version: str
    ignore_index: int
    classes: tuple[OntologyClass, ...]
    groups: Mapping[str, tuple[str, ...]]
    hash: str

    @property
    def class_names(self) -> tuple[str, ...]: return tuple(c.name for c in self.classes)
    @property
    def display_names(self) -> tuple[str, ...]: return tuple(c.display_name for c in self.classes)
    @property
    def palette(self) -> tuple[tuple[int, int, int], ...]: return tuple(c.color for c in self.classes)
    @property
    def valid_ids(self) -> frozenset[int]: return frozenset(c.id for c in self.classes)
    def by_name(self, name: str) -> OntologyClass:
        return next(c for c in self.classes if c.name == name)


def _canonical_hash(data: Mapping[str, Any]) -> str:
    normalized = json.dumps(data, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"))
    return sha256(normalized.encode("utf-8")).hexdigest()


def ontology_from_mapping(data: Mapping[str, Any]) -> Ontology:
    if not isinstance(data, Mapping): raise OntologyError("ontology root must be a mapping")
    version = str(data.get("version", ""))
    ignore = int(data.get("ignore_index", IGNORE_INDEX))
    raw_classes = data.get("classes")
    if not isinstance(raw_classes, Sequence) or isinstance(raw_classes, (str, bytes)) or not raw_classes:
        raise OntologyError("classes must be a non-empty list")
    classes = []
    for raw in raw_classes:
        try:
            cls = OntologyClass(int(raw["id"]), str(raw["name"]), str(raw["display_name"]),
                str(raw["description"]), tuple(raw["prompts"]), tuple(raw.get("aliases", [])),
                str(raw["role"]), bool(raw["is_heritage"]),
                tuple(raw.get("evaluation_groups", [])), tuple(int(x) for x in raw["color"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise OntologyError(f"invalid class entry: {raw!r}: {exc}") from exc
        if not cls.prompts or any(not str(p).strip() for p in cls.prompts):
            raise OntologyError(f"class {cls.name!r} has no usable prompts")
        if len(cls.color) != 3 or any(x < 0 or x > 255 for x in cls.color):
            raise OntologyError(f"invalid color for {cls.name!r}")
        classes.append(cls)
    ids, names = [c.id for c in classes], [c.name for c in classes]
    if len(ids) != len(set(ids)): raise OntologyError("duplicate numeric class IDs")
    if len(names) != len(set(names)): raise OntologyError("duplicate canonical class names")
    if ignore in ids: raise OntologyError(f"ignore_index {ignore} must not be a class")
    aliases = [a.casefold() for c in classes for a in c.aliases]
    reserved = {n.casefold() for n in names}
    if len(aliases) != len(set(aliases)) or reserved.intersection(aliases):
        raise OntologyError("duplicate/conflicting aliases")
    colors = [c.color for c in classes]
    if len(colors) != len(set(colors)): raise OntologyError("palette colors must be unique")
    if version == "heritage_facades_v2_12classes":
        if ids != list(range(12)): raise OntologyError("v2 heritage IDs must be ordered and continuous 0..11")
        if names[0] != "background" or ids[0] != 0: raise OntologyError("BACKGROUND must have ID 0")
    groups_raw = data.get("groups", {})
    groups = {str(k): tuple(str(x) for x in v) for k, v in groups_raw.items()}
    known = set(names)
    for group, members in groups.items():
        unknown = set(members) - known
        if unknown: raise OntologyError(f"group {group} references unknown classes: {sorted(unknown)}")
    for c in classes:
        unknown_groups = set(c.evaluation_groups) - set(groups)
        if unknown_groups: raise OntologyError(f"class {c.name} references unknown groups: {sorted(unknown_groups)}")
        for group in c.evaluation_groups:
            if c.name not in groups[group]: raise OntologyError(f"inconsistent membership for {c.name} in {group}")
    return Ontology(version, ignore, tuple(classes), groups, _canonical_hash(data))


def load_ontology(path: str | Path = DEFAULT_ONTOLOGY) -> Ontology:
    with Path(path).open(encoding="utf-8") as stream:
        # JSON is a strict subset of YAML.  Keeping the source in this canonical
        # subset avoids imposing a YAML dependency on CPU validation jobs.
        data = json.load(stream)
    return ontology_from_mapping(data)


def validate_mask_ids(values: Any, ontology: Ontology, source: str = "mask") -> set[int]:
    import numpy as np
    found = {int(x) for x in np.unique(np.asarray(values))}
    unknown = found - ontology.valid_ids - {ontology.ignore_index}
    if unknown:
        raise OntologyError(f"{source}: unknown mask IDs {sorted(unknown)}; allowed IDs are "
                            f"{sorted(ontology.valid_ids)} plus ignore {ontology.ignore_index}")
    return found
