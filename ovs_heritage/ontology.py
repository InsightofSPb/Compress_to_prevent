"""Typed, strictly validated ontology loaded from the single YAML source."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from yaml import YAMLError


IGNORE_INDEX = 255
DEFAULT_ONTOLOGY = Path(__file__).parent / "configs" / "heritage_vocab.yaml"
V1_VERSION = "heritage_facades_v1_11classes"
V2_VERSION = "heritage_facades_v2_12classes"
V2_CLASS_NAMES = (
    "background", "crack", "spalling", "delamination", "missing_element",
    "water_stain", "efflorescence", "corrosion", "ornament_intact",
    "repairs", "text_or_images", "advertisements",
)
V1_CLASS_NAMES = V2_CLASS_NAMES[:-1]
REQUIRED_V2_GROUPS = {
    "STRUCTURAL_DAMAGE": V2_CLASS_NAMES[1:5],
    "SURFACE_STAIN": V2_CLASS_NAMES[5:8],
    "HUMAN_ACTIVITY": ("repairs", "text_or_images", "advertisements"),
    "DAMAGE_MACRO": V2_CLASS_NAMES[1:8],
}


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
    version = data.get("version", "")
    if not isinstance(version, str): raise OntologyError("ontology version must be a string")
    ignore = data.get("ignore_index", IGNORE_INDEX)
    if type(ignore) is not int:
        raise OntologyError(f"ignore_index must be an integer, got {ignore!r} ({type(ignore).__name__})")
    raw_classes = data.get("classes")
    if not isinstance(raw_classes, Sequence) or isinstance(raw_classes, (str, bytes)) or not raw_classes:
        raise OntologyError("classes must be a non-empty list")
    classes = []
    for raw in raw_classes:
        if not isinstance(raw, Mapping): raise OntologyError(f"class entry must be a mapping: {raw!r}")
        raw_id = raw.get("id")
        if type(raw_id) is not int:
            raise OntologyError(f"class ID must be an integer, got {raw_id!r} ({type(raw_id).__name__})")
        try:
            color = raw["color"]
            if not isinstance(color, Sequence) or isinstance(color, (str, bytes)) or any(type(x) is not int for x in color):
                raise OntologyError(f"color for {raw.get('name')!r} must contain three integers")
            cls = OntologyClass(raw_id, str(raw["name"]), str(raw["display_name"]),
                str(raw["description"]), tuple(raw["prompts"]), tuple(raw.get("aliases", [])),
                str(raw["role"]), bool(raw["is_heritage"]),
                tuple(raw.get("evaluation_groups", [])), tuple(color))
        except OntologyError:
            raise
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
    expected_names = V2_CLASS_NAMES if version == V2_VERSION else V1_CLASS_NAMES if version == V1_VERSION else None
    if expected_names is not None:
        if ignore != IGNORE_INDEX: raise OntologyError(f"{version} requires ignore_index=255, got {ignore}")
        if ids != list(range(len(expected_names))):
            raise OntologyError(f"{version} requires ordered IDs 0..{len(expected_names) - 1}, got {ids}")
        if tuple(names) != expected_names:
            raise OntologyError(f"{version} requires canonical class order {list(expected_names)}, got {names}")
        if len(colors) != len(expected_names):
            raise OntologyError(f"{version} requires exactly {len(expected_names)} palette entries")
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
    class_groups = {c.name: set(c.evaluation_groups) for c in classes}
    for group, members in groups.items():
        for member in members:
            if group not in class_groups[member]:
                raise OntologyError(
                    f"inconsistent membership: top-level group {group} contains {member}, "
                    f"but {member}.evaluation_groups omits {group}"
                )
    if version == V2_VERSION:
        for group, required_members in REQUIRED_V2_GROUPS.items():
            actual = groups.get(group)
            if actual is None:
                raise OntologyError(f"{version} requires evaluation group {group}")
            if tuple(actual) != tuple(required_members):
                raise OntologyError(f"{group} must be {list(required_members)}, got {list(actual)}")
    return Ontology(version, ignore, tuple(classes), groups, _canonical_hash(data))


def load_ontology(path: str | Path = DEFAULT_ONTOLOGY) -> Ontology:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except YAMLError as exc:
        raise OntologyError(f"{path}: malformed YAML: {exc}") from exc
    return ontology_from_mapping(data)


def validate_mask_ids(values: Any, ontology: Ontology, source: str = "mask") -> set[int]:
    found = extract_mask_ids(values, source)
    unknown = found - ontology.valid_ids - {ontology.ignore_index}
    if unknown:
        raise OntologyError(f"{source}: unknown mask IDs {sorted(unknown)}; allowed IDs are "
                            f"{sorted(ontology.valid_ids)} plus ignore {ontology.ignore_index}")
    return found


def extract_mask_ids(values: Any, source: str = "mask") -> set[int]:
    """Extract IDs only after proving that a mask has a non-boolean integer dtype."""
    import numpy as np
    array = np.asarray(values)
    found_values = _display_unique_values(array)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise OntologyError(
            f"{source}: mask dtype must be an integer dtype, got {array.dtype}; "
            f"found IDs {found_values}"
        )
    return set(np.unique(array).tolist())


def _display_unique_values(array: Any) -> list[Any]:
    """Return JSON/error-friendly unique values without coercing their type."""
    import numpy as np
    try:
        return np.unique(array).tolist()
    except (TypeError, ValueError):
        return list(dict.fromkeys(repr(value) for value in np.asarray(array).flat))
