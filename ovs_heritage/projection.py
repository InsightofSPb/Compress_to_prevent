"""Canonical semantic-ID projection for the v2 two-target representation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .ontology import Ontology, V2_CLASS_NAMES, V2_VERSION


IGNORE_INDEX = 255
MAIN_SEMANTIC_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11)


@dataclass(frozen=True)
class MappingEntry:
    semantic_id: int
    canonical_name: str
    output_head: str
    channel_index: int
    interpretation: str
    ignore_behavior: str = "255 is preserved and excluded from loss"
    unknown_behavior: str = "raise an error; never remap to ignore"


@dataclass(frozen=True)
class OntologyProjection:
    entries: tuple[MappingEntry, ...]
    ignore_index: int = IGNORE_INDEX

    @classmethod
    def from_ontology(cls, ontology: Ontology) -> "OntologyProjection":
        if ontology.version != V2_VERSION:
            raise ValueError(f"canonical two-head projection requires {V2_VERSION}")
        if ontology.by_name("ornament_region").id != 8:
            raise ValueError("semantic ID 8 must be exactly ornament_region")
        if tuple(sorted(ontology.valid_ids)) != tuple(range(12)):
            raise ValueError("v2 projection requires semantic IDs 0..11")
        main = tuple(
            MappingEntry(
                semantic_id,
                ontology.by_name(next(item.name for item in ontology.classes if item.id == semantic_id)).name,
                "main",
                channel,
                "multiclass_softmax",
            )
            for channel, semantic_id in enumerate(MAIN_SEMANTIC_IDS)
        )
        ornament = MappingEntry(8, ontology.by_name("ornament_region").name, "ornament", 0, "independent_sigmoid")
        projection = cls(main + (ornament,))
        if tuple(entry.channel_index for entry in projection.main_entries) != tuple(range(11)):
            raise ValueError("main channel indices must be contiguous 0..10")
        return projection

    def __post_init__(self) -> None:
        if type(self.ignore_index) is not int or self.ignore_index != IGNORE_INDEX:
            raise ValueError("ignore_index must be the exact integer 255")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("projection entries must be a non-empty tuple")
        known_heads = {"main": "multiclass_softmax", "ornament": "independent_sigmoid"}
        for index, entry in enumerate(self.entries):
            if not isinstance(entry, MappingEntry):
                raise ValueError(f"entries[{index}] must be a MappingEntry")
            if (
                type(entry.semantic_id) is not int
                or not 0 <= entry.semantic_id < len(V2_CLASS_NAMES)
            ):
                raise ValueError(
                    f"entries[{index}].semantic_id must be an integer in 0..{len(V2_CLASS_NAMES) - 1}"
                )
            if not isinstance(entry.canonical_name, str) or not entry.canonical_name.strip():
                raise ValueError(f"entries[{index}].canonical_name must be non-empty")
            if entry.canonical_name != V2_CLASS_NAMES[entry.semantic_id]:
                raise ValueError(
                    f"entries[{index}].canonical_name must be "
                    f"{V2_CLASS_NAMES[entry.semantic_id]!r} for semantic ID {entry.semantic_id}"
                )
            if entry.output_head not in known_heads:
                raise ValueError(f"entries[{index}] has unknown output head {entry.output_head!r}")
            if entry.interpretation != known_heads[entry.output_head]:
                raise ValueError(f"entries[{index}] has inconsistent output interpretation")
            if type(entry.channel_index) is not int or entry.channel_index < 0:
                raise ValueError(f"entries[{index}].channel_index must be a non-negative integer")
            if not isinstance(entry.ignore_behavior, str) or not entry.ignore_behavior.strip():
                raise ValueError(f"entries[{index}].ignore_behavior must be non-empty")
            if not isinstance(entry.unknown_behavior, str) or not entry.unknown_behavior.strip():
                raise ValueError(f"entries[{index}].unknown_behavior must be non-empty")
        semantic_ids = [entry.semantic_id for entry in self.entries]
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError("projection has duplicate semantic IDs")
        head_channels = [(entry.output_head, entry.channel_index) for entry in self.entries]
        if len(head_channels) != len(set(head_channels)):
            raise ValueError("projection has duplicate channel indices within an output head")
        main_entries = tuple(entry for entry in self.entries if entry.output_head == "main")
        ornament_entries = tuple(entry for entry in self.entries if entry.output_head == "ornament")
        if tuple(entry.semantic_id for entry in main_entries) != MAIN_SEMANTIC_IDS:
            raise ValueError(f"main semantic IDs must be exactly {list(MAIN_SEMANTIC_IDS)}")
        if tuple(entry.channel_index for entry in main_entries) != tuple(range(len(MAIN_SEMANTIC_IDS))):
            raise ValueError("main channel indices must be contiguous 0..10")
        if len(ornament_entries) != 1 or (
            ornament_entries[0].semantic_id,
            ornament_entries[0].channel_index,
        ) != (8, 0):
            raise ValueError("ornament projection must be semantic ID 8 at channel 0")

    @property
    def main_entries(self) -> tuple[MappingEntry, ...]:
        return tuple(entry for entry in self.entries if entry.output_head == "main")

    @property
    def main_channel_count(self) -> int:
        return len(self.main_entries)

    def for_semantic_id(self, semantic_id: int) -> MappingEntry:
        matches = [entry for entry in self.entries if entry.semantic_id == semantic_id]
        if len(matches) != 1:
            raise ValueError(f"semantic ID {semantic_id} has no unique canonical projection")
        return matches[0]

    def for_channel(self, head: str, channel_index: int) -> MappingEntry:
        matches = [
            entry for entry in self.entries
            if entry.output_head == head and entry.channel_index == channel_index
        ]
        if len(matches) != 1:
            raise ValueError(f"{head} channel {channel_index} has no unique canonical projection")
        return matches[0]

    def semantic_main_to_channels(self, target: torch.Tensor) -> torch.Tensor:
        self._validate_integer_target(target, "Y_main")
        found = set(torch.unique(target.detach()).cpu().tolist())
        semantic_to_channel = {entry.semantic_id: entry.channel_index for entry in self.main_entries}
        allowed = set(semantic_to_channel) | {self.ignore_index}
        invalid = sorted(found - allowed)
        if invalid:
            detail = "semantic ID 8 belongs to the ornament target" if 8 in invalid else "unknown IDs"
            raise ValueError(f"Y_main contains invalid semantic IDs {invalid}: {detail}")
        result = torch.full_like(target, self.ignore_index, dtype=torch.long)
        for semantic_id, channel_index in semantic_to_channel.items():
            result[target == semantic_id] = channel_index
        return result

    def main_channels_to_semantic(self, channels: torch.Tensor) -> torch.Tensor:
        self._validate_integer_target(channels, "main channel prediction")
        found = set(torch.unique(channels.detach()).cpu().tolist())
        channel_to_semantic = {entry.channel_index: entry.semantic_id for entry in self.main_entries}
        allowed = set(channel_to_semantic) | {self.ignore_index}
        invalid = sorted(found - allowed)
        if invalid:
            raise ValueError(f"main channel prediction contains unknown channel indices {invalid}")
        result = torch.full_like(channels, self.ignore_index, dtype=torch.long)
        for channel_index, semantic_id in channel_to_semantic.items():
            result[channels == channel_index] = semantic_id
        return result

    def main_logits_to_semantic(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != self.main_channel_count or not logits.is_floating_point():
            raise ValueError(f"main logits must be finite floating [N,{self.main_channel_count},H,W]")
        if not torch.isfinite(logits).all():
            raise ValueError("main logits contain non-finite values")
        return self.main_channels_to_semantic(logits.argmax(dim=1))

    def ornament_logits_to_binary(self, logits: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1 or not logits.is_floating_point():
            raise ValueError("ornament logits must be floating [N,1,H,W] raw logits")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("ornament threshold must be in 0..1")
        if not torch.isfinite(logits).all():
            raise ValueError("ornament logits contain non-finite values")
        return (torch.sigmoid(logits) >= threshold).to(torch.uint8)

    def as_dict(self) -> dict[str, object]:
        return {
            "ignore_index": self.ignore_index,
            "entries": [asdict(entry) for entry in self.entries],
        }

    @staticmethod
    def _validate_integer_target(target: torch.Tensor, label: str) -> None:
        integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        if target.dtype not in integer_dtypes:
            raise ValueError(f"{label} must have an integer dtype, got {target.dtype}")
        if target.ndim not in (2, 3, 4):
            raise ValueError(f"{label} must be a spatial target tensor, got shape {tuple(target.shape)}")
        if target.ndim == 4 and target.shape[1] != 1:
            raise ValueError(f"{label} four-dimensional targets must have exactly one channel")
