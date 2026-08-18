"""Canonical semantic-ID projection for the v2 two-target representation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

IGNORE_INDEX = 255
MAIN_SEMANTIC_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11)
MAIN_NAMES = (
    "background", "crack", "spalling", "delamination", "missing_element",
    "water_stain", "efflorescence", "corrosion", "repairs",
    "text_or_images", "advertisements",
)


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
    def canonical_v2(cls) -> "OntologyProjection":
        main = tuple(
            MappingEntry(semantic_id, name, "main", channel, "multiclass_softmax")
            for channel, (semantic_id, name) in enumerate(zip(MAIN_SEMANTIC_IDS, MAIN_NAMES))
        )
        ornament = MappingEntry(8, "ornament_region", "ornament", 0, "independent_sigmoid")
        return cls(main + (ornament,))

    def __post_init__(self) -> None:
        semantic_ids = [entry.semantic_id for entry in self.entries]
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError("projection has duplicate semantic IDs")
        head_channels = [(entry.output_head, entry.channel_index) for entry in self.entries]
        if len(head_channels) != len(set(head_channels)):
            raise ValueError("projection has duplicate channel indices within an output head")

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
        allowed = set(MAIN_SEMANTIC_IDS) | {self.ignore_index}
        invalid = sorted(found - allowed)
        if invalid:
            detail = "semantic ID 8 belongs to the ornament target" if 8 in invalid else "unknown IDs"
            raise ValueError(f"Y_main contains invalid semantic IDs {invalid}: {detail}")
        result = torch.full_like(target, self.ignore_index, dtype=torch.long)
        for entry in self.main_entries:
            result[target == entry.semantic_id] = entry.channel_index
        return result

    def main_channels_to_semantic(self, channels: torch.Tensor) -> torch.Tensor:
        self._validate_integer_target(channels, "main channel prediction")
        found = set(torch.unique(channels.detach()).cpu().tolist())
        allowed = set(range(self.main_channel_count)) | {self.ignore_index}
        invalid = sorted(found - allowed)
        if invalid:
            raise ValueError(f"main channel prediction contains unknown channel indices {invalid}")
        result = torch.full_like(channels, self.ignore_index, dtype=torch.long)
        for entry in self.main_entries:
            result[channels == entry.channel_index] = entry.semantic_id
        return result

    def main_logits_to_semantic(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != self.main_channel_count:
            raise ValueError(f"main logits must be [N,{self.main_channel_count},H,W]")
        return self.main_channels_to_semantic(logits.argmax(dim=1))

    def ornament_logits_to_binary(self, logits: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1 or not logits.is_floating_point():
            raise ValueError("ornament logits must be floating [N,1,H,W] raw logits")
        if not 0 <= threshold <= 1:
            raise ValueError("ornament threshold must be in 0..1")
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


def mapping_semantic_ids(entries: Iterable[MappingEntry]) -> tuple[int, ...]:
    return tuple(entry.semantic_id for entry in entries)
