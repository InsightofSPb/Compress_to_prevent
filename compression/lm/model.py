from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class ByteEntropyModel:
    alpha: float = 1.0
    counts: List[int] | None = None

    def fit(self, payloads: Iterable[bytes]) -> None:
        counts = [0] * 256
        for payload in payloads:
            for b in payload:
                counts[b] += 1
        self.counts = counts

    def _probs(self) -> List[float]:
        if self.counts is None:
            raise RuntimeError("Model must be fitted before use")
        total = float(sum(self.counts) + self.alpha * 256)
        return [((count + self.alpha) / total) for count in self.counts]

    def nll_bits(self, payload: bytes) -> float:
        probs = self._probs()
        return sum(-math.log2(probs[b]) for b in payload)

    def save(self, path: Path) -> None:
        if self.counts is None:
            raise RuntimeError("Cannot save an unfitted model")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"alpha": self.alpha, "counts": self.counts}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteEntropyModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(alpha=float(payload["alpha"]), counts=[int(v) for v in payload["counts"]])
