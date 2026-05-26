from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


Context = Tuple[int, ...]


@dataclass
class ByteNGramEntropyModel:
    order: int = 1
    alpha: float = 1.0
    counts: Dict[Context, List[int]] | None = None

    def fit(self, payloads: Iterable[bytes]) -> None:
        context_counts: DefaultDict[Context, List[int]] = defaultdict(lambda: [0] * 256)
        for payload in payloads:
            seq: Sequence[int] = payload
            for idx, symbol in enumerate(seq):
                if self.order <= 0:
                    context: Context = ()
                else:
                    left = max(0, idx - self.order)
                    context = tuple(seq[left:idx])
                context_counts[context][symbol] += 1
        self.counts = dict(context_counts)

    def _distribution(self, context: Context) -> List[float]:
        if self.counts is None:
            raise RuntimeError("Model must be fitted before use")
        counts = self.counts.get(context)
        if counts is None:
            counts = self.counts.get((), [0] * 256)
        total = float(sum(counts) + self.alpha * 256)
        return [((count + self.alpha) / total) for count in counts]

    def nll_bits_with_components(self, payload: bytes) -> Tuple[float, List[float]]:
        seq: Sequence[int] = payload
        total_bits = 0.0
        per_symbol_bits: List[float] = []
        for idx, symbol in enumerate(seq):
            if self.order <= 0:
                context: Context = ()
            else:
                left = max(0, idx - self.order)
                context = tuple(seq[left:idx])
            prob = self._distribution(context)[symbol]
            bits = -math.log2(prob)
            per_symbol_bits.append(bits)
            total_bits += bits
        return total_bits, per_symbol_bits

    def nll_bits(self, payload: bytes) -> float:
        total_bits, _ = self.nll_bits_with_components(payload)
        return total_bits

    def save(self, path: Path) -> None:
        if self.counts is None:
            raise RuntimeError("Cannot save an unfitted model")
        serial_counts = {"|".join(str(v) for v in context): values for context, values in self.counts.items()}
        payload = {
            "model_type": "byte_ngram",
            "order": self.order,
            "alpha": self.alpha,
            "counts": serial_counts,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteNGramEntropyModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts: Dict[Context, List[int]] = {}
        for key, values in payload["counts"].items():
            context = tuple(int(v) for v in key.split("|") if v != "")
            counts[context] = [int(v) for v in values]
        return cls(order=int(payload["order"]), alpha=float(payload["alpha"]), counts=counts)


@dataclass
class ByteUnigramEntropyModel(ByteNGramEntropyModel):
    order: int = 0
