from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .types import ConditioningMechanism, TileSample


def _signed(v: int) -> float:
    return float(((int(v) + 128) % 256) - 128)


@dataclass
class ConditionedResidualModel:
    conditioning_mechanism: ConditioningMechanism = "concat_context"
    ridge_lambda: float = 1e-3
    sigma_floor: float = 1.0
    weights: List[float] | None = None
    sigma: float = 10.0

    def _feature_vector(self, prev_symbol: float, context: Sequence[float]) -> List[float]:
        ctx = [float(v) for v in context]
        if self.conditioning_mechanism == "concat_context":
            return [1.0, prev_symbol] + ctx
        if self.conditioning_mechanism == "film_context":
            return [1.0, prev_symbol] + ctx + [prev_symbol * c for c in ctx]
        raise ValueError(f"Unsupported conditioning_mechanism: {self.conditioning_mechanism}")

    def fit(
        self,
        samples: Sequence[TileSample],
        max_symbols_per_tile: int = 512,
        learning_rate: float = 1e-3,
        epochs: int = 30,
    ) -> Dict[str, float]:
        x_rows: List[List[float]] = []
        y_vals: List[float] = []
        for sample in samples:
            limit = min(len(sample.residual_bytes), max_symbols_per_tile)
            prev = 0.0
            for idx in range(limit):
                current = _signed(sample.residual_bytes[idx])
                x_rows.append(self._feature_vector(prev, sample.context_vector))
                y_vals.append(current)
                prev = current

        if not x_rows:
            self.weights = [0.0, 0.0]
            self.sigma = 10.0
            return {"n_symbols": 0, "sigma": self.sigma}

        dim = len(x_rows[0])
        w = [0.0] * dim
        n = float(len(x_rows))

        for _ in range(epochs):
            grad = [0.0] * dim
            for x, y in zip(x_rows, y_vals):
                pred = sum(wi * xi for wi, xi in zip(w, x))
                err = pred - y
                for i in range(dim):
                    grad[i] += (2.0 / n) * err * x[i]
            for i in range(dim):
                grad[i] += 2.0 * self.ridge_lambda * w[i]
                w[i] -= learning_rate * grad[i]

        residual_sq = 0.0
        for x, y in zip(x_rows, y_vals):
            pred = sum(wi * xi for wi, xi in zip(w, x))
            residual_sq += (y - pred) ** 2
        mse = residual_sq / max(len(y_vals), 1)
        self.weights = w
        self.sigma = max(self.sigma_floor, math.sqrt(max(mse, 1e-9)))
        return {"n_symbols": int(len(y_vals)), "sigma": self.sigma}

    def _predict_mean(self, prev_symbol: float, context: Sequence[float]) -> float:
        if self.weights is None:
            raise RuntimeError("Model not fitted")
        fv = self._feature_vector(prev_symbol, context)
        return sum(w * x for w, x in zip(self.weights, fv))

    def nll_bits_with_components(self, payload: bytes, context: Sequence[float]) -> Tuple[float, List[float]]:
        if self.weights is None:
            raise RuntimeError("Model not fitted")
        bits: List[float] = []
        total = 0.0
        prev = 0.0
        sigma = max(self.sigma, self.sigma_floor)
        const_bits = 0.5 * math.log2(2.0 * math.pi * sigma * sigma)
        inv = 1.0 / (2.0 * sigma * sigma * math.log(2.0))
        for byte_v in payload:
            target = _signed(byte_v)
            mean = self._predict_mean(prev, context)
            b = const_bits + ((target - mean) ** 2) * inv
            bits.append(b)
            total += b
            prev = target
        return total, bits

    def save(self, path: Path) -> None:
        if self.weights is None:
            raise RuntimeError("Cannot save an unfitted model")
        payload = {
            "model_type": "conditioned_residual_linear_gaussian",
            "conditioning_mechanism": self.conditioning_mechanism,
            "ridge_lambda": self.ridge_lambda,
            "sigma_floor": self.sigma_floor,
            "sigma": self.sigma,
            "weights": self.weights,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ConditionedResidualModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            conditioning_mechanism=payload["conditioning_mechanism"],
            ridge_lambda=float(payload.get("ridge_lambda", 1e-3)),
            sigma_floor=float(payload.get("sigma_floor", 1.0)),
            sigma=float(payload["sigma"]),
            weights=[float(v) for v in payload["weights"]],
        )
