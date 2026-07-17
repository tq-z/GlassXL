"""Reusable deterministic distribution wrappers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Protocol


class Distribution(Protocol):
    def sample(self, rng: random.Random) -> float: ...


@dataclass(frozen=True)
class ConstantDistribution:
    value: float

    def sample(self, rng: random.Random) -> float:
        del rng
        return self.value


@dataclass(frozen=True)
class UniformDistribution:
    minimum: float
    maximum: float

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.minimum, self.maximum)


@dataclass(frozen=True)
class ExponentialDistribution:
    rate: float

    def sample(self, rng: random.Random) -> float:
        if self.rate <= 0.0:
            raise ValueError("exponential rate must be positive")
        return rng.expovariate(self.rate)


@dataclass(frozen=True)
class LognormalDistribution:
    mean: float
    sigma: float
    minimum: float | None = None
    maximum: float | None = None

    def sample(self, rng: random.Random) -> float:
        value = rng.lognormvariate(self.mean, self.sigma)
        if self.minimum is not None:
            value = max(value, self.minimum)
        if self.maximum is not None:
            value = min(value, self.maximum)
        return value


@dataclass(frozen=True)
class DiscreteDistribution:
    values: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values or len(self.values) != len(self.probabilities):
            raise ValueError("discrete values and probabilities must have equal non-zero length")
        if any(probability < 0.0 for probability in self.probabilities):
            raise ValueError("discrete probabilities cannot be negative")
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("discrete probabilities must sum to one")

    def sample(self, rng: random.Random) -> float:
        draw = rng.random()
        cumulative = 0.0
        for value, probability in zip(self.values, self.probabilities, strict=True):
            cumulative += probability
            if draw < cumulative:
                return value
        return self.values[-1]


@dataclass(frozen=True)
class BoundedIntegerDistribution:
    minimum: int
    maximum: int

    def sample(self, rng: random.Random) -> float:
        if self.minimum > self.maximum:
            raise ValueError("bounded integer minimum exceeds maximum")
        return float(rng.randint(self.minimum, self.maximum))


def distribution_from_config(config: dict[str, Any]) -> Distribution:
    """Build a distribution from a workload YAML mapping."""

    distribution_type = str(config.get("type", "")).lower()
    if distribution_type == "constant":
        return ConstantDistribution(float(config["value"]))
    if distribution_type == "uniform":
        return UniformDistribution(float(config["min"]), float(config["max"]))
    if distribution_type == "exponential":
        return ExponentialDistribution(float(config["rate"]))
    if distribution_type == "lognormal":
        return LognormalDistribution(
            mean=float(config["mean"]),
            sigma=float(config["sigma"]),
            minimum=float(config["min"]) if "min" in config else None,
            maximum=float(config["max"]) if "max" in config else None,
        )
    if distribution_type in {"discrete", "categorical"}:
        return DiscreteDistribution(
            tuple(float(value) for value in config["values"]),
            tuple(float(value) for value in config["probabilities"]),
        )
    if distribution_type in {"bounded_integer", "bounded_int"}:
        return BoundedIntegerDistribution(int(config["min"]), int(config["max"]))
    raise ValueError(f"unsupported distribution type: {distribution_type!r}")
