"""Reusable deterministic distributions built from typed workload config."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from agentic_sim.config.models import (
    BoundedIntegerDistributionConfig,
    ConstantDistributionConfig,
    DiscreteDistributionConfig,
    DistributionConfig,
    ExponentialDistributionConfig,
    LognormalDistributionConfig,
    UniformDistributionConfig,
)


class Distribution(Protocol):
    def sample(self, rng: random.Random) -> float:
        """Draw one deterministic value from the supplied RNG."""


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
        if any(not math.isfinite(probability) or probability < 0.0 for probability in self.probabilities):
            raise ValueError("discrete probabilities must be finite and non-negative")
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

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("bounded integer minimum exceeds maximum")

    def sample(self, rng: random.Random) -> float:
        return float(rng.randint(self.minimum, self.maximum))


def distribution_from_config(config: DistributionConfig) -> Distribution:
    """Build a distribution exclusively from the strict typed V1 schema."""

    if isinstance(config, ConstantDistributionConfig):
        return ConstantDistribution(float(config.value))
    if isinstance(config, UniformDistributionConfig):
        return UniformDistribution(config.minimum, config.maximum)
    if isinstance(config, ExponentialDistributionConfig):
        return ExponentialDistribution(config.rate)
    if isinstance(config, LognormalDistributionConfig):
        return LognormalDistribution(
            mean=config.mean,
            sigma=config.sigma,
            minimum=config.minimum,
            maximum=config.maximum,
        )
    if isinstance(config, DiscreteDistributionConfig):
        return DiscreteDistribution(
            tuple(float(value) for value in config.values),
            config.probabilities,
        )
    if isinstance(config, BoundedIntegerDistributionConfig):
        return BoundedIntegerDistribution(config.minimum, config.maximum)
    raise TypeError(f"unsupported typed distribution config: {type(config).__name__}")
