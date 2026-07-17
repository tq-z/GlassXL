"""Arrival process models."""

from __future__ import annotations

import random
from collections.abc import Iterator


def poisson_arrival_times(
    num_arrivals: int,
    rate_per_second: float,
    rng: random.Random,
) -> Iterator[float]:
    """Yield cumulative Poisson-process arrival timestamps."""

    if num_arrivals < 0:
        raise ValueError("num_arrivals must be non-negative")
    if num_arrivals and rate_per_second <= 0.0:
        raise ValueError("Poisson rate must be positive")
    current_time = 0.0
    for _ in range(num_arrivals):
        current_time += rng.expovariate(rate_per_second)
        yield current_time
