import random

import pytest

from agentic_sim.workload.arrival import poisson_arrival_times
from agentic_sim.workload.distributions import distribution_from_config


def test_poisson_arrivals_are_monotonic_and_reproducible() -> None:
    first = list(poisson_arrival_times(5, 2.0, random.Random(4)))
    second = list(poisson_arrival_times(5, 2.0, random.Random(4)))
    assert first == second
    assert first == sorted(first)
    assert first[0] > 0.0


def test_supported_distributions_are_bounded() -> None:
    rng = random.Random(3)
    lognormal = distribution_from_config(
        {"type": "lognormal", "mean": 2.0, "sigma": 4.0, "min": 5, "max": 8}
    )
    bounded = distribution_from_config({"type": "bounded_integer", "min": 2, "max": 4})
    discrete = distribution_from_config(
        {"type": "discrete", "values": [1, 2], "probabilities": [0.25, 0.75]}
    )
    assert all(5 <= lognormal.sample(rng) <= 8 for _ in range(20))
    assert all(bounded.sample(rng) in {2.0, 3.0, 4.0} for _ in range(20))
    assert all(discrete.sample(rng) in {1.0, 2.0} for _ in range(20))


def test_invalid_discrete_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        distribution_from_config(
            {"type": "discrete", "values": [1, 2], "probabilities": [0.2, 0.2]}
        )
