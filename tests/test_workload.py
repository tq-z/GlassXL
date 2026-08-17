from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from agentic_sim.config import load_workload_config
from agentic_sim.config.models import (
    ConstantDistributionConfig,
    DistributionType,
    PoissonArrivalConfig,
    SyntheticSourceConfig,
    WorkloadConfig,
)
from agentic_sim.workload import SessionSpec, WorkloadGenerator


EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "example" / "workload.yaml"


def constant_workload(*, seed: int = 7, num_sessions: int = 2) -> WorkloadConfig:
    constant = DistributionType.CONSTANT
    return WorkloadConfig(
        schema_version=1,
        seed=seed,
        simulation_end_s=None,
        source=SyntheticSourceConfig(
            type="synthetic",
            num_sessions=num_sessions,
            arrival=PoissonArrivalConfig("poisson", 1.0),
            turns=ConstantDistributionConfig(constant, 2),
            context_tokens=ConstantDistributionConfig(constant, 10),
            prefill_tokens=ConstantDistributionConfig(constant, 3),
            output_tokens=ConstantDistributionConfig(constant, 2),
            tool_wait_ms=ConstantDistributionConfig(constant, 100.0),
        ),
        source_path=Path("workload.yaml"),
    )


def test_fixed_seed_generation_is_identical_across_calls_and_instances() -> None:
    config = load_workload_config(EXAMPLE)
    generator = WorkloadGenerator(config)

    first = generator.generate()
    second = generator.generate()
    third = WorkloadGenerator(config).generate()

    assert first == second == third
    assert len(first) == 1000
    assert tuple(session.arrival_time_s for session in first) == tuple(
        sorted(session.arrival_time_s for session in first)
    )


def test_different_seed_changes_the_synthetic_workload() -> None:
    config = constant_workload(seed=4)

    assert WorkloadGenerator(config).generate() != WorkloadGenerator(
        replace(config, seed=5)
    ).generate()


def test_warm_context_and_every_turn_prefill_are_distinct() -> None:
    sessions = WorkloadGenerator(constant_workload()).generate()
    session = sessions[0]

    assert isinstance(session, SessionSpec)
    assert session.initial_context_tokens == 10
    assert session.num_turns == 2
    assert [turn.prefill_tokens for turn in session.turns] == [3, 3]
    assert [turn.output_tokens for turn in session.turns] == [2, 2]
    assert [turn.tool_wait_s for turn in session.turns] == pytest.approx([0.1, 0.1])
    assert session.turns[0].prefill_tokens > 0


def test_generated_workload_records_are_recursively_immutable() -> None:
    session = WorkloadGenerator(constant_workload()).generate()[0]

    assert isinstance(session.turns, tuple)
    with pytest.raises(FrozenInstanceError):
        session.initial_context_tokens = 11  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        session.turns[0].prefill_tokens = 4  # type: ignore[misc]


def test_zero_session_workload_is_supported() -> None:
    assert WorkloadGenerator(constant_workload(num_sessions=0)).generate() == ()
