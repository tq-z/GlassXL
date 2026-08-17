"""Seeded synthetic generation of immutable V1 session records."""

from __future__ import annotations

import random

from agentic_sim.config.models import WorkloadConfig
from agentic_sim.workload.agent import SessionSpec
from agentic_sim.workload.arrival import poisson_arrival_times
from agentic_sim.workload.distributions import distribution_from_config
from agentic_sim.workload.invocation import TurnSpec


class WorkloadGenerator:
    """Generate the same typed workload for every call under one fixed seed."""

    def __init__(self, config: WorkloadConfig) -> None:
        self.config = config

    def generate(self) -> tuple[SessionSpec, ...]:
        rng = random.Random(self.config.seed)
        source = self.config.source
        arrival_times = poisson_arrival_times(
            source.num_sessions,
            source.arrival.rate_per_second,
            rng,
        )
        turns_distribution = distribution_from_config(source.turns)
        context_distribution = distribution_from_config(source.context_tokens)
        prefill_distribution = distribution_from_config(source.prefill_tokens)
        output_distribution = distribution_from_config(source.output_tokens)
        wait_distribution = distribution_from_config(source.tool_wait_ms)

        sessions: list[SessionSpec] = []
        for session_id, arrival_time_s in enumerate(arrival_times):
            num_turns = int(round(turns_distribution.sample(rng)))
            initial_context_tokens = int(round(context_distribution.sample(rng)))
            turns = tuple(
                TurnSpec(
                    turn_index=turn_index,
                    prefill_tokens=int(round(prefill_distribution.sample(rng))),
                    output_tokens=int(round(output_distribution.sample(rng))),
                    tool_wait_s=wait_distribution.sample(rng) / 1_000.0,
                )
                for turn_index in range(num_turns)
            )
            sessions.append(
                SessionSpec(
                    session_id=session_id,
                    arrival_time_s=arrival_time_s,
                    initial_context_tokens=initial_context_tokens,
                    turns=turns,
                )
            )
        return tuple(sessions)
