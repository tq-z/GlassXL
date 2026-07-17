"""Seeded synthetic multi-turn workload generation."""

from __future__ import annotations

import random
from typing import Any

from agentic_sim.workload.agent import Agent, AgentState
from agentic_sim.workload.arrival import poisson_arrival_times
from agentic_sim.workload.distributions import distribution_from_config


class WorkloadGenerator:
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.rng = random.Random(seed)

    def generate(self) -> list[Agent]:
        num_agents = int(self.config["num_agents"])
        arrival = self.config["arrival"]
        if str(arrival.get("type", "")).lower() != "poisson":
            raise ValueError("PR2 supports only Poisson arrivals")
        arrival_times = poisson_arrival_times(
            num_agents,
            float(arrival["rate_per_second"]),
            self.rng,
        )
        turns_distribution = distribution_from_config(self.config["turns"])
        input_distribution = distribution_from_config(self.config["input_tokens"])
        output_distribution = distribution_from_config(self.config["output_tokens"])
        wait_distribution = distribution_from_config(self.config["tool_wait_ms"])

        agents: list[Agent] = []
        for agent_id, arrival_time in enumerate(arrival_times):
            num_turns = max(1, int(round(turns_distribution.sample(self.rng))))
            inputs = [max(0, int(round(input_distribution.sample(self.rng)))) for _ in range(num_turns)]
            outputs = [max(0, int(round(output_distribution.sample(self.rng)))) for _ in range(num_turns)]
            waits = [max(0.0, wait_distribution.sample(self.rng) / 1_000.0) for _ in range(num_turns)]
            agents.append(
                Agent(
                    agent_id=agent_id,
                    arrival_time=arrival_time,
                    state=AgentState.NOT_ARRIVED,
                    num_turns=num_turns,
                    current_turn=0,
                    input_tokens_per_turn=inputs,
                    output_tokens_per_turn=outputs,
                    tool_wait_per_turn=waits,
                )
            )
        return agents
