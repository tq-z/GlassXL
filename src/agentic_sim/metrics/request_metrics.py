"""Per-agent and per-invocation metrics records."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from agentic_sim.workload.agent import Agent


@dataclass(frozen=True)
class AgentMetrics:
    agent_id: int
    arrival_time_s: float
    start_time_s: float
    finish_time_s: float
    total_latency_s: float
    active_execution_time_s: float
    tool_wait_time_s: float
    queueing_time_s: float
    num_turns: int
    final_context_tokens: int

    @classmethod
    def from_agent(cls, agent: Agent) -> AgentMetrics:
        if agent.start_time is None or agent.finish_time is None:
            raise ValueError("cannot create completed metrics for an incomplete agent")
        return cls(
            agent_id=agent.agent_id,
            arrival_time_s=agent.arrival_time,
            start_time_s=agent.start_time,
            finish_time_s=agent.finish_time,
            total_latency_s=agent.finish_time - agent.arrival_time,
            active_execution_time_s=agent.active_execution_time,
            tool_wait_time_s=agent.actual_tool_wait_time,
            queueing_time_s=agent.queueing_time,
            num_turns=agent.num_turns,
            final_context_tokens=agent.current_context_tokens,
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass
class InvocationMetrics:
    invocation_id: int
    agent_id: int
    turn_index: int
    ready_time_s: float
    finish_time_s: float | None = None
    queueing_time_s: float = 0.0
    service_time_s: float = 0.0

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)
