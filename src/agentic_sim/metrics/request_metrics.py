"""Per-agent and per-invocation metrics records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0

    @classmethod
    def from_agent(
        cls,
        agent: Agent,
        prefill_time_s: float = 0.0,
        decode_time_s: float = 0.0,
    ) -> AgentMetrics:
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
            prefill_time_s=prefill_time_s,
            decode_time_s=decode_time_s,
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
    prefill_queued_time_s: float | None = None
    prefill_start_time_s: float | None = None
    prefill_finish_time_s: float | None = None
    ingress_submit_time_s: float | None = None
    ingress_finish_time_s: float | None = None
    decode_queued_time_s: float | None = None
    decode_start_time_s: float | None = None
    first_token_time_s: float | None = None
    token_times_s: list[float] = field(default_factory=list)
    prefill_queueing_time_s: float = 0.0
    decode_queueing_time_s: float = 0.0

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_time_s is None:
            return None
        return self.first_token_time_s - self.ready_time_s

    @property
    def inter_token_times_s(self) -> list[float]:
        return [later - earlier for earlier, later in zip(self.token_times_s, self.token_times_s[1:])]

    @property
    def prefill_time_s(self) -> float:
        if self.prefill_start_time_s is None or self.prefill_finish_time_s is None:
            return 0.0
        return self.prefill_finish_time_s - self.prefill_start_time_s

    @property
    def decode_time_s(self) -> float:
        if self.decode_start_time_s is None or self.finish_time_s is None:
            return 0.0
        return self.finish_time_s - self.decode_start_time_s

    def to_dict(self) -> dict[str, int | float | list[float] | None]:
        result = asdict(self)
        result["prefill_time_s"] = self.prefill_time_s
        result["decode_time_s"] = self.decode_time_s
        result["ttft_s"] = self.ttft_s
        result["inter_token_times_s"] = self.inter_token_times_s
        return result
