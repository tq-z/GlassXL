"""Invocation-level state for dynamic prefill, ingress, and decode."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from agentic_sim.workload.invocation import Invocation


class InvocationPhase(Enum):
    WAITING_RESTORE = auto()
    WAITING_PREFILL = auto()
    PREFILL_RUNNING = auto()
    WAITING_KV_INGRESS = auto()
    KV_INGRESS_RUNNING = auto()
    WAITING_DECODE = auto()
    DECODING = auto()
    COMPLETE = auto()


@dataclass
class InvocationSession:
    invocation: Invocation
    effective_prefill_tokens: int
    prompt_kv_bytes: int
    remaining_output_tokens: int
    phase: InvocationPhase
    generated_tokens: int = 0
    prefill_queued_at: float | None = None
    prefill_started_at: float | None = None
    prefill_finished_at: float | None = None
    ingress_submitted_at: float | None = None
    ingress_finished_at: float | None = None
    decode_queued_at: float | None = None
    decode_started_at: float | None = None
    token_times: list[float] = field(default_factory=list)

    @property
    def session_id(self) -> int:
        return self.invocation.invocation_id

    @property
    def agent_id(self) -> int:
        return self.invocation.agent_id

    @property
    def final_context_tokens(self) -> int:
        return (
            self.invocation.context_tokens
            + self.invocation.input_tokens
            + self.generated_tokens
        )

