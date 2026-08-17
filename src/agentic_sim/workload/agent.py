"""Multi-turn agent state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class AgentState(Enum):
    NOT_ARRIVED = auto()
    READY = auto()
    RUNNING = auto()
    TOOL_WAIT = auto()
    KV_WAIT = auto()
    FINISHED = auto()


@dataclass
class Agent:
    agent_id: int
    arrival_time: float
    state: AgentState
    num_turns: int
    current_turn: int
    input_tokens_per_turn: list[int]
    output_tokens_per_turn: list[int]
    tool_wait_per_turn: list[float]
    initial_context_tokens: int | None = None
    current_context_tokens: int = 0
    start_time: float | None = None
    finish_time: float | None = None
    active_execution_time: float = 0.0
    queueing_time: float = 0.0
    actual_tool_wait_time: float = 0.0

    def __post_init__(self) -> None:
        if self.num_turns <= 0:
            raise ValueError("an agent must have at least one turn")
        for values in (
            self.input_tokens_per_turn,
            self.output_tokens_per_turn,
            self.tool_wait_per_turn,
        ):
            if len(values) != self.num_turns:
                raise ValueError("per-turn arrays must match num_turns")
        if self.initial_context_tokens is not None and self.initial_context_tokens < 0:
            raise ValueError("initial context tokens must be non-negative")

    @property
    def is_last_turn(self) -> bool:
        return self.current_turn == self.num_turns - 1

    @property
    def current_input_tokens(self) -> int:
        """Return the first prompt or the current turn's newly arrived input."""

        if self.current_turn == 0 and self.initial_context_tokens is not None:
            return self.initial_context_tokens
        return self.input_tokens_per_turn[self.current_turn]

    def complete_current_turn(self) -> None:
        self.current_context_tokens += self.current_input_tokens + self.output_tokens_per_turn[self.current_turn]
