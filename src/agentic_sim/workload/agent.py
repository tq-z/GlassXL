"""Immutable long-lived session workload specification."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agentic_sim.workload.invocation import TurnSpec, _nonnegative_integer


@dataclass(frozen=True)
class SessionSpec:
    """One warm-start session and all of its oracle turn records."""

    session_id: int
    arrival_time_s: float
    initial_context_tokens: int
    turns: tuple[TurnSpec, ...]

    def __post_init__(self) -> None:
        _nonnegative_integer(self.session_id, "session_id")
        _nonnegative_integer(self.initial_context_tokens, "initial_context_tokens")
        if (
            isinstance(self.arrival_time_s, bool)
            or not isinstance(self.arrival_time_s, (int, float))
            or not math.isfinite(float(self.arrival_time_s))
            or self.arrival_time_s < 0.0
        ):
            raise ValueError("arrival_time_s must be a finite non-negative number")
        if not self.turns:
            raise ValueError("a session must contain at least one turn")
        expected_indices = tuple(range(len(self.turns)))
        actual_indices = tuple(turn.turn_index for turn in self.turns)
        if actual_indices != expected_indices:
            raise ValueError("turn indices must be contiguous and start at zero")

    @property
    def num_turns(self) -> int:
        return len(self.turns)
