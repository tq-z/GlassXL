"""Immutable per-turn workload specification."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _nonnegative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True)
class TurnSpec:
    """Oracle input/output and tool wait for one decode episode."""

    turn_index: int
    prefill_tokens: int
    output_tokens: int
    tool_wait_s: float

    def __post_init__(self) -> None:
        _nonnegative_integer(self.turn_index, "turn_index")
        _nonnegative_integer(self.prefill_tokens, "prefill_tokens")
        _nonnegative_integer(self.output_tokens, "output_tokens")
        if (
            isinstance(self.tool_wait_s, bool)
            or not isinstance(self.tool_wait_s, (int, float))
            or not math.isfinite(float(self.tool_wait_s))
            or self.tool_wait_s < 0.0
        ):
            raise ValueError("tool_wait_s must be a finite non-negative number")
