"""Hardware-independent model invocation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KVRetentionMode(Enum):
    SAME_DEVICE = "same_device"
    MEMORY = "memory"
    FAR_MEMORY = "far_memory"
    DISCARD = "discard"


@dataclass(frozen=True)
class Invocation:
    invocation_id: int
    agent_id: int
    turn_index: int
    input_tokens: int
    output_tokens: int
    context_tokens: int
    ready_time: float
