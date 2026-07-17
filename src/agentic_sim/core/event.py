"""Discrete-event definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    AGENT_ARRIVAL = auto()
    INVOCATION_READY = auto()
    TASK_READY = auto()
    TASK_START = auto()
    TASK_FINISH = auto()
    TOOL_WAIT_START = auto()
    TOOL_RETURN = auto()
    AGENT_FINISH = auto()


@dataclass(order=True, frozen=True)
class Event:
    time: float
    sequence_id: int
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False)
