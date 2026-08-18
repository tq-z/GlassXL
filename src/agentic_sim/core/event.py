"""Discrete events used by the reconstructed PD/AFD runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    SESSION_ARRIVAL = auto()
    TURN_READY = auto()
    PREFILL_LAUNCH = auto()
    DECODE_LAUNCH = auto()
    POOL_WRITE_COMPLETE = auto()


@dataclass(order=True, frozen=True)
class Event:
    time: float
    sequence_id: int
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False)
