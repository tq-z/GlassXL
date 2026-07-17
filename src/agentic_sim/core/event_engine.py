"""Deterministic priority-queue event engine."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Any

from agentic_sim.core.event import Event, EventType

EventHandler = Callable[[Event], None]


class EventEngine:
    """Own simulation time and dispatch events in deterministic order."""

    def __init__(self, end_time: float | None = None) -> None:
        if end_time is not None and end_time < 0.0:
            raise ValueError("end_time must be non-negative")
        self.now = 0.0
        self.end_time = end_time
        self._next_sequence_id = 0
        self._queue: list[Event] = []
        self._handlers: dict[EventType, EventHandler] = {}
        self.stopped_at_end_time = False

    @property
    def pending_events(self) -> int:
        return len(self._queue)

    def register_handler(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers[event_type] = handler

    def schedule(self, time: float, event_type: EventType, payload: Any = None) -> Event:
        if time < self.now:
            raise ValueError(f"cannot schedule event in the past: {time} < {self.now}")
        event = Event(time, self._next_sequence_id, event_type, payload)
        self._next_sequence_id += 1
        heapq.heappush(self._queue, event)
        return event

    def run(self) -> None:
        while self._queue:
            event = heapq.heappop(self._queue)
            if self.end_time is not None and event.time > self.end_time:
                heapq.heappush(self._queue, event)
                self.now = self.end_time
                self.stopped_at_end_time = True
                return
            self.now = event.time
            try:
                handler = self._handlers[event.event_type]
            except KeyError as error:
                raise RuntimeError(f"no handler registered for {event.event_type.name}") from error
            handler(event)
