"""Deterministic FCFS greedy batching for typed prefill sessions."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from agentic_sim.runtime.session import DecodeEpisodePhase, SessionState


def _time(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class PrefillRequest:
    """One session waiting on its selected prefill device."""

    sequence_id: int
    ready_time_s: float
    device_id: int
    session: SessionState


@dataclass(frozen=True)
class PrefillBatch:
    """All currently ready FCFS requests greedily selected for one device."""

    batch_id: int
    device_id: int
    launch_time_s: float
    requests: tuple[PrefillRequest, ...]


class PrefillBatcher:
    """Form one unbounded greedy batch per idle device without formation wait."""

    def __init__(self, device_ids: tuple[int, ...]) -> None:
        if not device_ids or len(set(device_ids)) != len(device_ids):
            raise ValueError("prefill batcher requires unique non-empty device IDs")
        if any(
            isinstance(device_id, bool)
            or not isinstance(device_id, int)
            or device_id < 0
            for device_id in device_ids
        ):
            raise ValueError("prefill device IDs must be non-negative integers")
        self.device_ids = device_ids
        self._waiting: dict[int, deque[PrefillRequest]] = {
            device_id: deque() for device_id in device_ids
        }
        self._next_request_sequence = 0
        self._next_batch_id = 0
        self._last_enqueue_time_s = 0.0

    @property
    def last_enqueue_time_s(self) -> float:
        return self._last_enqueue_time_s

    def enqueue(
        self,
        session: SessionState,
        device_id: int,
        ready_time_s: float,
    ) -> PrefillRequest:
        ready = _time(ready_time_s, "ready_time_s")
        if ready < self._last_enqueue_time_s:
            raise ValueError("prefill requests must be enqueued in timestamp order")
        if device_id not in self._waiting:
            raise ValueError(f"device {device_id} is not a mapped prefill device")
        episode = session.current_episode
        if episode is None or episode.phase is not DecodeEpisodePhase.TURN_READY:
            raise ValueError("prefill enqueue requires a turn-ready episode")
        if ready < episode.turn_ready_time_s:
            raise ValueError("prefill enqueue cannot precede turn readiness")
        if any(
            request.session is session
            for queue in self._waiting.values()
            for request in queue
        ):
            raise ValueError("session is already waiting for prefill")
        request = PrefillRequest(
            sequence_id=self._next_request_sequence,
            ready_time_s=ready,
            device_id=device_id,
            session=session,
        )
        self._waiting[device_id].append(request)
        self._next_request_sequence += 1
        self._last_enqueue_time_s = ready
        return request

    def form_batch(
        self,
        device_id: int,
        launch_time_s: float,
        *,
        device_available_time_s: float,
    ) -> PrefillBatch | None:
        launch = _time(launch_time_s, "launch_time_s")
        available = _time(device_available_time_s, "device_available_time_s")
        if device_id not in self._waiting:
            raise ValueError(f"device {device_id} is not a mapped prefill device")
        if available > launch:
            return None
        queue = self._waiting[device_id]
        selected: list[PrefillRequest] = []
        while queue and queue[0].ready_time_s <= launch:
            selected.append(queue.popleft())
        if not selected:
            return None
        batch = PrefillBatch(
            batch_id=self._next_batch_id,
            device_id=device_id,
            launch_time_s=launch,
            requests=tuple(selected),
        )
        self._next_batch_id += 1
        return batch

    def waiting(self, device_id: int | None = None) -> tuple[PrefillRequest, ...]:
        if device_id is not None:
            if device_id not in self._waiting:
                raise ValueError(f"device {device_id} is not a mapped prefill device")
            return tuple(self._waiting[device_id])
        return tuple(
            sorted(
                (
                    request
                    for queue in self._waiting.values()
                    for request in queue
                ),
                key=lambda request: request.sequence_id,
            )
        )
