"""Deterministic fluid equal-share bandwidth pool."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentic_sim.core.event import EventType
from agentic_sim.core.event_engine import EventEngine


@dataclass
class BandwidthFlow:
    flow_id: int
    payload: Any
    byte_count: int
    bytes_remaining: float
    submitted_time: float
    data_start_time: float | None = None


@dataclass(frozen=True)
class PoolCompletion:
    generation: int
    flow_id: int


class BandwidthPool:
    def __init__(
        self,
        engine: EventEngine,
        fixed_latency: float,
        per_session_max_bandwidth: float,
        max_system_bandwidth: float,
        on_flow_complete: Callable[[BandwidthFlow, float], None],
        on_active_change: Callable[[float, int, float], None],
    ) -> None:
        if fixed_latency < 0.0:
            raise ValueError("bandwidth-pool latency must be non-negative")
        if per_session_max_bandwidth <= 0.0 or max_system_bandwidth <= 0.0:
            raise ValueError("bandwidth-pool caps must be positive")
        self.engine = engine
        self.fixed_latency = fixed_latency
        self.per_session_max_bandwidth = per_session_max_bandwidth
        self.max_system_bandwidth = max_system_bandwidth
        self.on_flow_complete = on_flow_complete
        self.on_active_change = on_active_change
        self.active: dict[int, BandwidthFlow] = {}
        self.flows: dict[int, BandwidthFlow] = {}
        self._next_flow_id = 0
        self._generation = 0
        self._last_update_time = 0.0

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active_count(self) -> int:
        return len(self.active)

    @property
    def flow_bandwidth(self) -> float:
        if not self.active:
            return 0.0
        return min(
            self.per_session_max_bandwidth,
            self.max_system_bandwidth / len(self.active),
        )

    def submit(self, payload: Any, byte_count: int, now: float) -> BandwidthFlow:
        if byte_count < 0:
            raise ValueError("flow byte count must be non-negative")
        flow = BandwidthFlow(
            flow_id=self._next_flow_id,
            payload=payload,
            byte_count=byte_count,
            bytes_remaining=float(byte_count),
            submitted_time=now,
        )
        self._next_flow_id += 1
        self.flows[flow.flow_id] = flow
        self.engine.schedule(now + self.fixed_latency, EventType.KV_INGRESS_DATA_READY, flow.flow_id)
        return flow

    def handle_data_ready(self, flow_id: int, now: float) -> None:
        self._advance(now)
        flow = self.flows[flow_id]
        flow.data_start_time = now
        self.active[flow_id] = flow
        self._generation += 1
        self._notify_active(now)
        if flow.bytes_remaining <= 0.0:
            self._complete_finished(now)
        else:
            self._schedule_next(now)

    def handle_completion(self, completion: PoolCompletion, now: float) -> bool:
        if completion.generation != self._generation:
            return False
        self._advance(now)
        self._complete_finished(now, expected_flow_id=completion.flow_id)
        return True

    def _advance(self, now: float) -> None:
        if now < self._last_update_time:
            raise RuntimeError("bandwidth pool moved backwards in time")
        elapsed = now - self._last_update_time
        rate = self.flow_bandwidth
        if rate > 0.0 and elapsed > 0.0:
            transferred = elapsed * rate
            for flow in self.active.values():
                flow.bytes_remaining = max(0.0, flow.bytes_remaining - transferred)
        self._last_update_time = now

    def _complete_finished(self, now: float, expected_flow_id: int | None = None) -> None:
        tolerance = max(1e-9, self.flow_bandwidth * 1e-12)
        finished = [
            flow for flow in self.active.values() if flow.bytes_remaining <= tolerance
        ]
        if not finished and expected_flow_id is not None and expected_flow_id in self.active:
            finished = [self.active[expected_flow_id]]
            finished[0].bytes_remaining = 0.0
        for flow in sorted(finished, key=lambda item: item.flow_id):
            del self.active[flow.flow_id]
        if finished:
            self._generation += 1
            self._notify_active(now)
            for flow in sorted(finished, key=lambda item: item.flow_id):
                self.on_flow_complete(flow, now)
        self._schedule_next(now)

    def _schedule_next(self, now: float) -> None:
        if not self.active:
            return
        rate = self.flow_bandwidth
        next_flow = min(
            self.active.values(),
            key=lambda flow: (flow.bytes_remaining, flow.flow_id),
        )
        finish_time = now + next_flow.bytes_remaining / rate
        self.engine.schedule(
            finish_time,
            EventType.KV_INGRESS_FINISH,
            PoolCompletion(self._generation, next_flow.flow_id),
        )

    def _notify_active(self, now: float) -> None:
        self.on_active_change(now, len(self.active), self.flow_bandwidth * len(self.active))

