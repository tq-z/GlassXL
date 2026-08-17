"""Iteration-level continuous decode batching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agentic_sim.core.event import EventType
from agentic_sim.core.event_engine import EventEngine
from agentic_sim.hardware.device import DecodeSettings, Device
from agentic_sim.metrics.batching_metrics import DecodeIterationMetric
from agentic_sim.metrics.collector import MetricsCollector
from agentic_sim.runtime.phase_state import InvocationPhase, InvocationSession


@dataclass(frozen=True)
class DecodeStepCompletion:
    iteration_id: int
    sessions: tuple[InvocationSession, ...]
    service_time: float


class DecodeBatchController:
    def __init__(
        self,
        engine: EventEngine,
        device: Device,
        settings: DecodeSettings,
        metrics: MetricsCollector,
        on_session_admitted: Callable[[InvocationSession, float], None],
        on_token: Callable[[InvocationSession, float, float], None],
        on_session_complete: Callable[[InvocationSession], None],
    ) -> None:
        self.engine = engine
        self.device = device
        self.settings = settings
        self.metrics = metrics
        self.on_session_admitted = on_session_admitted
        self.on_token = on_token
        self.on_session_complete = on_session_complete
        self.waiting: list[InvocationSession] = []
        self.active: list[InvocationSession] = []
        self.in_flight = False
        self._launch_scheduled = False
        self._next_iteration_id = 0

    def enqueue(self, session: InvocationSession) -> None:
        session.phase = InvocationPhase.WAITING_DECODE
        session.decode_queued_at = self.engine.now
        self.waiting.append(session)
        self.metrics.decode_state_changed(self.engine.now, len(self.waiting), len(self.active))
        self._request_iteration()

    def handle_start(self) -> None:
        self._launch_scheduled = False
        if self.in_flight:
            return
        while self.waiting and len(self.active) < self.settings.max_active_sessions:
            session = self.waiting.pop(0)
            if session.remaining_output_tokens <= 0:
                session.phase = InvocationPhase.COMPLETE
                self.on_session_complete(session)
                continue
            session.phase = InvocationPhase.DECODING
            if session.decode_started_at is None:
                session.decode_started_at = self.engine.now
                queued_at = session.decode_queued_at if session.decode_queued_at is not None else self.engine.now
                self.on_session_admitted(session, self.engine.now - queued_at)
            self.active.append(session)
        self.metrics.decode_state_changed(self.engine.now, len(self.waiting), len(self.active))
        if not self.active:
            return

        batch_size = len(self.active)
        service_time = self.device.fixed_latency + max(
            1.0 / self.settings.single_session_tokens_per_second,
            batch_size / self.settings.max_system_tokens_per_second,
        )
        reservation = self.device.queue.reserve(self.engine.now, service_time)
        iteration_id = self._next_iteration_id
        self._next_iteration_id += 1
        completion = DecodeStepCompletion(iteration_id, tuple(self.active), service_time)
        self.in_flight = True
        self.metrics.decode_iteration_started(
            DecodeIterationMetric(
                iteration_id=iteration_id,
                start_time_s=self.engine.now,
                finish_time_s=reservation.finish_time,
                active_sessions=batch_size,
                waiting_sessions=len(self.waiting),
                emitted_tokens=batch_size,
            )
        )
        self.engine.schedule(reservation.finish_time, EventType.DECODE_STEP_FINISH, completion)

    def handle_finish(self, completion: DecodeStepCompletion) -> None:
        self.in_flight = False
        finished: list[InvocationSession] = []
        for session in completion.sessions:
            session.generated_tokens += 1
            session.remaining_output_tokens -= 1
            session.token_times.append(self.engine.now)
            self.on_token(session, self.engine.now, completion.service_time)
            if session.remaining_output_tokens == 0:
                finished.append(session)
        for session in finished:
            self.active.remove(session)
            session.phase = InvocationPhase.COMPLETE
            self.on_session_complete(session)
        self.metrics.decode_state_changed(self.engine.now, len(self.waiting), len(self.active))
        self._request_iteration()

    def _request_iteration(self) -> None:
        if self.in_flight or self._launch_scheduled or not (self.active or self.waiting):
            return
        self._launch_scheduled = True
        self.engine.schedule(self.engine.now, EventType.DECODE_ITERATION_START)
