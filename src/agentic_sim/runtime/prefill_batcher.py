"""Dynamic FCFS whole-prompt prefill batching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agentic_sim.core.event import EventType
from agentic_sim.core.event_engine import EventEngine
from agentic_sim.hardware.device import Device, PrefillSettings
from agentic_sim.metrics.batching_metrics import PrefillBatchMetric
from agentic_sim.metrics.collector import MetricsCollector
from agentic_sim.runtime.phase_state import InvocationPhase, InvocationSession


@dataclass(frozen=True)
class PrefillBatchCompletion:
    batch_id: int
    sessions: tuple[InvocationSession, ...]


class PrefillBatchController:
    def __init__(
        self,
        engine: EventEngine,
        device: Device,
        settings: PrefillSettings,
        metrics: MetricsCollector,
        on_session_start: Callable[[InvocationSession, float, float], None],
        on_session_complete: Callable[[InvocationSession], None],
    ) -> None:
        self.engine = engine
        self.device = device
        self.settings = settings
        self.metrics = metrics
        self.on_session_start = on_session_start
        self.on_session_complete = on_session_complete
        self.waiting: list[InvocationSession] = []
        self.busy = False
        self._launch_scheduled = False
        self._next_batch_id = 0

    def enqueue(self, session: InvocationSession) -> None:
        session.phase = InvocationPhase.WAITING_PREFILL
        session.prefill_queued_at = self.engine.now
        self.waiting.append(session)
        self.metrics.prefill_queue_changed(self.engine.now, len(self.waiting))
        self._request_launch()

    def handle_start(self) -> None:
        self._launch_scheduled = False
        if self.busy or not self.waiting:
            return
        selected = [self.waiting.pop(0)]
        total_tokens = selected[0].effective_prefill_tokens
        while self.waiting and len(selected) < self.settings.max_batch_requests:
            candidate = self.waiting[0]
            if total_tokens + candidate.effective_prefill_tokens > self.settings.max_batch_tokens:
                break
            selected.append(self.waiting.pop(0))
            total_tokens += candidate.effective_prefill_tokens
        self.metrics.prefill_queue_changed(self.engine.now, len(self.waiting))

        longest = max(session.effective_prefill_tokens for session in selected)
        service_time = self.device.fixed_latency + max(
            longest / self.settings.single_request_tokens_per_second,
            total_tokens / self.settings.max_system_tokens_per_second,
        )
        reservation = self.device.queue.reserve(self.engine.now, service_time)
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        completion = PrefillBatchCompletion(batch_id, tuple(selected))
        self.busy = True
        for session in selected:
            session.phase = InvocationPhase.PREFILL_RUNNING
            session.prefill_started_at = self.engine.now
            queued_at = session.prefill_queued_at if session.prefill_queued_at is not None else self.engine.now
            self.on_session_start(session, self.engine.now - queued_at, service_time)
        self.metrics.prefill_batch_started(
            PrefillBatchMetric(
                batch_id=batch_id,
                start_time_s=self.engine.now,
                finish_time_s=reservation.finish_time,
                request_count=len(selected),
                total_tokens=total_tokens,
                longest_tokens=longest,
                queue_length_at_start=len(self.waiting),
            )
        )
        self.engine.schedule(reservation.finish_time, EventType.PREFILL_BATCH_FINISH, completion)

    def handle_finish(self, completion: PrefillBatchCompletion) -> None:
        self.busy = False
        for session in completion.sessions:
            session.prefill_finished_at = self.engine.now
            session.phase = InvocationPhase.WAITING_KV_INGRESS
            self.on_session_complete(session)
        self._request_launch()

    def _request_launch(self) -> None:
        if self.busy or self._launch_scheduled or not self.waiting:
            return
        self._launch_scheduled = True
        self.engine.schedule(self.engine.now, EventType.PREFILL_BATCH_START)
