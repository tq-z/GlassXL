"""Event-driven assembly of the reconstructed PD/AFD controllers."""

from __future__ import annotations

from collections.abc import Sequence

from agentic_sim.config import SimulationConfig
from agentic_sim.core.event import Event, EventType
from agentic_sim.core.event_engine import EventEngine
from agentic_sim.hardware.topology import Topology
from agentic_sim.metrics import MetricsCollector, MetricsSnapshot
from agentic_sim.runtime.attention_admission import AttentionAdmissionController
from agentic_sim.runtime.decode_controller import DecodeController, PendingPoolWrite
from agentic_sim.runtime.prefill_controller import PrefillController
from agentic_sim.runtime.session import SessionState
from agentic_sim.workload import SessionSpec, WorkloadGenerator


class AgenticRuntime:
    """Own session lifecycle and stage dependencies on one deterministic calendar."""

    def __init__(
        self,
        config: SimulationConfig,
        session_specs: Sequence[SessionSpec] | None = None,
    ) -> None:
        self.config = config
        self.topology = Topology.from_config(config.hardware)
        self.attention = AttentionAdmissionController.from_config(
            config.mapping.decode_attn,
            self.topology.devices,
        )
        self.prefill = PrefillController(
            self.topology,
            config.mapping.prefill,
            self.attention,
        )
        self.decode = DecodeController(
            self.topology,
            config.mapping,
            self.attention,
        )
        specs = tuple(session_specs) if session_specs is not None else WorkloadGenerator(config.workload).generate()
        if len({item.session_id for item in specs}) != len(specs):
            raise ValueError("runtime session IDs must be unique")
        self.sessions = {
            item.session_id: SessionState(item)
            for item in sorted(specs, key=lambda item: item.session_id)
        }
        self.engine = EventEngine(end_time=config.workload.simulation_end_s)
        self.metrics = MetricsCollector()
        self._prefill_launch_time_s: float | None = None
        self._decode_launch_time_s: float | None = None
        self._decode_available_time_s = 0.0
        self._register_handlers()
        for session in self.sessions.values():
            self.engine.schedule(
                session.spec.arrival_time_s,
                EventType.SESSION_ARRIVAL,
                session.session_id,
            )
        self.metrics.capture_attention(0.0, self.attention)

    def run(self) -> MetricsSnapshot:
        self.engine.run()
        return self.metrics.snapshot(
            tuple(self.sessions.values()),
            self.topology,
            self.engine.now,
        )

    def _register_handlers(self) -> None:
        self.engine.register_handler(EventType.SESSION_ARRIVAL, self._on_session_arrival)
        self.engine.register_handler(EventType.TURN_READY, self._on_turn_ready)
        self.engine.register_handler(EventType.PREFILL_LAUNCH, self._on_prefill_launch)
        self.engine.register_handler(EventType.DECODE_LAUNCH, self._on_decode_launch)
        self.engine.register_handler(
            EventType.POOL_WRITE_COMPLETE,
            self._on_pool_write_complete,
        )

    def _on_session_arrival(self, event: Event) -> None:
        session = self.sessions[int(event.payload)]
        session.arrive(event.time)
        self._start_turn(session, event.time)

    def _on_turn_ready(self, event: Event) -> None:
        session = self.sessions[int(event.payload)]
        self._start_turn(session, event.time)

    def _start_turn(self, session: SessionState, ready_time_s: float) -> None:
        session.start_episode(ready_time_s)
        self.prefill.enqueue(session, ready_time_s)
        self._request_prefill_launch(ready_time_s)

    def _request_prefill_launch(self, ready_time_s: float) -> None:
        next_time = self.prefill.next_launch_time_s()
        if next_time is None:
            return
        target = max(ready_time_s, next_time)
        if self._prefill_launch_time_s is None or target < self._prefill_launch_time_s:
            self._prefill_launch_time_s = target
            self.engine.schedule(target, EventType.PREFILL_LAUNCH, target)

    def _on_prefill_launch(self, event: Event) -> None:
        if self._prefill_launch_time_s != event.time:
            return
        self._prefill_launch_time_s = None
        results = self.prefill.process_ready(event.time)
        for result in results:
            self.metrics.record_prefill(result, self.config.mapping.prefill.state.value)
            for transfer in result.transfers:
                self._request_decode_launch(transfer.reservation.finish_time)
        if results:
            snapshot_time = max(
                (
                    transfer.reservation.finish_time
                    for result in results
                    for transfer in result.transfers
                ),
                default=max(result.compute.finish_time for result in results),
            )
            self.metrics.capture_attention(snapshot_time, self.attention)
        self._request_prefill_launch(event.time)

    def _request_decode_launch(self, ready_time_s: float) -> None:
        target = max(ready_time_s, self._decode_available_time_s)
        if self._decode_launch_time_s is None or target < self._decode_launch_time_s:
            self._decode_launch_time_s = target
            self.engine.schedule(target, EventType.DECODE_LAUNCH, target)

    def _on_decode_launch(self, event: Event) -> None:
        if self._decode_launch_time_s != event.time:
            return
        self._decode_launch_time_s = None

        if self.decode.ready_zero_output_sessions(event.time):
            pending = self.decode.flush_zero_output_sessions(event.time)
            for item in pending:
                self.metrics.record_pool_write(item)
                self._schedule_pool_completion(item)
            self.metrics.capture_attention(event.time, self.attention)

        if not self.decode.ready_sessions(event.time):
            return
        result = self.decode.run_step(event.time)
        self._decode_available_time_s = result.token_completion_time_s
        self.metrics.record_decode(result)
        for pending in result.pool_writes:
            self._schedule_pool_completion(pending)
        self.metrics.capture_attention(result.token_completion_time_s, self.attention)
        if any(
            session.current_episode is not None
            and session.current_episode.remaining_output_tokens > 0
            for session in result.batch.sessions
        ):
            self._request_decode_launch(result.token_completion_time_s)

    def _schedule_pool_completion(self, pending: PendingPoolWrite) -> None:
        self.engine.schedule(
            pending.reservation.finish_time,
            EventType.POOL_WRITE_COMPLETE,
            pending.session_id,
        )

    def _on_pool_write_complete(self, event: Event) -> None:
        session_id = int(event.payload)
        pending = next(
            item
            for item in self.decode.pending_pool_writes
            if item.session_id == session_id
        )
        session = pending.session
        episode = session.current_episode
        if episode is None:
            raise RuntimeError("pool completion lost its decode episode")
        decode_finish = (
            episode.token_completion_times_s[-1]
            if episode.token_completion_times_s
            else (episode.attention_ready_time_s or event.time)
        )
        tool_ready = decode_finish + episode.turn.tool_wait_s
        completed = self.decode.complete_pool_write(session_id)
        transfers = self.prefill.complete_attention_admissions(
            completed.completion.admissions
        )
        self.metrics.record_transfers(transfers)
        for transfer in transfers:
            self._request_decode_launch(transfer.reservation.finish_time)
        session.complete_episode()
        if not session.is_finished:
            self.engine.schedule(
                max(event.time, tool_ready),
                EventType.TURN_READY,
                session_id,
            )
        self.metrics.capture_attention(event.time, self.attention)
