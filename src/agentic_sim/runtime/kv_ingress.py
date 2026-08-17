"""Per-session P-to-D KV ingress controller."""

from __future__ import annotations

from collections.abc import Callable

from agentic_sim.core.event_engine import EventEngine
from agentic_sim.hardware.bandwidth_pool import BandwidthFlow, BandwidthPool, PoolCompletion
from agentic_sim.metrics.batching_metrics import IngressFlowMetric
from agentic_sim.metrics.collector import MetricsCollector
from agentic_sim.runtime.phase_state import InvocationPhase, InvocationSession


class KVIngressController:
    def __init__(
        self,
        engine: EventEngine,
        fixed_latency: float,
        per_session_max_bandwidth: float,
        max_system_bandwidth: float,
        metrics: MetricsCollector,
        on_session_complete: Callable[[InvocationSession, float], None],
    ) -> None:
        self.engine = engine
        self.metrics = metrics
        self.on_session_complete = on_session_complete
        self.pool = BandwidthPool(
            engine=engine,
            fixed_latency=fixed_latency,
            per_session_max_bandwidth=per_session_max_bandwidth,
            max_system_bandwidth=max_system_bandwidth,
            on_flow_complete=self._on_flow_complete,
            on_active_change=self.metrics.ingress_active_changed,
        )

    def submit(self, session: InvocationSession) -> None:
        session.phase = InvocationPhase.KV_INGRESS_RUNNING
        session.ingress_submitted_at = self.engine.now
        flow = self.pool.submit(session, session.prompt_kv_bytes, self.engine.now)
        self.metrics.ingress_flow_submitted(flow.flow_id, session.session_id, flow.byte_count)

    def handle_data_ready(self, flow_id: int) -> None:
        self.pool.handle_data_ready(flow_id, self.engine.now)

    def handle_finish(self, completion: PoolCompletion) -> bool:
        return self.pool.handle_completion(completion, self.engine.now)

    def _on_flow_complete(self, flow: BandwidthFlow, finish_time: float) -> None:
        session = flow.payload
        if not isinstance(session, InvocationSession):
            raise TypeError("KV ingress flow payload must be an InvocationSession")
        data_start = flow.data_start_time if flow.data_start_time is not None else finish_time
        data_duration = finish_time - data_start
        achieved = flow.byte_count / data_duration if data_duration > 0.0 else 0.0
        session.ingress_finished_at = finish_time
        session.phase = InvocationPhase.WAITING_DECODE
        self.metrics.ingress_flow_finished(
            IngressFlowMetric(
                flow_id=flow.flow_id,
                session_id=session.session_id,
                submitted_time_s=flow.submitted_time,
                data_start_time_s=data_start,
                finish_time_s=finish_time,
                byte_count=flow.byte_count,
                achieved_bandwidth_bytes_per_s=achieved,
            )
        )
        self.on_session_complete(session, finish_time - flow.submitted_time)

