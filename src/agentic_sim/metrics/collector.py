"""In-memory metrics collection for the coarse P-D simulator."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from agentic_sim.hardware.simple_backend import SimpleHardwareBackend, TaskExecution
from agentic_sim.metrics.batching_metrics import DecodeIterationMetric, IngressFlowMetric, PrefillBatchMetric
from agentic_sim.metrics.request_metrics import AgentMetrics, InvocationMetrics
from agentic_sim.metrics.resource_metrics import ResourceMetrics
from agentic_sim.workload.agent import Agent, AgentState
from agentic_sim.workload.invocation import Invocation


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class MetricsSnapshot:
    summary: dict[str, int | float]
    agents: list[AgentMetrics]
    invocations: list[InvocationMetrics]
    resources: list[ResourceMetrics]
    prefill_batches: list[PrefillBatchMetric]
    ingress_flows: list[IngressFlowMetric]
    decode_iterations: list[DecodeIterationMetric]


class MetricsCollector:
    def __init__(self) -> None:
        self.invocations: dict[int, InvocationMetrics] = {}
        self.prefill_batches: list[PrefillBatchMetric] = []
        self.ingress_flows: list[IngressFlowMetric] = []
        self.decode_iterations: list[DecodeIterationMetric] = []
        self._ingress_submissions: dict[int, tuple[int, int]] = {}

        self._active_count = 0
        self._tool_waiting_count = 0
        self._kv_waiting_count = 0
        self._last_state_time = 0.0
        self._active_area = 0.0
        self._tool_waiting_area = 0.0
        self._kv_waiting_area = 0.0

        self._prefill_queue = 0
        self._prefill_queue_area = 0.0
        self._prefill_queue_last_time = 0.0
        self.max_prefill_queue = 0
        self._decode_queue = 0
        self._decode_active = 0
        self._decode_queue_area = 0.0
        self._decode_active_area = 0.0
        self._decode_last_time = 0.0
        self.max_decode_queue = 0
        self._ingress_active = 0
        self._ingress_bandwidth = 0.0
        self._ingress_active_area = 0.0
        self._ingress_bandwidth_area = 0.0
        self._ingress_last_time = 0.0
        self.peak_ingress_active = 0

    @staticmethod
    def _state_counts(state: AgentState) -> tuple[int, int, int]:
        return (
            int(state in {AgentState.READY, AgentState.RUNNING}),
            int(state is AgentState.TOOL_WAIT),
            int(state is AgentState.KV_WAIT),
        )

    def state_transition(self, time: float, old: AgentState, new: AgentState) -> None:
        elapsed = time - self._last_state_time
        self._active_area += elapsed * self._active_count
        self._tool_waiting_area += elapsed * self._tool_waiting_count
        self._kv_waiting_area += elapsed * self._kv_waiting_count
        self._last_state_time = time
        old_active, old_tool, old_kv = self._state_counts(old)
        new_active, new_tool, new_kv = self._state_counts(new)
        self._active_count += new_active - old_active
        self._tool_waiting_count += new_tool - old_tool
        self._kv_waiting_count += new_kv - old_kv

    def invocation_ready(self, invocation: Invocation) -> None:
        self.invocations[invocation.invocation_id] = InvocationMetrics(
            invocation_id=invocation.invocation_id,
            agent_id=invocation.agent_id,
            turn_index=invocation.turn_index,
            ready_time_s=invocation.ready_time,
        )

    def invocation_prefill_queued(self, invocation_id: int, time: float) -> None:
        self.invocations[invocation_id].prefill_queued_time_s = time

    def invocation_prefill_started(self, invocation_id: int, time: float, queue_delay: float, service_time: float) -> None:
        metric = self.invocations[invocation_id]
        metric.prefill_start_time_s = time
        metric.prefill_queueing_time_s += queue_delay
        metric.queueing_time_s += queue_delay
        metric.service_time_s += service_time

    def invocation_prefill_finished(self, invocation_id: int, time: float) -> None:
        self.invocations[invocation_id].prefill_finish_time_s = time

    def invocation_ingress_started(self, invocation_id: int, time: float) -> None:
        self.invocations[invocation_id].ingress_submit_time_s = time

    def invocation_ingress_finished(self, invocation_id: int, time: float, duration: float) -> None:
        metric = self.invocations[invocation_id]
        metric.ingress_finish_time_s = time
        metric.service_time_s += duration

    def invocation_decode_queued(self, invocation_id: int, time: float) -> None:
        self.invocations[invocation_id].decode_queued_time_s = time

    def invocation_decode_admitted(self, invocation_id: int, time: float, queue_delay: float) -> None:
        metric = self.invocations[invocation_id]
        metric.decode_start_time_s = time
        metric.decode_queueing_time_s += queue_delay
        metric.queueing_time_s += queue_delay

    def invocation_decode_token(self, invocation_id: int, time: float, service_time: float) -> None:
        metric = self.invocations[invocation_id]
        metric.token_times_s.append(time)
        if metric.first_token_time_s is None:
            metric.first_token_time_s = time
        metric.service_time_s += service_time

    def task_reserved(self, execution: TaskExecution) -> None:
        if execution.task.metadata.get("background", False):
            return
        metric = self.invocations[execution.task.invocation_id]
        metric.queueing_time_s += execution.reservation.queue_delay
        metric.service_time_s += execution.reservation.service_time

    def invocation_finished(self, invocation_id: int, time: float) -> None:
        self.invocations[invocation_id].finish_time_s = time

    def prefill_queue_changed(self, time: float, queue_length: int) -> None:
        self._prefill_queue_area += (time - self._prefill_queue_last_time) * self._prefill_queue
        self._prefill_queue_last_time = time
        self._prefill_queue = queue_length
        self.max_prefill_queue = max(self.max_prefill_queue, queue_length)

    def prefill_batch_started(self, metric: PrefillBatchMetric) -> None:
        self.prefill_batches.append(metric)

    def decode_state_changed(self, time: float, waiting: int, active: int) -> None:
        elapsed = time - self._decode_last_time
        self._decode_queue_area += elapsed * self._decode_queue
        self._decode_active_area += elapsed * self._decode_active
        self._decode_last_time = time
        self._decode_queue = waiting
        self._decode_active = active
        self.max_decode_queue = max(self.max_decode_queue, waiting)

    def decode_iteration_started(self, metric: DecodeIterationMetric) -> None:
        self.decode_iterations.append(metric)

    def ingress_active_changed(self, time: float, active: int, aggregate_bandwidth: float) -> None:
        elapsed = time - self._ingress_last_time
        self._ingress_active_area += elapsed * self._ingress_active
        self._ingress_bandwidth_area += elapsed * self._ingress_bandwidth
        self._ingress_last_time = time
        self._ingress_active = active
        self._ingress_bandwidth = aggregate_bandwidth
        self.peak_ingress_active = max(self.peak_ingress_active, active)

    def ingress_flow_submitted(self, flow_id: int, session_id: int, byte_count: int) -> None:
        self._ingress_submissions[flow_id] = (session_id, byte_count)

    def ingress_flow_finished(self, metric: IngressFlowMetric) -> None:
        self.ingress_flows.append(metric)

    def snapshot(self, agents: list[Agent], backend: SimpleHardwareBackend, horizon: float) -> MetricsSnapshot:
        self._advance_areas(horizon)
        invocation_metrics = list(self.invocations.values())
        prefill_time_by_agent: dict[int, float] = {}
        decode_time_by_agent: dict[int, float] = {}
        for invocation in invocation_metrics:
            prefill_time_by_agent[invocation.agent_id] = (
                prefill_time_by_agent.get(invocation.agent_id, 0.0) + invocation.prefill_time_s
            )
            decode_time_by_agent[invocation.agent_id] = (
                decode_time_by_agent.get(invocation.agent_id, 0.0) + invocation.decode_time_s
            )
        agent_metrics = [
            AgentMetrics.from_agent(
                agent,
                prefill_time_by_agent.get(agent.agent_id, 0.0),
                decode_time_by_agent.get(agent.agent_id, 0.0),
            )
            for agent in agents
            if agent.state is AgentState.FINISHED
        ]
        completed_invocations = [item for item in invocation_metrics if item.finish_time_s is not None]
        resources = self._resource_metrics(backend, horizon)
        latencies_ms = [item.total_latency_s * 1_000.0 for item in agent_metrics]
        compute = [item for item in resources if item.resource_type == "compute"]
        memory = [item for item in resources if item.resource_type == "memory"]
        links = [item for item in resources if item.resource_type == "link"]
        offered = sum(agent.arrival_time <= horizon for agent in agents)
        ttft_ms = [item.ttft_s * 1_000.0 for item in completed_invocations if item.ttft_s is not None]
        inter_token_ms = [delta * 1_000.0 for item in completed_invocations for delta in item.inter_token_times_s]
        prefill_queue_ms = [item.prefill_queueing_time_s * 1_000.0 for item in invocation_metrics]
        decode_queue_ms = [item.decode_queueing_time_s * 1_000.0 for item in invocation_metrics]
        emitted_tokens = sum(item.emitted_tokens for item in self.decode_iterations if item.finish_time_s <= horizon)
        total_ingress_bytes = sum(item.byte_count for item in self.ingress_flows if item.finish_time_s <= horizon)
        decode_session_ms = [
            (item.finish_time_s - item.decode_start_time_s) * 1_000.0
            for item in completed_invocations
            if item.decode_start_time_s is not None and item.finish_time_s is not None
        ]

        summary: dict[str, int | float] = {
            "simulation_time_s": horizon,
            "offered_agents": offered,
            "completed_agents": len(agent_metrics),
            "agent_throughput_per_s": len(agent_metrics) / horizon if horizon > 0.0 else 0.0,
            "invocation_throughput_per_s": len(completed_invocations) / horizon if horizon > 0.0 else 0.0,
            "mean_latency_ms": fmean(latencies_ms) if latencies_ms else 0.0,
            "p50_latency_ms": _percentile(latencies_ms, 0.50),
            "p95_latency_ms": _percentile(latencies_ms, 0.95),
            "p99_latency_ms": _percentile(latencies_ms, 0.99),
            "mean_queueing_ms": fmean(item.queueing_time_s for item in agent_metrics) * 1_000.0 if agent_metrics else 0.0,
            "mean_tool_wait_ms": fmean(item.tool_wait_time_s for item in agent_metrics) * 1_000.0 if agent_metrics else 0.0,
            "mean_agent_prefill_time_ms": fmean(item.prefill_time_s for item in agent_metrics) * 1_000.0 if agent_metrics else 0.0,
            "mean_agent_decode_time_ms": fmean(item.decode_time_s for item in agent_metrics) * 1_000.0 if agent_metrics else 0.0,
            "compute_utilization": fmean(item.utilization for item in compute) if compute else 0.0,
            "memory_utilization": fmean(item.utilization for item in memory) if memory else 0.0,
            "link_utilization": fmean(item.utilization for item in links) if links else 0.0,
            "max_compute_queue": max(max((item.max_queue_length for item in compute), default=0), self.max_prefill_queue, self.max_decode_queue),
            "max_memory_queue": max((item.max_queue_length for item in memory), default=0),
            "max_link_queue": max((item.max_queue_length for item in links), default=0),
            "average_active_agents": self._active_area / horizon if horizon > 0.0 else 0.0,
            "average_tool_wait_agents": self._tool_waiting_area / horizon if horizon > 0.0 else 0.0,
            "average_kv_wait_agents": self._kv_waiting_area / horizon if horizon > 0.0 else 0.0,
            "offered_load_per_s": offered / horizon if horizon > 0.0 else 0.0,
            "completed_load_per_s": len(agent_metrics) / horizon if horizon > 0.0 else 0.0,
            "prefill_batches": len(self.prefill_batches),
            "mean_prefill_batch_requests": fmean(item.request_count for item in self.prefill_batches) if self.prefill_batches else 0.0,
            "mean_prefill_batch_tokens": fmean(item.total_tokens for item in self.prefill_batches) if self.prefill_batches else 0.0,
            "mean_prefill_batch_service_ms": fmean((item.finish_time_s - item.start_time_s) * 1_000.0 for item in self.prefill_batches) if self.prefill_batches else 0.0,
            "mean_prefill_queueing_ms": fmean(prefill_queue_ms) if prefill_queue_ms else 0.0,
            "max_prefill_queue": self.max_prefill_queue,
            "average_prefill_queue": self._prefill_queue_area / horizon if horizon > 0.0 else 0.0,
            "ingress_flows": len(self.ingress_flows),
            "total_ingress_bytes": total_ingress_bytes,
            "mean_ingress_completion_ms": fmean((item.finish_time_s - item.submitted_time_s) * 1_000.0 for item in self.ingress_flows) if self.ingress_flows else 0.0,
            "mean_ingress_achieved_bandwidth_bytes_per_s": fmean(item.achieved_bandwidth_bytes_per_s for item in self.ingress_flows) if self.ingress_flows else 0.0,
            "average_aggregate_ingress_bandwidth_bytes_per_s": self._ingress_bandwidth_area / horizon if horizon > 0.0 else 0.0,
            "peak_concurrent_ingress_flows": self.peak_ingress_active,
            "decode_iterations": len(self.decode_iterations),
            "decode_tokens_emitted": emitted_tokens,
            "mean_decode_batch_size": fmean(item.active_sessions for item in self.decode_iterations) if self.decode_iterations else 0.0,
            "mean_decode_step_ms": fmean((item.finish_time_s - item.start_time_s) * 1_000.0 for item in self.decode_iterations) if self.decode_iterations else 0.0,
            "decode_token_throughput_per_s": emitted_tokens / horizon if horizon > 0.0 else 0.0,
            "mean_decode_session_completion_ms": fmean(decode_session_ms) if decode_session_ms else 0.0,
            "mean_ttft_ms": fmean(ttft_ms) if ttft_ms else 0.0,
            "p50_ttft_ms": _percentile(ttft_ms, 0.50),
            "p95_ttft_ms": _percentile(ttft_ms, 0.95),
            "p99_ttft_ms": _percentile(ttft_ms, 0.99),
            "mean_inter_token_ms": fmean(inter_token_ms) if inter_token_ms else 0.0,
            "p95_inter_token_ms": _percentile(inter_token_ms, 0.95),
            "p99_inter_token_ms": _percentile(inter_token_ms, 0.99),
            "mean_decode_queueing_ms": fmean(decode_queue_ms) if decode_queue_ms else 0.0,
            "max_decode_queue": self.max_decode_queue,
            "mean_ingress_wait_ms": fmean((item.data_start_time_s - item.submitted_time_s) * 1_000.0 for item in self.ingress_flows) if self.ingress_flows else 0.0,
            "average_decode_queue": self._decode_queue_area / horizon if horizon > 0.0 else 0.0,
            "average_decode_active_sessions": self._decode_active_area / horizon if horizon > 0.0 else 0.0,
            "average_active_ingress_flows": self._ingress_active_area / horizon if horizon > 0.0 else 0.0,
        }
        return MetricsSnapshot(
            summary,
            agent_metrics,
            invocation_metrics,
            resources,
            list(self.prefill_batches),
            list(self.ingress_flows),
            list(self.decode_iterations),
        )

    def _advance_areas(self, horizon: float) -> None:
        if horizon > self._last_state_time:
            elapsed = horizon - self._last_state_time
            self._active_area += elapsed * self._active_count
            self._tool_waiting_area += elapsed * self._tool_waiting_count
            self._kv_waiting_area += elapsed * self._kv_waiting_count
            self._last_state_time = horizon
        if horizon > self._prefill_queue_last_time:
            self._prefill_queue_area += (horizon - self._prefill_queue_last_time) * self._prefill_queue
            self._prefill_queue_last_time = horizon
        if horizon > self._decode_last_time:
            elapsed = horizon - self._decode_last_time
            self._decode_queue_area += elapsed * self._decode_queue
            self._decode_active_area += elapsed * self._decode_active
            self._decode_last_time = horizon
        if horizon > self._ingress_last_time:
            elapsed = horizon - self._ingress_last_time
            self._ingress_active_area += elapsed * self._ingress_active
            self._ingress_bandwidth_area += elapsed * self._ingress_bandwidth
            self._ingress_last_time = horizon

    @staticmethod
    def _resource_metrics(backend: SimpleHardwareBackend, horizon: float) -> list[ResourceMetrics]:
        result: list[ResourceMetrics] = []
        completed_bytes: dict[str, int] = {}
        for execution in backend.executions:
            if execution.reservation.finish_time <= horizon:
                resource_id = execution.reservation.resource_id
                completed_bytes[resource_id] = completed_bytes.get(resource_id, 0) + execution.byte_count
        for device in backend.topology.devices.values():
            result.append(ResourceMetrics(device.queue.resource_id, "compute", device.queue.busy_time(horizon), device.queue.utilization(horizon), device.queue.max_queue_length, 0))
        for memory_resource in backend.topology.memories.values():
            result.append(ResourceMetrics(memory_resource.queue.resource_id, "memory", memory_resource.queue.busy_time(horizon), memory_resource.queue.utilization(horizon), memory_resource.queue.max_queue_length, completed_bytes.get(memory_resource.queue.resource_id, 0)))
        for link in backend.topology.links.values():
            result.append(ResourceMetrics(link.queue.resource_id, "link", link.queue.busy_time(horizon), link.queue.utilization(horizon), link.queue.max_queue_length, completed_bytes.get(link.queue.resource_id, 0)))
        return result
