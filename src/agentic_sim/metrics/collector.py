"""In-memory metrics collection for the coarse P-D simulator."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from agentic_sim.hardware.simple_backend import SimpleHardwareBackend, TaskExecution
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


class MetricsCollector:
    def __init__(self) -> None:
        self.invocations: dict[int, InvocationMetrics] = {}
        self._active_count = 0
        self._tool_waiting_count = 0
        self._kv_waiting_count = 0
        self._last_state_time = 0.0
        self._active_area = 0.0
        self._tool_waiting_area = 0.0
        self._kv_waiting_area = 0.0

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

    def task_reserved(self, execution: TaskExecution) -> None:
        if execution.task.metadata.get("background", False):
            return
        metric = self.invocations[execution.task.invocation_id]
        metric.queueing_time_s += execution.reservation.queue_delay
        metric.service_time_s += execution.reservation.service_time

    def invocation_finished(self, invocation_id: int, time: float) -> None:
        self.invocations[invocation_id].finish_time_s = time

    def snapshot(
        self,
        agents: list[Agent],
        backend: SimpleHardwareBackend,
        horizon: float,
    ) -> MetricsSnapshot:
        if horizon > self._last_state_time:
            elapsed = horizon - self._last_state_time
            self._active_area += elapsed * self._active_count
            self._tool_waiting_area += elapsed * self._tool_waiting_count
            self._kv_waiting_area += elapsed * self._kv_waiting_count
            self._last_state_time = horizon

        agent_metrics = [
            AgentMetrics.from_agent(agent)
            for agent in agents
            if agent.state is AgentState.FINISHED
        ]
        invocation_metrics = list(self.invocations.values())
        completed_invocations = [item for item in invocation_metrics if item.finish_time_s is not None]
        resources = self._resource_metrics(backend, horizon)
        latencies_ms = [item.total_latency_s * 1_000.0 for item in agent_metrics]
        compute = [item for item in resources if item.resource_type == "compute"]
        memory = [item for item in resources if item.resource_type == "memory"]
        links = [item for item in resources if item.resource_type == "link"]
        offered = sum(agent.arrival_time <= horizon for agent in agents)

        summary: dict[str, int | float] = {
            "simulation_time_s": horizon,
            "offered_agents": offered,
            "completed_agents": len(agent_metrics),
            "agent_throughput_per_s": len(agent_metrics) / horizon if horizon > 0.0 else 0.0,
            "invocation_throughput_per_s": (
                len(completed_invocations) / horizon if horizon > 0.0 else 0.0
            ),
            "mean_latency_ms": fmean(latencies_ms) if latencies_ms else 0.0,
            "p50_latency_ms": _percentile(latencies_ms, 0.50),
            "p95_latency_ms": _percentile(latencies_ms, 0.95),
            "p99_latency_ms": _percentile(latencies_ms, 0.99),
            "mean_queueing_ms": (
                fmean(item.queueing_time_s for item in agent_metrics) * 1_000.0
                if agent_metrics
                else 0.0
            ),
            "mean_tool_wait_ms": (
                fmean(item.tool_wait_time_s for item in agent_metrics) * 1_000.0
                if agent_metrics
                else 0.0
            ),
            "compute_utilization": fmean(item.utilization for item in compute) if compute else 0.0,
            "memory_utilization": fmean(item.utilization for item in memory) if memory else 0.0,
            "link_utilization": fmean(item.utilization for item in links) if links else 0.0,
            "max_compute_queue": max((item.max_queue_length for item in compute), default=0),
            "max_memory_queue": max((item.max_queue_length for item in memory), default=0),
            "max_link_queue": max((item.max_queue_length for item in links), default=0),
            "average_active_agents": self._active_area / horizon if horizon > 0.0 else 0.0,
            "average_tool_wait_agents": (
                self._tool_waiting_area / horizon if horizon > 0.0 else 0.0
            ),
            "average_kv_wait_agents": self._kv_waiting_area / horizon if horizon > 0.0 else 0.0,
            "offered_load_per_s": offered / horizon if horizon > 0.0 else 0.0,
            "completed_load_per_s": len(agent_metrics) / horizon if horizon > 0.0 else 0.0,
        }
        return MetricsSnapshot(summary, agent_metrics, invocation_metrics, resources)

    @staticmethod
    def _resource_metrics(
        backend: SimpleHardwareBackend,
        horizon: float,
    ) -> list[ResourceMetrics]:
        result: list[ResourceMetrics] = []
        completed_bytes: dict[str, int] = {}
        for execution in backend.executions:
            if execution.reservation.finish_time <= horizon:
                resource_id = execution.reservation.resource_id
                completed_bytes[resource_id] = completed_bytes.get(resource_id, 0) + execution.byte_count
        for device in backend.topology.devices.values():
            result.append(
                ResourceMetrics(
                    device.queue.resource_id,
                    "compute",
                    device.queue.busy_time(horizon),
                    device.queue.utilization(horizon),
                    device.queue.max_queue_length,
                    0,
                )
            )
        for memory_resource in backend.topology.memories.values():
            result.append(
                ResourceMetrics(
                    memory_resource.queue.resource_id,
                    "memory",
                    memory_resource.queue.busy_time(horizon),
                    memory_resource.queue.utilization(horizon),
                    memory_resource.queue.max_queue_length,
                    completed_bytes.get(memory_resource.queue.resource_id, 0),
                )
            )
        for link in backend.topology.links.values():
            result.append(
                ResourceMetrics(
                    link.queue.resource_id,
                    "link",
                    link.queue.busy_time(horizon),
                    link.queue.utilization(horizon),
                    link.queue.max_queue_length,
                    completed_bytes.get(link.queue.resource_id, 0),
                )
            )
        return result
