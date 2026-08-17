"""Top-level assembly for the coarse P-D simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_sim.config import ConfigurationError, validate_config
from agentic_sim.core.event_engine import EventEngine
from agentic_sim.hardware.simple_backend import SimpleHardwareBackend
from agentic_sim.hardware.topology import Topology
from agentic_sim.metrics.batching_metrics import DecodeIterationMetric, IngressFlowMetric, PrefillBatchMetric
from agentic_sim.metrics.collector import MetricsCollector
from agentic_sim.metrics.request_metrics import AgentMetrics, InvocationMetrics
from agentic_sim.metrics.resource_metrics import ResourceMetrics
from agentic_sim.runtime.reference_policy import StaticFCFSPolicy
from agentic_sim.runtime.runtime import AgenticRuntime
from agentic_sim.runtime.task_builder import TaskBuilder
from agentic_sim.workload.agent import Agent
from agentic_sim.workload.invocation import KVRetentionMode
from agentic_sim.workload.workload_generator import WorkloadGenerator


@dataclass(frozen=True)
class SimulationResult:
    summary: dict[str, int | float]
    agents: list[AgentMetrics]
    invocations: list[InvocationMetrics]
    resources: list[ResourceMetrics]
    all_agents: list[Agent]
    prefill_batches: list[PrefillBatchMetric]
    ingress_flows: list[IngressFlowMetric]
    decode_iterations: list[DecodeIterationMetric]


class Simulation:
    def __init__(self, config: dict[str, Any]) -> None:
        validate_config(config)
        self.config = config

    def run(self) -> SimulationResult:
        workload = WorkloadGenerator(self.config["workload"], int(self.config["seed"]))
        agents = workload.generate()
        hardware_config = self.config["hardware"]
        topology = Topology.from_config(hardware_config)
        backend = SimpleHardwareBackend(topology)
        policy_config = self.config["policy"]
        kv_config = self.config["invocation_model"]["kv"]
        retention = KVRetentionMode(str(kv_config["retention"]))
        policy = StaticFCFSPolicy(
            prefill_device=int(policy_config["prefill_device"]),
            decode_device=int(policy_config["decode_device"]),
            retention=retention,
            memory_resource=int(policy_config["memory_resource"]) if "memory_resource" in policy_config else None,
            far_memory_resource=int(policy_config["far_memory_resource"]) if "far_memory_resource" in policy_config else None,
        )
        self._validate_resources(topology, policy, shared_ingress="kv_ingress" in hardware_config)
        task_builder = TaskBuilder(self.config["invocation_model"])
        configured_end = self.config.get("simulation_end_s")
        end_time = float(configured_end) if configured_end is not None else None
        engine = EventEngine(end_time=end_time)
        metrics = MetricsCollector()
        runtime = AgenticRuntime(
            engine,
            agents,
            backend,
            policy,
            task_builder,
            metrics,
            kv_ingress_config=hardware_config.get("kv_ingress"),
        )
        runtime.schedule_arrivals()
        engine.run()
        snapshot = metrics.snapshot(agents, backend, engine.now)
        return SimulationResult(
            snapshot.summary,
            snapshot.agents,
            snapshot.invocations,
            snapshot.resources,
            agents,
            snapshot.prefill_batches,
            snapshot.ingress_flows,
            snapshot.decode_iterations,
        )

    @staticmethod
    def _validate_resources(topology: Topology, policy: StaticFCFSPolicy, shared_ingress: bool = False) -> None:
        for device_id, role in ((policy.prefill_device, "prefill"), (policy.decode_device, "decode")):
            if device_id not in topology.devices:
                raise ConfigurationError(f"unknown {role} device {device_id}")
        if topology.devices[policy.prefill_device].prefill is None:
            raise ConfigurationError("selected prefill device has no prefill settings")
        if topology.devices[policy.decode_device].decode is None:
            raise ConfigurationError("selected decode device has no decode settings")
        if not shared_ingress:
            try:
                topology.get_link(policy.prefill_device, policy.decode_device)
            except KeyError as error:
                raise ConfigurationError("a direct prefill-to-decode link is required") from error
        memory_id = policy.selected_memory_resource
        if memory_id is None:
            return
        if memory_id not in topology.memories:
            raise ConfigurationError(f"unknown selected KV memory resource {memory_id}")
        try:
            topology.get_link(policy.decode_device, memory_id)
            topology.get_link(memory_id, policy.prefill_device)
        except KeyError as error:
            raise ConfigurationError("retained KV requires decode-to-memory and memory-to-prefill links") from error
