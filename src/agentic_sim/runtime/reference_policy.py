"""Static FCFS reference execution policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agentic_sim.core.task import ComputePhase, Task, TaskType
from agentic_sim.hardware.hardware import HardwareState
from agentic_sim.workload.invocation import KVRetentionMode

ResourceIdentifier = int | tuple[int, int]


@dataclass(frozen=True)
class Placement:
    resource_id: ResourceIdentifier


class ExecutionPolicy(ABC):
    @abstractmethod
    def place_task(self, task: Task, hardware_state: HardwareState) -> Placement:
        """Select a resource without mutating hardware state."""


class StaticFCFSPolicy(ExecutionPolicy):
    def __init__(
        self,
        prefill_device: int,
        decode_device: int,
        retention: KVRetentionMode,
        memory_resource: int | None = None,
        far_memory_resource: int | None = None,
    ) -> None:
        self.prefill_device = prefill_device
        self.decode_device = decode_device
        self.retention = retention
        self.memory_resource = memory_resource
        self.far_memory_resource = far_memory_resource

    @property
    def selected_memory_resource(self) -> int | None:
        if self.retention is KVRetentionMode.MEMORY:
            return self.memory_resource
        if self.retention is KVRetentionMode.FAR_MEMORY:
            return self.far_memory_resource
        return None

    def place_task(self, task: Task, hardware_state: HardwareState) -> Placement:
        topology = hardware_state.topology
        if task.task_type is TaskType.COMPUTE:
            phase = task.metadata["phase"]
            compute_phase = phase if isinstance(phase, ComputePhase) else ComputePhase(phase)
            resource_id = (
                self.prefill_device
                if compute_phase is ComputePhase.PREFILL
                else self.decode_device
            )
            if resource_id not in topology.devices:
                raise KeyError(f"unknown compute device {resource_id}")
            return Placement(resource_id)
        if task.task_type is TaskType.MEMORY:
            resource_id = self.selected_memory_resource
            if resource_id is None or resource_id not in topology.memories:
                raise KeyError("no valid memory resource configured for KV retention")
            return Placement(resource_id)
        if task.task_type is TaskType.TRANSFER:
            direction = str(task.metadata["direction"])
            memory_id = self.selected_memory_resource
            if direction == "prefill_to_decode":
                endpoints = (self.prefill_device, self.decode_device)
            elif direction == "store":
                if memory_id is None:
                    raise KeyError("KV store requested without a memory resource")
                endpoints = (self.decode_device, memory_id)
            elif direction == "restore":
                if memory_id is None:
                    raise KeyError("KV restore requested without a memory resource")
                endpoints = (memory_id, self.prefill_device)
            else:
                raise ValueError(f"unknown transfer direction {direction!r}")
            topology.get_link(*endpoints)
            return Placement(endpoints)
        raise ValueError("wait tasks do not receive hardware placement")
