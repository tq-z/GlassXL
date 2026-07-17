"""Analytical phase-aware compute, memory, and link backend."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_sim.core.resource import Reservation
from agentic_sim.core.task import Task, TaskType
from agentic_sim.hardware.topology import Topology
from agentic_sim.runtime.reference_policy import Placement


@dataclass(frozen=True)
class TaskExecution:
    task: Task
    reservation: Reservation
    byte_count: int = 0


class SimpleHardwareBackend:
    def __init__(self, topology: Topology) -> None:
        self.topology = topology
        self.executions: list[TaskExecution] = []

    def submit(self, task: Task, placement: Placement, now: float) -> TaskExecution:
        if task.task_type is TaskType.COMPUTE:
            if not isinstance(placement.resource_id, int):
                raise TypeError("compute placement must contain a device id")
            device = self.topology.devices[placement.resource_id]
            service_time = device.service_time(task.metadata["phase"], int(task.metadata["tokens"]))
            reservation = device.queue.reserve(now, service_time)
            execution = TaskExecution(task, reservation)
        elif task.task_type is TaskType.MEMORY:
            if not isinstance(placement.resource_id, int):
                raise TypeError("memory placement must contain a memory id")
            memory = self.topology.memories[placement.resource_id]
            byte_count = int(task.metadata["bytes"])
            reservation = memory.queue.reserve(now, memory.service_time(byte_count))
            execution = TaskExecution(task, reservation, byte_count)
        elif task.task_type is TaskType.TRANSFER:
            if not isinstance(placement.resource_id, tuple):
                raise TypeError("transfer placement must contain link endpoints")
            link = self.topology.get_link(*placement.resource_id)
            byte_count = int(task.metadata["bytes"])
            reservation = link.queue.reserve(now, link.service_time(byte_count))
            execution = TaskExecution(task, reservation, byte_count)
        else:
            raise ValueError("wait tasks are handled by the agent runtime, not hardware")
        self.executions.append(execution)
        return execution
