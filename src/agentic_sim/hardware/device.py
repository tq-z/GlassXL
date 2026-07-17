"""Generic phase-aware compute and memory resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_sim.core.resource import FCFSResource
from agentic_sim.core.task import ComputePhase


@dataclass
class Device:
    device_id: int
    device_class: str
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    fixed_latency: float
    queue_policy: str = "fcfs"
    queue: FCFSResource = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.prefill_tokens_per_second <= 0.0 or self.decode_tokens_per_second <= 0.0:
            raise ValueError("prefill and decode token rates must be positive")
        if self.fixed_latency < 0.0:
            raise ValueError("device latency must be non-negative")
        if self.queue_policy != "fcfs":
            raise ValueError("the simple backend supports only FCFS queues")
        self.queue = FCFSResource(f"compute:{self.device_id}")

    def service_time(self, phase: ComputePhase | str, tokens: int) -> float:
        if tokens < 0:
            raise ValueError("compute token count cannot be negative")
        compute_phase = phase if isinstance(phase, ComputePhase) else ComputePhase(phase)
        rate = (
            self.prefill_tokens_per_second
            if compute_phase is ComputePhase.PREFILL
            else self.decode_tokens_per_second
        )
        return self.fixed_latency + tokens / rate


@dataclass
class MemoryResource:
    memory_id: int
    capacity_bytes: int
    bandwidth_bytes_per_sec: float
    fixed_latency: float
    queue_policy: str = "fcfs"
    queue: FCFSResource = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity_bytes < 0 or self.bandwidth_bytes_per_sec <= 0.0 or self.fixed_latency < 0.0:
            raise ValueError("invalid memory capacity, bandwidth, or latency")
        if self.queue_policy != "fcfs":
            raise ValueError("the simple backend supports only FCFS queues")
        self.queue = FCFSResource(f"memory:{self.memory_id}")

    def service_time(self, byte_count: int) -> float:
        if byte_count < 0:
            raise ValueError("memory byte count cannot be negative")
        return self.fixed_latency + byte_count / self.bandwidth_bytes_per_sec
