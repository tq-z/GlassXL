"""Generic phase-aware compute and memory resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_sim.core.resource import FCFSResource
from agentic_sim.core.task import ComputePhase


@dataclass(frozen=True)
class PrefillSettings:
    """Whole-prompt batching limits and rate envelope."""

    single_request_tokens_per_second: float
    max_system_tokens_per_second: float
    max_batch_requests: int
    max_batch_tokens: int
    layout: str = "packed"
    batch_wait: float = 0.0

    def __post_init__(self) -> None:
        if self.single_request_tokens_per_second <= 0.0 or self.max_system_tokens_per_second <= 0.0:
            raise ValueError("prefill token rates must be positive")
        if self.max_batch_requests <= 0 or self.max_batch_tokens <= 0:
            raise ValueError("prefill batch limits must be positive")
        if self.layout != "packed" or self.batch_wait != 0.0:
            raise ValueError("Phase 1.5 supports packed prefill batches with no batch wait")


@dataclass(frozen=True)
class DecodeSettings:
    """Continuous decode batching limits and rate envelope."""

    single_session_tokens_per_second: float
    max_system_tokens_per_second: float
    max_active_sessions: int

    def __post_init__(self) -> None:
        if self.single_session_tokens_per_second <= 0.0 or self.max_system_tokens_per_second <= 0.0:
            raise ValueError("decode token rates must be positive")
        if self.max_active_sessions <= 0:
            raise ValueError("decode.max_active_sessions must be positive")


@dataclass
class Device:
    device_id: int
    device_class: str
    fixed_latency: float
    prefill: PrefillSettings | None = None
    decode: DecodeSettings | None = None
    queue_policy: str = "fcfs"
    queue: FCFSResource = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.prefill is None and self.decode is None:
            raise ValueError("compute device must provide prefill and/or decode settings")
        if self.fixed_latency < 0.0:
            raise ValueError("device latency must be non-negative")
        if self.queue_policy != "fcfs":
            raise ValueError("the simple backend supports only FCFS queues")
        self.queue = FCFSResource(f"compute:{self.device_id}")

    @property
    def prefill_tokens_per_second(self) -> float:
        if self.prefill is None:
            raise ValueError(f"device {self.device_id} has no prefill capability")
        return self.prefill.single_request_tokens_per_second

    @property
    def decode_tokens_per_second(self) -> float:
        if self.decode is None:
            raise ValueError(f"device {self.device_id} has no decode capability")
        return self.decode.single_session_tokens_per_second

    def service_time(self, phase: ComputePhase | str, tokens: int) -> float:
        """Legacy single-request service equation used by generic task submission."""

        if tokens < 0:
            raise ValueError("compute token count cannot be negative")
        compute_phase = phase if isinstance(phase, ComputePhase) else ComputePhase(phase)
        rate = self.prefill_tokens_per_second if compute_phase is ComputePhase.PREFILL else self.decode_tokens_per_second
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
