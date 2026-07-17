"""Directed latency-plus-bandwidth link."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_sim.core.resource import FCFSResource


@dataclass
class Link:
    src: int
    dst: int
    bandwidth_bytes_per_sec: float
    fixed_latency: float
    queue_policy: str = "fcfs"
    queue: FCFSResource = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.src == self.dst:
            raise ValueError("links must connect distinct endpoints")
        if self.bandwidth_bytes_per_sec <= 0.0 or self.fixed_latency < 0.0:
            raise ValueError("link bandwidth must be positive and latency non-negative")
        if self.queue_policy != "fcfs":
            raise ValueError("the simple backend supports only FCFS queues")
        self.queue = FCFSResource(f"link:{self.src}->{self.dst}")

    def service_time(self, byte_count: int) -> float:
        if byte_count < 0:
            raise ValueError("transfer byte count cannot be negative")
        return self.fixed_latency + byte_count / self.bandwidth_bytes_per_sec
