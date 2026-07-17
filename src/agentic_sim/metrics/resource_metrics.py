"""Resource utilization records."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResourceMetrics:
    resource_id: str
    resource_type: str
    busy_time_s: float
    utilization: float
    max_queue_length: int
    bytes_processed: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)
