"""Generic deterministic FCFS resource queue."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reservation:
    resource_id: str
    start_time: float
    finish_time: float
    queue_delay: float
    service_time: float


class FCFSResource:
    """A serial resource whose service reservations are made in submit order."""

    def __init__(self, resource_id: str) -> None:
        self.resource_id = resource_id
        self.available_time = 0.0
        self.reservations: list[Reservation] = []
        self.queue_samples: list[tuple[float, int]] = [(0.0, 0)]
        self.max_queue_length = 0

    def reserve(self, ready_time: float, service_time: float) -> Reservation:
        if ready_time < 0.0 or service_time < 0.0:
            raise ValueError("ready time and service time must be non-negative")
        start_time = max(ready_time, self.available_time)
        finish_time = start_time + service_time
        reservation = Reservation(
            resource_id=self.resource_id,
            start_time=start_time,
            finish_time=finish_time,
            queue_delay=start_time - ready_time,
            service_time=service_time,
        )
        self.reservations.append(reservation)
        self.available_time = finish_time
        queue_length = (
            1
            + sum(
                1
                for previous in self.reservations[:-1]
                if previous.start_time > ready_time
            )
            if start_time > ready_time
            else 0
        )
        self.max_queue_length = max(self.max_queue_length, queue_length)
        self.queue_samples.append((ready_time, queue_length))
        if start_time > ready_time:
            self.queue_samples.append((start_time, max(0, queue_length - 1)))
        return reservation

    def busy_time(self, horizon: float) -> float:
        return sum(
            max(0.0, min(reservation.finish_time, horizon) - reservation.start_time)
            for reservation in self.reservations
            if reservation.start_time < horizon
        )

    def utilization(self, horizon: float) -> float:
        return self.busy_time(horizon) / horizon if horizon > 0.0 else 0.0
