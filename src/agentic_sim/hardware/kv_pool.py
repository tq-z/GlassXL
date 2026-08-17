"""Unbounded V1 KV-pool latency and shared queue."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agentic_sim.config.models import KVPoolConfig
from agentic_sim.core.resource import FCFSResource, Reservation
from agentic_sim.hardware.device import Device, DeviceOperation, DeviceReservation
from agentic_sim.hardware.interconnect import (
    _finite_nonnegative,
    _nonnegative_integer,
    _positive,
    reserve_synchronized,
)


class KVPoolOperation(StrEnum):
    LOAD = "load"
    OFFLOAD = "offload"


@dataclass(frozen=True)
class SymmetricKVPoolLatencyModel:
    """Pool startup plus token payload divided by one symmetric rate."""

    start_latency_s: float
    tokens_per_second: float

    def __post_init__(self) -> None:
        _finite_nonnegative(self.start_latency_s, "start_latency_s")
        _positive(self.tokens_per_second, "tokens_per_second")

    def latency_s(self, tokens: int) -> float:
        _nonnegative_integer(tokens, "tokens")
        return self.start_latency_s + tokens / self.tokens_per_second


@dataclass(frozen=True)
class KVPoolReservation:
    """One load or offload occupying the pool and one physical device."""

    operation: KVPoolOperation
    resource_reservation: Reservation
    endpoint_reservation: DeviceReservation

    @property
    def start_time(self) -> float:
        return self.resource_reservation.start_time

    @property
    def finish_time(self) -> float:
        return self.resource_reservation.finish_time

    @property
    def queue_delay(self) -> float:
        return self.resource_reservation.queue_delay

    @property
    def service_time(self) -> float:
        return self.resource_reservation.service_time


@dataclass
class KVPool:
    """The V1 unbounded backing pool with one shared load/offload queue."""

    model: SymmetricKVPoolLatencyModel
    capacity_tokens: None = None
    queue: FCFSResource = field(
        default_factory=lambda: FCFSResource("kv_pool"),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.capacity_tokens is not None:
            raise ValueError("V1 KV-pool capacity must be unbounded (None)")

    @classmethod
    def from_config(cls, config: KVPoolConfig) -> KVPool:
        if config.capacity_tokens is not None:
            raise ValueError("V1 KV-pool capacity must be unbounded (None)")
        if config.load_tokens_per_second != config.offload_tokens_per_second:
            raise ValueError("V1 KV-pool load and offload rates must be equal")
        return cls(
            model=SymmetricKVPoolLatencyModel(
                start_latency_s=config.start_latency_us / 1_000_000.0,
                tokens_per_second=config.tokens_per_second,
            )
        )

    def can_fit(self, tokens: int) -> bool:
        """Return the frozen V1 capacity policy: every valid payload fits."""

        _nonnegative_integer(tokens, "tokens")
        return True

    def submit_load(
        self,
        tokens: int,
        device: Device,
        ready_time: float,
    ) -> KVPoolReservation:
        return self._submit(KVPoolOperation.LOAD, tokens, device, ready_time)

    def submit_offload(
        self,
        tokens: int,
        device: Device,
        ready_time: float,
    ) -> KVPoolReservation:
        return self._submit(KVPoolOperation.OFFLOAD, tokens, device, ready_time)

    def _submit(
        self,
        operation: KVPoolOperation,
        tokens: int,
        device: Device,
        ready_time: float,
    ) -> KVPoolReservation:
        service_time = self.model.latency_s(tokens)
        device_operation = (
            DeviceOperation.KV_POOL_LOAD
            if operation is KVPoolOperation.LOAD
            else DeviceOperation.KV_POOL_OFFLOAD
        )
        synchronized = reserve_synchronized(
            self.queue,
            (device,),
            device_operation,
            ready_time,
            service_time,
        )
        return KVPoolReservation(
            operation,
            synchronized.resource_reservation,
            synchronized.endpoint_reservations[0],
        )
