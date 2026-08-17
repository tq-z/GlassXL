"""Token-unit interconnect latency models and queued resources."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from agentic_sim.config.models import FFNAllToAllConfig, KVP2PConfig
from agentic_sim.core.resource import FCFSResource, Reservation
from agentic_sim.hardware.compute_models import CollectiveLatencyModel, P2PLatencyModel
from agentic_sim.hardware.device import Device, DeviceOperation, DeviceReservation


def _finite_nonnegative(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return result


def _positive(value: float, field_name: str) -> float:
    result = _finite_nonnegative(value, field_name)
    if result == 0.0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _nonnegative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True)
class StartupMinPackageCollectiveModel:
    """START_LAT + max(send_tokens, min_package_tokens) / throughput."""

    start_latency_s: float
    min_package_tokens: int
    throughput_tokens_per_second: float

    def __post_init__(self) -> None:
        _finite_nonnegative(self.start_latency_s, "start_latency_s")
        _nonnegative_integer(self.min_package_tokens, "min_package_tokens")
        _positive(self.throughput_tokens_per_second, "throughput_tokens_per_second")

    def latency_s(self, send_tokens: int, participants: int) -> float:
        _nonnegative_integer(send_tokens, "send_tokens")
        if isinstance(participants, bool) or not isinstance(participants, int) or participants <= 0:
            raise ValueError("participants must be a positive integer")
        return (
            self.start_latency_s
            + max(send_tokens, self.min_package_tokens) / self.throughput_tokens_per_second
        )


@dataclass(frozen=True)
class StartupThroughputP2PModel:
    """START_LAT + tokens / throughput, with no minimum-package floor."""

    start_latency_s: float
    throughput_tokens_per_second: float

    def __post_init__(self) -> None:
        _finite_nonnegative(self.start_latency_s, "start_latency_s")
        _positive(self.throughput_tokens_per_second, "throughput_tokens_per_second")

    def latency_s(self, tokens: int, src: int, dst: int) -> float:
        _nonnegative_integer(tokens, "tokens")
        _nonnegative_integer(src, "src")
        _nonnegative_integer(dst, "dst")
        return self.start_latency_s + tokens / self.throughput_tokens_per_second


@dataclass(frozen=True)
class InterconnectReservation:
    """One operation occupying a fabric queue and all endpoint device queues."""

    resource_reservation: Reservation
    endpoint_reservations: tuple[DeviceReservation, ...]

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


def reserve_synchronized(
    resource: FCFSResource,
    devices: Sequence[Device],
    operation: DeviceOperation,
    ready_time: float,
    service_time: float,
) -> InterconnectReservation:
    """Atomically reserve a logical resource and unique physical endpoints."""

    ready = _finite_nonnegative(ready_time, "ready_time")
    service = _finite_nonnegative(service_time, "service_time")
    if not devices:
        raise ValueError("at least one endpoint device is required")
    if len({device.device_id for device in devices}) != len(devices):
        raise ValueError("endpoint devices must be unique")

    common_start = max(
        ready,
        resource.available_time,
        *(device.queue.available_time for device in devices),
    )
    resource_reservation = resource.reserve(
        ready,
        service,
        earliest_start=common_start,
    )
    endpoint_reservations = tuple(
        device.reserve_external(
            operation,
            ready,
            service,
            earliest_start=common_start,
        )
        for device in devices
    )
    return InterconnectReservation(resource_reservation, endpoint_reservations)


@dataclass
class FFNAllToAllInterconnect:
    """One globally contended V1 FFN dispatch/combine resource."""

    model: CollectiveLatencyModel
    queue: FCFSResource = field(
        default_factory=lambda: FCFSResource("interconnect:ffn_all_to_all"),
        init=False,
        repr=False,
    )

    @classmethod
    def from_config(cls, config: FFNAllToAllConfig) -> FFNAllToAllInterconnect:
        return cls(
            StartupMinPackageCollectiveModel(
                start_latency_s=config.start_latency_us / 1_000_000.0,
                min_package_tokens=config.min_package_tokens,
                throughput_tokens_per_second=config.throughput_tokens_per_second,
            )
        )

    def submit(
        self,
        send_tokens: int,
        participants: Sequence[Device],
        ready_time: float,
    ) -> InterconnectReservation:
        service_time = self.model.latency_s(send_tokens, len(participants))
        return reserve_synchronized(
            self.queue,
            participants,
            DeviceOperation.FFN_ALL_TO_ALL,
            ready_time,
            service_time,
        )


@dataclass
class KVP2PInterconnect:
    """One globally contended V1 KV-migration resource."""

    model: P2PLatencyModel
    queue: FCFSResource = field(
        default_factory=lambda: FCFSResource("interconnect:kv_p2p"),
        init=False,
        repr=False,
    )

    @classmethod
    def from_config(cls, config: KVP2PConfig) -> KVP2PInterconnect:
        return cls(
            StartupThroughputP2PModel(
                start_latency_s=config.start_latency_us / 1_000_000.0,
                throughput_tokens_per_second=config.throughput_tokens_per_second,
            )
        )

    def submit(
        self,
        tokens: int,
        src: Device,
        dst: Device,
        ready_time: float,
    ) -> InterconnectReservation:
        service_time = self.model.latency_s(tokens, src.device_id, dst.device_id)
        endpoints = (src,) if src.device_id == dst.device_id else (src, dst)
        return reserve_synchronized(
            self.queue,
            endpoints,
            DeviceOperation.KV_P2P,
            ready_time,
            service_time,
        )
