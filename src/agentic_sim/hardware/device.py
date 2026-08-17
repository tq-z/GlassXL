"""Capability-bearing physical device with one shared FCFS queue."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from agentic_sim.config.models import DeviceConfig
from agentic_sim.core.resource import FCFSResource, Reservation
from agentic_sim.hardware.compute_models import (
    AggregatePrefillLatencyModel,
    AttentionLatencyModel,
    AttentionWork,
    ExpertComputeLatencyModel,
    ExpertWork,
    LinearAttentionLatencyModel,
    LinearExpertBatchLatencyModel,
    PrefillLatencyModel,
    PrefillWork,
    StageContext,
)


class DeviceOperation(StrEnum):
    PREFILL = "prefill"
    DECODE_ATTN = "decode_attn"
    DECODE_FFN = "decode_ffn"
    FFN_ALL_TO_ALL = "ffn_all_to_all"
    KV_P2P = "kv_p2p"
    KV_POOL_LOAD = "kv_pool_load"
    KV_POOL_OFFLOAD = "kv_pool_offload"


_COMPUTE_OPERATIONS = {
    DeviceOperation.PREFILL,
    DeviceOperation.DECODE_ATTN,
    DeviceOperation.DECODE_FFN,
}


@dataclass(frozen=True)
class DeviceReservation:
    """One job reserved on a physical device queue."""

    device_id: int
    operation: DeviceOperation
    reservation: Reservation

    @property
    def start_time(self) -> float:
        return self.reservation.start_time

    @property
    def finish_time(self) -> float:
        return self.reservation.finish_time

    @property
    def queue_delay(self) -> float:
        return self.reservation.queue_delay

    @property
    def service_time(self) -> float:
        return self.reservation.service_time


@dataclass
class Device:
    """A physical device whose capability jobs all serialize on one queue."""

    device_id: int
    prefill_model: PrefillLatencyModel | None = None
    attention_model: AttentionLatencyModel | None = None
    expert_model: ExpertComputeLatencyModel | None = None
    attention_kv_capacity_tokens: int | None = None
    queue: FCFSResource = field(init=False, repr=False)
    reservations: list[DeviceReservation] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.device_id, bool) or not isinstance(self.device_id, int) or self.device_id < 0:
            raise ValueError("device_id must be a non-negative integer")
        if self.prefill_model is None and self.attention_model is None and self.expert_model is None:
            raise ValueError("device must provide at least one compute capability")
        if self.attention_model is None:
            if self.attention_kv_capacity_tokens is not None:
                raise ValueError("attention KV capacity requires a decode_attn capability")
        elif (
            isinstance(self.attention_kv_capacity_tokens, bool)
            or not isinstance(self.attention_kv_capacity_tokens, int)
            or self.attention_kv_capacity_tokens < 0
        ):
            raise ValueError("decode_attn capability requires non-negative attention KV capacity")
        self.queue = FCFSResource(f"device:{self.device_id}")

    @classmethod
    def from_config(cls, config: DeviceConfig) -> Device:
        """Construct capability models from one typed hardware device record."""

        capabilities = config.capabilities
        return cls(
            device_id=config.id,
            prefill_model=(
                AggregatePrefillLatencyModel(capabilities.prefill.tokens_per_second)
                if capabilities.prefill is not None
                else None
            ),
            attention_model=(
                LinearAttentionLatencyModel(capabilities.decode_attn.context_tokens_per_second)
                if capabilities.decode_attn is not None
                else None
            ),
            expert_model=(
                LinearExpertBatchLatencyModel(capabilities.decode_ffn.tokens_per_second)
                if capabilities.decode_ffn is not None
                else None
            ),
            attention_kv_capacity_tokens=(
                capabilities.decode_attn.kv_capacity_tokens
                if capabilities.decode_attn is not None
                else None
            ),
        )

    @property
    def capabilities(self) -> frozenset[DeviceOperation]:
        result: set[DeviceOperation] = set()
        if self.prefill_model is not None:
            result.add(DeviceOperation.PREFILL)
        if self.attention_model is not None:
            result.add(DeviceOperation.DECODE_ATTN)
        if self.expert_model is not None:
            result.add(DeviceOperation.DECODE_FFN)
        return frozenset(result)

    def _reserve(
        self,
        operation: DeviceOperation,
        ready_time: float,
        service_time: float,
        earliest_start: float | None = None,
    ) -> DeviceReservation:
        if not math.isfinite(ready_time) or ready_time < 0.0:
            raise ValueError("ready_time must be finite and non-negative")
        if not math.isfinite(service_time) or service_time < 0.0:
            raise ValueError("service_time must be finite and non-negative")
        result = DeviceReservation(
            device_id=self.device_id,
            operation=operation,
            reservation=self.queue.reserve(ready_time, service_time, earliest_start=earliest_start),
        )
        self.reservations.append(result)
        return result

    def submit_prefill(
        self,
        batch: Sequence[PrefillWork],
        context: StageContext,
        ready_time: float,
    ) -> DeviceReservation:
        if self.prefill_model is None:
            raise ValueError(f"device {self.device_id} has no prefill capability")
        service_time = self.prefill_model.latency_s(batch, context)
        return self._reserve(DeviceOperation.PREFILL, ready_time, service_time)

    def submit_attention(
        self,
        batch: Sequence[AttentionWork],
        context: StageContext,
        ready_time: float,
    ) -> DeviceReservation:
        if self.attention_model is None:
            raise ValueError(f"device {self.device_id} has no decode_attn capability")
        if any(work.device_id != self.device_id for work in batch):
            raise ValueError(f"all attention work must target device {self.device_id}")
        service_time = self.attention_model.latency_s(batch, context)
        return self._reserve(DeviceOperation.DECODE_ATTN, ready_time, service_time)

    def submit_experts(
        self,
        expert_batches: Sequence[ExpertWork],
        context: StageContext,
        ready_time: float,
    ) -> DeviceReservation:
        """Submit all resident-expert batches as one atomic physical-device job."""

        if self.expert_model is None:
            raise ValueError(f"device {self.device_id} has no decode_ffn capability")
        if any(work.device_id != self.device_id for work in expert_batches):
            raise ValueError(f"all expert work must target device {self.device_id}")
        service_time = self.expert_model.latency_s(expert_batches, context)
        return self._reserve(DeviceOperation.DECODE_FFN, ready_time, service_time)

    def reserve_external(
        self,
        operation: DeviceOperation,
        ready_time: float,
        service_time: float,
        earliest_start: float | None = None,
    ) -> DeviceReservation:
        """Reserve this device for a communication or KV-pool operation."""

        if operation in _COMPUTE_OPERATIONS:
            raise ValueError("compute operations must be submitted through their capability model")
        return self._reserve(operation, ready_time, service_time, earliest_start)
