"""Replaceable token-unit analytical interfaces and V1 compute models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


def _nonnegative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _positive_rate(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite positive number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive number")


@dataclass(frozen=True)
class PrefillWork:
    """One session's input to a prefill model."""

    session_id: int
    context_tokens: int
    prefill_tokens: int
    unsynced_decode_tokens: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.session_id, "session_id")
        _nonnegative_integer(self.context_tokens, "context_tokens")
        _nonnegative_integer(self.prefill_tokens, "prefill_tokens")
        _nonnegative_integer(self.unsynced_decode_tokens, "unsynced_decode_tokens")


@dataclass(frozen=True)
class AttentionWork:
    """One session's input to a decode-attention model."""

    session_id: int
    context_tokens: int
    device_id: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.session_id, "session_id")
        _nonnegative_integer(self.context_tokens, "context_tokens")
        _nonnegative_integer(self.device_id, "device_id")


@dataclass(frozen=True)
class ExpertWork:
    """One expert batch assigned to one physical FFN device."""

    expert_id: int
    activation_tokens: int
    device_id: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.expert_id, "expert_id")
        _nonnegative_integer(self.activation_tokens, "activation_tokens")
        _nonnegative_integer(self.device_id, "device_id")


@dataclass(frozen=True)
class StageContext:
    """Execution position exposed to both V1 and future layer-wise models."""

    decode_step: int
    layer_id: int | None = None

    def __post_init__(self) -> None:
        _nonnegative_integer(self.decode_step, "decode_step")
        if self.layer_id is not None:
            _nonnegative_integer(self.layer_id, "layer_id")


@runtime_checkable
class PrefillLatencyModel(Protocol):
    def latency_s(
        self,
        batch: Sequence[PrefillWork],
        context: StageContext,
    ) -> float:
        """Return prefill service time in seconds."""


@runtime_checkable
class AttentionLatencyModel(Protocol):
    def latency_s(
        self,
        batch: Sequence[AttentionWork],
        context: StageContext,
    ) -> float:
        """Return attention service time in seconds."""


@runtime_checkable
class ExpertComputeLatencyModel(Protocol):
    def latency_s(
        self,
        expert_batches: Sequence[ExpertWork],
        context: StageContext,
    ) -> float:
        """Return one atomic per-device FFN job's service time in seconds."""


@runtime_checkable
class CollectiveLatencyModel(Protocol):
    def latency_s(self, send_tokens: int, participants: int) -> float:
        """Return one collective's service time in seconds."""


@runtime_checkable
class P2PLatencyModel(Protocol):
    def latency_s(self, tokens: int, src: int, dst: int) -> float:
        """Return one point-to-point transfer's service time in seconds."""


@dataclass(frozen=True)
class AggregatePrefillLatencyModel:
    """V1 prefill model: sum(P_i) / aggregate token rate."""

    tokens_per_second: float

    def __post_init__(self) -> None:
        _positive_rate(self.tokens_per_second, "tokens_per_second")

    def latency_s(
        self,
        batch: Sequence[PrefillWork],
        context: StageContext,
    ) -> float:
        del context
        return sum(work.prefill_tokens for work in batch) / self.tokens_per_second


@dataclass(frozen=True)
class LinearAttentionLatencyModel:
    """V1 attention model: sum(C_i) / context token rate."""

    context_tokens_per_second: float

    def __post_init__(self) -> None:
        _positive_rate(self.context_tokens_per_second, "context_tokens_per_second")

    def latency_s(
        self,
        batch: Sequence[AttentionWork],
        context: StageContext,
    ) -> float:
        del context
        return sum(work.context_tokens for work in batch) / self.context_tokens_per_second


@dataclass(frozen=True)
class LinearExpertBatchLatencyModel:
    """V1 atomic FFN-device model: sum(B_e) / routed token rate."""

    tokens_per_second: float

    def __post_init__(self) -> None:
        _positive_rate(self.tokens_per_second, "tokens_per_second")

    def latency_s(
        self,
        expert_batches: Sequence[ExpertWork],
        context: StageContext,
    ) -> float:
        del context
        return sum(work.activation_tokens for work in expert_batches) / self.tokens_per_second
