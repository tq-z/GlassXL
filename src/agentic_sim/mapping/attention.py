"""Deterministic attention placement policy implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from agentic_sim.config.models import AttentionPlacement


class AttentionPlacementPolicy(Protocol):
    """Select one feasible device, or return ``None`` without side effects."""

    device_ids: tuple[int, ...]

    def select(
        self,
        feasible_device_ids: frozenset[int],
        active_context_loads: Mapping[int, int],
    ) -> int | None: ...


def _validate_device_ids(device_ids: tuple[int, ...]) -> None:
    if not device_ids:
        raise ValueError("attention placement requires at least one device")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("attention placement device IDs must be unique")
    if any(
        isinstance(device_id, bool)
        or not isinstance(device_id, int)
        or device_id < 0
        for device_id in device_ids
    ):
        raise ValueError("attention device IDs must be non-negative integers")


@dataclass
class RoundRobinAttentionPlacement:
    """Scan once from the cursor and advance only after successful placement."""

    device_ids: tuple[int, ...]
    cursor: int = 0

    def __post_init__(self) -> None:
        _validate_device_ids(self.device_ids)
        if not 0 <= self.cursor < len(self.device_ids):
            raise ValueError("round-robin cursor is outside the device list")

    def select(
        self,
        feasible_device_ids: frozenset[int],
        active_context_loads: Mapping[int, int],
    ) -> int | None:
        del active_context_loads
        for offset in range(len(self.device_ids)):
            index = (self.cursor + offset) % len(self.device_ids)
            device_id = self.device_ids[index]
            if device_id in feasible_device_ids:
                self.cursor = (index + 1) % len(self.device_ids)
                return device_id
        return None


@dataclass(frozen=True)
class LeastActiveContextPlacement:
    """Choose minimum active context load, breaking ties by physical ID."""

    device_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_device_ids(self.device_ids)

    def select(
        self,
        feasible_device_ids: frozenset[int],
        active_context_loads: Mapping[int, int],
    ) -> int | None:
        candidates = set(self.device_ids) & feasible_device_ids
        if not candidates:
            return None
        missing = candidates - set(active_context_loads)
        if missing:
            raise ValueError(
                f"missing active-context load for device {min(missing)}"
            )
        return min(candidates, key=lambda item: (active_context_loads[item], item))


def build_attention_placement_policy(
    placement: AttentionPlacement,
    device_ids: tuple[int, ...],
) -> AttentionPlacementPolicy:
    """Construct exactly one of the two confirmed V1 placement policies."""

    if placement is AttentionPlacement.ROUND_ROBIN:
        return RoundRobinAttentionPlacement(device_ids)
    if placement is AttentionPlacement.LEAST_ACTIVE_CONTEXT:
        return LeastActiveContextPlacement(device_ids)
    raise ValueError(f"unsupported attention placement policy: {placement}")
