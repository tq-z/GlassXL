"""Deterministic stateful and stateless prefill placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from agentic_sim.config.models import PrefillPlacement, PrefillState


class PrefillPlacementPolicy(Protocol):
    device_ids: tuple[int, ...]

    def select(self, session_id: int) -> int:
        """Select the prefill device for one session turn."""

    def affinity_for(self, session_id: int) -> int | None:
        """Return retained stateful affinity, if this policy retains one."""


def _validate(device_ids: tuple[int, ...]) -> None:
    if not device_ids:
        raise ValueError("prefill placement requires at least one device")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("prefill placement device IDs must be unique")
    if any(
        isinstance(device_id, bool)
        or not isinstance(device_id, int)
        or device_id < 0
        for device_id in device_ids
    ):
        raise ValueError("prefill device IDs must be non-negative integers")


def _session_id(session_id: int) -> None:
    if isinstance(session_id, bool) or not isinstance(session_id, int) or session_id < 0:
        raise ValueError("session_id must be a non-negative integer")


@dataclass
class StickyRoundRobinPrefillPlacement:
    """Assign once by round robin and retain affinity with the prefill mirror."""

    device_ids: tuple[int, ...]
    cursor: int = 0
    affinities: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate(self.device_ids)
        if not 0 <= self.cursor < len(self.device_ids):
            raise ValueError("sticky round-robin cursor is outside the device list")

    def select(self, session_id: int) -> int:
        _session_id(session_id)
        existing = self.affinities.get(session_id)
        if existing is not None:
            return existing
        selected = self.device_ids[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.device_ids)
        self.affinities[session_id] = selected
        return selected

    def affinity_for(self, session_id: int) -> int | None:
        _session_id(session_id)
        return self.affinities.get(session_id)


@dataclass
class RoundRobinPrefillPlacement:
    """Select a device for every stateless turn without retaining affinity."""

    device_ids: tuple[int, ...]
    cursor: int = 0

    def __post_init__(self) -> None:
        _validate(self.device_ids)
        if not 0 <= self.cursor < len(self.device_ids):
            raise ValueError("round-robin cursor is outside the device list")

    def select(self, session_id: int) -> int:
        _session_id(session_id)
        selected = self.device_ids[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.device_ids)
        return selected

    def affinity_for(self, session_id: int) -> int | None:
        _session_id(session_id)
        return None


def build_prefill_placement_policy(
    state: PrefillState,
    placement: PrefillPlacement,
    device_ids: tuple[int, ...],
) -> PrefillPlacementPolicy:
    """Build the one valid confirmed placement for the configured state mode."""

    if (
        state is PrefillState.STATEFUL
        and placement is PrefillPlacement.STICKY_ROUND_ROBIN
    ):
        return StickyRoundRobinPrefillPlacement(device_ids)
    if (
        state is PrefillState.STATELESS
        and placement is PrefillPlacement.ROUND_ROBIN
    ):
        return RoundRobinPrefillPlacement(device_ids)
    raise ValueError(
        f"invalid V1 prefill state/placement combination: {state}/{placement}"
    )
