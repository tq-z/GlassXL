"""Deterministic local attention batches for one synchronized decode step."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from agentic_sim.hardware.compute_models import AttentionWork
from agentic_sim.runtime.session import (
    AttentionKVResidency,
    DecodeEpisodePhase,
    SessionState,
)


def _time(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class LocalAttentionBatch:
    """The active sessions and analytical work assigned to one DP device."""

    device_id: int
    sessions: tuple[SessionState, ...]
    work: tuple[AttentionWork, ...]


@dataclass(frozen=True)
class DecodeBatch:
    """One global active batch and all mapped DP-local batches."""

    decode_step: int
    ready_time_s: float
    sessions: tuple[SessionState, ...]
    local_batches: tuple[LocalAttentionBatch, ...]

    @property
    def size(self) -> int:
        return len(self.sessions)


class DecodeBatcher:
    """Build immutable step batches without owning queues or session progress."""

    def __init__(self, device_ids: tuple[int, ...]) -> None:
        if not device_ids or len(set(device_ids)) != len(device_ids):
            raise ValueError("attention batch device IDs must be non-empty and unique")
        self.device_ids = device_ids

    def form(
        self,
        sessions: Sequence[SessionState],
        decode_step: int,
        ready_time_s: float,
    ) -> DecodeBatch:
        if isinstance(decode_step, bool) or not isinstance(decode_step, int) or decode_step < 0:
            raise ValueError("decode_step must be a non-negative integer")
        ready = _time(ready_time_s, "ready_time_s")
        ordered = tuple(sorted(sessions, key=lambda session: session.session_id))
        session_ids = tuple(session.session_id for session in ordered)
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("decode batch contains duplicate session IDs")
        if not ordered:
            raise ValueError("decode batch requires at least one ready session")

        grouped: dict[int, list[SessionState]] = {
            device_id: [] for device_id in self.device_ids
        }
        for session in ordered:
            episode = session.current_episode
            attention = session.kv.attention
            if episode is None or episode.phase is not DecodeEpisodePhase.ACTIVE:
                raise ValueError("decode batch sessions must have active episodes")
            if attention.state is not AttentionKVResidency.ACTIVE:
                raise ValueError("decode batch sessions must have active attention KV")
            if episode.remaining_output_tokens <= 0:
                raise ValueError("decode batch sessions must have output remaining")
            if episode.attention_ready_time_s is None:
                raise ValueError("decode batch session has a pending attention transfer")
            if episode.attention_ready_time_s > ready:
                raise ValueError("decode batch session is not ready at the launch time")
            if attention.device_id not in grouped:
                raise ValueError("decode batch session affinity is outside the DP mapping")
            grouped[attention.device_id].append(session)

        local_batches = tuple(
            LocalAttentionBatch(
                device_id=device_id,
                sessions=tuple(grouped[device_id]),
                work=tuple(
                    AttentionWork(
                        session_id=session.session_id,
                        context_tokens=session.current_context_tokens,
                        device_id=device_id,
                    )
                    for session in grouped[device_id]
                ),
            )
            for device_id in self.device_ids
        )
        return DecodeBatch(decode_step, ready, ordered, local_batches)
