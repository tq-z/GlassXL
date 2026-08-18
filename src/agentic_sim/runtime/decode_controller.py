"""Synchronized whole-model PD/AFD decode steps and episode output flushes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agentic_sim.config.models import MappingConfig
from agentic_sim.hardware.compute_models import ExpertWork, StageContext
from agentic_sim.hardware.device import DeviceReservation
from agentic_sim.hardware.interconnect import InterconnectReservation
from agentic_sim.hardware.kv_pool import KVPoolReservation
from agentic_sim.hardware.topology import Topology
from agentic_sim.mapping.experts import ExpertDeviceBatch, ExpertMapping
from agentic_sim.runtime.attention_admission import (
    AttentionAdmissionController,
    PoolWriteCompletion,
)
from agentic_sim.runtime.decode_batcher import DecodeBatch, DecodeBatcher
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
class PendingPoolWrite:
    """A modeled episode write that remains dirty until explicitly completed."""

    session: SessionState
    payload_tokens: int
    reservation: KVPoolReservation

    @property
    def session_id(self) -> int:
        return self.session.session_id


@dataclass(frozen=True)
class CompletedPoolWrite:
    """A completed pool write and any capacity waiters admitted by cleanup."""

    pending: PendingPoolWrite
    completion: PoolWriteCompletion


@dataclass(frozen=True)
class DecodeStepResult:
    """All work and barriers for one whole-model synchronized decode step."""

    decode_step: int
    batch: DecodeBatch
    attention: tuple[DeviceReservation, ...]
    attention_barrier_s: float
    dispatch: InterconnectReservation
    activation_tokens: int
    send_tokens: int
    expert_loads: tuple[ExpertWork, ...]
    expert_batches: tuple[ExpertDeviceBatch, ...]
    ffn: tuple[DeviceReservation, ...]
    ffn_barrier_s: float
    combine: InterconnectReservation
    token_completion_time_s: float
    emitted_context_tokens: tuple[tuple[int, int], ...]
    pool_writes: tuple[PendingPoolWrite, ...]


class DecodeController:
    """Execute the frozen attention/barrier/AFD/barrier/combine step contract."""

    def __init__(
        self,
        topology: Topology,
        mapping: MappingConfig,
        attention: AttentionAdmissionController,
    ) -> None:
        if mapping.granularity != "decode_step":
            raise ValueError("V1 decode requires decode_step granularity")
        if attention.device_ids != mapping.decode_attn.device_ids:
            raise ValueError("attention admission and decode mapping device order must match")
        self.topology = topology
        self.mapping = mapping
        self.attention = attention
        self.batcher = DecodeBatcher(mapping.decode_attn.device_ids)
        self.experts = ExpertMapping.from_config(
            mapping.decode_ffn,
            mapping.model.num_experts,
        )
        for device_id in mapping.decode_attn.device_ids:
            if topology.get_device(device_id).attention_model is None:
                raise ValueError(f"device {device_id} has no decode_attn capability")
        for device_id in mapping.decode_ffn.device_ids:
            if topology.get_device(device_id).expert_model is None:
                raise ValueError(f"device {device_id} has no decode_ffn capability")
        self.participant_ids = tuple(
            dict.fromkeys(
                mapping.decode_attn.device_ids + mapping.decode_ffn.device_ids
            )
        )
        self._next_decode_step = 0
        self._pending_pool_writes: dict[int, PendingPoolWrite] = {}

    @property
    def next_decode_step(self) -> int:
        return self._next_decode_step

    @property
    def pending_pool_writes(self) -> tuple[PendingPoolWrite, ...]:
        return tuple(
            sorted(
                self._pending_pool_writes.values(),
                key=lambda item: (item.reservation.finish_time, item.session_id),
            )
        )

    def ready_sessions(self, ready_time_s: float) -> tuple[SessionState, ...]:
        """Return positive-output sessions eligible for a step at this time."""

        ready = _time(ready_time_s, "ready_time_s")
        return self._ready_sessions(ready, require_output=True)

    def ready_zero_output_sessions(
        self,
        ready_time_s: float,
    ) -> tuple[SessionState, ...]:
        """Return admitted zero-output sessions eligible for their pool flush."""

        ready = _time(ready_time_s, "ready_time_s")
        return self._ready_sessions(ready, require_output=False)

    def run_step(self, ready_time_s: float) -> DecodeStepResult:
        """Run exactly one whole-model step for every decode-ready active session."""

        ready = _time(ready_time_s, "ready_time_s")
        sessions = self._ready_sessions(ready, require_output=True)
        batch = self.batcher.form(sessions, self._next_decode_step, ready)
        context = StageContext(decode_step=self._next_decode_step)
        affinities = {
            session.session_id: session.kv.attention.device_id
            for session in batch.sessions
        }

        attention_reservations = tuple(
            self.topology.get_device(local.device_id).submit_attention(
                local.work,
                context,
                ready,
            )
            for local in batch.local_batches
        )
        attention_barrier = max(item.finish_time for item in attention_reservations)

        activation_tokens = batch.size * self.mapping.model.top_k
        ep_width = len(self.mapping.decode_ffn.device_ids)
        send_tokens = max(1, (activation_tokens + ep_width - 1) // ep_width)
        dispatch = self.topology.submit_all_to_all(
            send_tokens,
            self.participant_ids,
            attention_barrier,
        )

        expert_batches = self.experts.device_batches(activation_tokens)
        expert_loads = tuple(
            sorted(
                (
                    work
                    for device_batch in expert_batches
                    for work in device_batch.experts
                ),
                key=lambda work: work.expert_id,
            )
        )
        if sum(work.activation_tokens for work in expert_loads) != activation_tokens:
            raise RuntimeError("expert work does not conserve routed activations")
        ffn_reservations = tuple(
            self.topology.get_device(device_batch.device_id).submit_experts(
                device_batch.experts,
                context,
                dispatch.finish_time,
            )
            for device_batch in expert_batches
        )
        ffn_barrier = max(item.finish_time for item in ffn_reservations)
        combine = self.topology.submit_all_to_all(
            send_tokens,
            self.participant_ids,
            ffn_barrier,
        )

        emitted: list[tuple[int, int]] = []
        pool_writes: list[PendingPoolWrite] = []
        for session in batch.sessions:
            context_tokens = session.emit_decode_token(combine.finish_time)
            emitted.append((session.session_id, context_tokens))
            if session.kv.attention.device_id != affinities[session.session_id]:
                raise RuntimeError("attention affinity changed during a decode episode")
            episode = session.current_episode
            if episode is None:
                raise RuntimeError("decode session lost its active episode")
            if episode.remaining_output_tokens == 0:
                pool_writes.append(
                    self._schedule_pool_write(session, combine.finish_time)
                )

        result = DecodeStepResult(
            decode_step=self._next_decode_step,
            batch=batch,
            attention=attention_reservations,
            attention_barrier_s=attention_barrier,
            dispatch=dispatch,
            activation_tokens=activation_tokens,
            send_tokens=send_tokens,
            expert_loads=expert_loads,
            expert_batches=expert_batches,
            ffn=ffn_reservations,
            ffn_barrier_s=ffn_barrier,
            combine=combine,
            token_completion_time_s=combine.finish_time,
            emitted_context_tokens=tuple(emitted),
            pool_writes=tuple(pool_writes),
        )
        self._next_decode_step += 1
        return result

    def flush_zero_output_sessions(
        self,
        ready_time_s: float,
    ) -> tuple[PendingPoolWrite, ...]:
        """Schedule P-only episode writes without fabricating a decode token."""

        ready = _time(ready_time_s, "ready_time_s")
        sessions = self._ready_sessions(ready, require_output=False)
        return tuple(self._schedule_pool_write(session, ready) for session in sessions)

    def complete_pool_write(self, session_id: int) -> CompletedPoolWrite:
        """Apply one dirty-to-clean transition at its reserved completion time."""

        try:
            pending = self._pending_pool_writes[session_id]
        except KeyError as error:
            raise KeyError(f"session {session_id} has no pending pool write") from error
        completion = self.attention.complete_pool_write(
            pending.session,
            pending.reservation.finish_time,
        )
        if completion.pool_write_tokens != pending.payload_tokens:
            raise RuntimeError("completed pool payload differs from its reservation")
        del self._pending_pool_writes[session_id]
        return CompletedPoolWrite(pending, completion)

    def complete_pool_writes_through(
        self,
        completion_time_s: float,
    ) -> tuple[CompletedPoolWrite, ...]:
        """Complete every reserved write whose finish time is at or before time."""

        completion_time = _time(completion_time_s, "completion_time_s")
        ready = tuple(
            pending
            for pending in self.pending_pool_writes
            if pending.reservation.finish_time <= completion_time
        )
        return tuple(self.complete_pool_write(item.session_id) for item in ready)

    def _ready_sessions(
        self,
        ready_time_s: float,
        *,
        require_output: bool,
    ) -> tuple[SessionState, ...]:
        sessions: list[SessionState] = []
        seen: set[int] = set()
        for device_id in self.mapping.decode_attn.device_ids:
            device_state = self.attention.devices[device_id]
            for session in device_state.resident_sessions.values():
                if session.session_id in seen:
                    raise RuntimeError("session is resident on multiple attention devices")
                seen.add(session.session_id)
                episode = session.current_episode
                if (
                    episode is None
                    or episode.phase is not DecodeEpisodePhase.ACTIVE
                    or session.kv.attention.state is not AttentionKVResidency.ACTIVE
                    or episode.attention_ready_time_s is None
                    or episode.attention_ready_time_s > ready_time_s
                ):
                    continue
                has_output = episode.remaining_output_tokens > 0
                if has_output is require_output:
                    sessions.append(session)
        return tuple(sorted(sessions, key=lambda session: session.session_id))

    def _schedule_pool_write(
        self,
        session: SessionState,
        ready_time_s: float,
    ) -> PendingPoolWrite:
        if session.session_id in self._pending_pool_writes:
            raise ValueError("session already has a pending pool write")
        episode = session.current_episode
        if episode is None or episode.phase is not DecodeEpisodePhase.ACTIVE:
            raise ValueError("pool write scheduling requires an active decode episode")
        if episode.remaining_output_tokens != 0:
            raise ValueError("pool write scheduling requires completed oracle output")
        device_id = session.kv.attention.device_id
        if device_id is None:
            raise ValueError("pool write requires attention device affinity")
        payload_tokens = session.kv.pool.missing_tokens(session.kv.current_snapshot)
        expected_tokens = episode.turn.prefill_tokens + episode.generated_tokens
        if payload_tokens != expected_tokens:
            raise ValueError(
                f"pool payload {payload_tokens} does not match unsynchronized P+O "
                f"{expected_tokens}"
            )
        session.finish_decode()
        reservation = self.topology.submit_pool_offload(
            payload_tokens,
            device_id,
            ready_time_s,
        )
        pending = PendingPoolWrite(session, payload_tokens, reservation)
        self._pending_pool_writes[session.session_id] = pending
        return pending
