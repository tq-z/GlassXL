"""Sequential stateful/stateless prefill orchestration for Milestone 5."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from agentic_sim.config.models import PrefillMappingConfig, PrefillState
from agentic_sim.hardware.compute_models import PrefillWork, StageContext
from agentic_sim.hardware.device import DeviceReservation
from agentic_sim.hardware.interconnect import InterconnectReservation
from agentic_sim.hardware.kv_pool import KVPoolReservation
from agentic_sim.hardware.topology import Topology
from agentic_sim.mapping.prefill import (
    PrefillPlacementPolicy,
    build_prefill_placement_policy,
)
from agentic_sim.runtime.attention_admission import (
    AttentionAdmission,
    AttentionAdmissionController,
)
from agentic_sim.runtime.prefill_batcher import PrefillBatch, PrefillBatcher
from agentic_sim.runtime.session import (
    DecodeEpisodePhase,
    KVSnapshot,
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
class PrefillSessionResult:
    """Token movement and model input for one session in a prefill batch."""

    session_id: int
    context_tokens: int
    prefill_tokens: int
    unsynced_decode_tokens: int
    load_tokens: int


@dataclass(frozen=True)
class PrefillCopyLease:
    """A retained or transient current prefill copy awaiting attention transfer."""

    session: SessionState
    device_id: int
    batch_id: int
    state: PrefillState


@dataclass(frozen=True)
class PrefillTransferResult:
    """One-startup P2P transfer for a logical batch of admitted sessions."""

    source_device_id: int
    destination_device_ids: tuple[int, ...]
    admissions: tuple[AttentionAdmission, ...]
    total_tokens: int
    reservation: InterconnectReservation


@dataclass(frozen=True)
class PrefillBatchResult:
    """All sequential reservations and outcomes for one completed prefill batch."""

    batch: PrefillBatch
    sessions: tuple[PrefillSessionResult, ...]
    pool_load: KVPoolReservation
    compute: DeviceReservation
    transfers: tuple[PrefillTransferResult, ...]
    blocked_session_ids: tuple[int, ...]


@dataclass(frozen=True)
class _ScheduledBatch:
    batch: PrefillBatch
    sessions: tuple[PrefillSessionResult, ...]
    pool_load: KVPoolReservation
    compute: DeviceReservation


class PrefillController:
    """Own placement, FCFS batches, and serialized pool/compute/P2P stages."""

    def __init__(
        self,
        topology: Topology,
        mapping: PrefillMappingConfig,
        attention: AttentionAdmissionController,
    ) -> None:
        self.topology = topology
        self.mapping = mapping
        self.attention = attention
        self.placement: PrefillPlacementPolicy = build_prefill_placement_policy(
            mapping.state,
            mapping.placement,
            mapping.device_ids,
        )
        for device_id in mapping.device_ids:
            device = topology.get_device(device_id)
            if device.prefill_model is None:
                raise ValueError(f"device {device_id} has no prefill capability")
        self.batcher = PrefillBatcher(mapping.device_ids)
        self._pending_copies: dict[int, PrefillCopyLease] = {}
        self._stateful_sessions: dict[int, SessionState] = {}

    def enqueue(
        self,
        session: SessionState,
        ready_time_s: float,
    ) -> int:
        ready = _time(ready_time_s, "ready_time_s")
        if session.session_id in self._pending_copies:
            raise ValueError("session already has a prefill copy awaiting transfer")
        episode = session.current_episode
        if episode is None or episode.phase is not DecodeEpisodePhase.TURN_READY:
            raise ValueError("prefill enqueue requires a turn-ready episode")
        if ready < episode.turn_ready_time_s:
            raise ValueError("prefill enqueue cannot precede turn readiness")
        if ready < self.batcher.last_enqueue_time_s:
            raise ValueError("prefill requests must be enqueued in timestamp order")
        if any(
            request.session is session for request in self.batcher.waiting()
        ):
            raise ValueError("session is already waiting for prefill")
        device_id = self.placement.select(session.session_id)
        self.batcher.enqueue(session, device_id, ready)
        return device_id

    def process_ready(self, launch_time_s: float) -> tuple[PrefillBatchResult, ...]:
        """Launch one ready batch per idle prefill device at this timestamp."""

        launch = _time(launch_time_s, "launch_time_s")
        scheduled: list[_ScheduledBatch] = []
        for device_id in self.mapping.device_ids:
            device = self.topology.get_device(device_id)
            batch = self.batcher.form_batch(
                device_id,
                launch,
                device_available_time_s=device.queue.available_time,
            )
            if batch is not None:
                scheduled.append(self._schedule_batch(batch))

        results = [
            self._complete_batch(item)
            for item in sorted(
                scheduled,
                key=lambda item: (item.compute.finish_time, item.batch.batch_id),
            )
        ]
        return tuple(results)

    def complete_attention_admissions(
        self,
        admissions: Sequence[AttentionAdmission],
    ) -> tuple[PrefillTransferResult, ...]:
        """Transfer copies for immediate or later capacity admissions."""

        grouped: dict[tuple[int, float], list[AttentionAdmission]] = {}
        session_ids = [admission.session_id for admission in admissions]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("attention admissions contain a duplicate session")
        for admission in admissions:
            try:
                lease = self._pending_copies[admission.session_id]
            except KeyError as error:
                raise ValueError(
                    f"session {admission.session_id} has no pending prefill copy"
                ) from error
            key = (lease.device_id, admission.admission_time_s)
            grouped.setdefault(key, []).append(admission)

        transfers: list[PrefillTransferResult] = []
        for (source_id, ready_time), group in grouped.items():
            destination_ids = tuple(
                dict.fromkeys(admission.device_id for admission in group)
            )
            total_tokens = sum(admission.transfer_tokens for admission in group)
            reservation = self.topology.submit_kv_p2p_batch(
                total_tokens,
                source_id,
                destination_ids,
                ready_time,
            )
            for admission in group:
                lease = self._pending_copies[admission.session_id]
                session = lease.session
                session.complete_attention_transfer(reservation.finish_time)
                if lease.state is PrefillState.STATELESS:
                    session.kv.discard_prefill_copy()
                del self._pending_copies[admission.session_id]
            transfers.append(
                PrefillTransferResult(
                    source_device_id=source_id,
                    destination_device_ids=destination_ids,
                    admissions=tuple(group),
                    total_tokens=total_tokens,
                    reservation=reservation,
                )
            )
        return tuple(transfers)

    def pending_copy(self, session_id: int) -> PrefillCopyLease | None:
        return self._pending_copies.get(session_id)

    @property
    def capacity_tokens(self) -> None:
        """V1 models shared retained-prefill memory as explicitly unbounded."""

        return None

    @property
    def retained_stateful_tokens(self) -> int:
        """Unbounded V1 retained-prefill footprint, reported but never rejected."""

        return sum(
            snapshot.context_tokens
            for session in self._stateful_sessions.values()
            if (snapshot := session.kv.prefill_snapshot) is not None
        )

    def next_launch_time_s(self) -> float | None:
        candidates: list[float] = []
        for device_id in self.mapping.device_ids:
            waiting = self.batcher.waiting(device_id)
            if waiting:
                candidates.append(
                    max(
                        waiting[0].ready_time_s,
                        self.topology.get_device(device_id).queue.available_time,
                    )
                )
        return min(candidates) if candidates else None

    def _schedule_batch(self, batch: PrefillBatch) -> _ScheduledBatch:
        device = self.topology.get_device(batch.device_id)
        session_results: list[PrefillSessionResult] = []
        works: list[PrefillWork] = []

        for request in batch.requests:
            session = request.session
            if session.session_id in self._pending_copies:
                raise ValueError("session already has a prefill copy awaiting transfer")
            previous = self._stateful_sessions.get(session.session_id)
            if (
                self.mapping.state is PrefillState.STATEFUL
                and previous is not None
                and previous is not session
            ):
                raise ValueError("stateful session ID is already owned by another object")

        for request in batch.requests:
            session = request.session
            if self.mapping.state is PrefillState.STATELESS:
                session.kv.discard_prefill_copy()
            unsynced_decode_tokens = self._unsynced_decode_tokens(session)
            episode = session.current_episode
            if episode is None:
                raise ValueError("prefill request has no active episode")
            context_tokens = episode.context_before_prefill
            load_tokens = session.prepare_prefill()
            result = PrefillSessionResult(
                session_id=session.session_id,
                context_tokens=context_tokens,
                prefill_tokens=episode.turn.prefill_tokens,
                unsynced_decode_tokens=unsynced_decode_tokens,
                load_tokens=load_tokens,
            )
            session_results.append(result)
            works.append(
                PrefillWork(
                    session_id=session.session_id,
                    context_tokens=context_tokens,
                    prefill_tokens=episode.turn.prefill_tokens,
                    unsynced_decode_tokens=unsynced_decode_tokens,
                )
            )
            self._pending_copies[session.session_id] = PrefillCopyLease(
                session=session,
                device_id=batch.device_id,
                batch_id=batch.batch_id,
                state=self.mapping.state,
            )
            if self.mapping.state is PrefillState.STATEFUL:
                self._stateful_sessions[session.session_id] = session

        pool_load = self.topology.submit_pool_load(
            sum(item.load_tokens for item in session_results),
            batch.device_id,
            batch.launch_time_s,
        )
        compute = device.submit_prefill(
            tuple(works),
            StageContext(decode_step=0),
            ready_time=pool_load.finish_time,
        )
        return _ScheduledBatch(
            batch=batch,
            sessions=tuple(session_results),
            pool_load=pool_load,
            compute=compute,
        )

    def _complete_batch(self, scheduled: _ScheduledBatch) -> PrefillBatchResult:
        admissions: list[AttentionAdmission] = []
        blocked: list[int] = []
        for request in scheduled.batch.requests:
            session = request.session
            session.apply_prefill()
            admission = self.attention.submit(
                session,
                scheduled.compute.finish_time,
            )
            if admission is None:
                blocked.append(session.session_id)
            else:
                admissions.append(admission)
        transfers = self.complete_attention_admissions(admissions)
        return PrefillBatchResult(
            batch=scheduled.batch,
            sessions=scheduled.sessions,
            pool_load=scheduled.pool_load,
            compute=scheduled.compute,
            transfers=transfers,
            blocked_session_ids=tuple(blocked),
        )

    @staticmethod
    def _unsynced_decode_tokens(session: SessionState) -> int:
        snapshot: KVSnapshot | None = session.kv.prefill_snapshot
        if snapshot is None:
            return 0
        version_delta = session.kv.version - snapshot.version
        token_delta = session.current_context_tokens - snapshot.context_tokens
        if version_delta < 0 or token_delta != version_delta:
            raise ValueError("prefill mirror has inconsistent token/version deltas")
        return token_delta
