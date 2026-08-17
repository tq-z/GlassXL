"""Attention capacity admission, placement, and cross-turn residency."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from agentic_sim.config.models import (
    AttentionPlacement,
    AttentionResidency,
    DecodeAttentionMappingConfig,
)
from agentic_sim.hardware.device import Device
from agentic_sim.mapping.attention import (
    AttentionPlacementPolicy,
    build_attention_placement_policy,
)
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
class AttentionAdmission:
    """One successful capacity reservation and logical KV restore decision."""

    session_id: int
    device_id: int
    admission_time_s: float
    required_final_kv_tokens: int
    transfer_tokens: int
    cache_hit: bool
    evicted_session_ids: tuple[int, ...]
    capacity_wait_s: float
    transfer_pending: bool = False


@dataclass(frozen=True)
class CapacityWaiter:
    """One FCFS capacity waiter ordered by insertion sequence."""

    sequence_id: int
    enqueued_time_s: float
    session: SessionState
    transfer_pending: bool = False


@dataclass(frozen=True)
class PoolWriteCompletion:
    """Pool-write result plus waiters admitted by the resulting capacity change."""

    pool_write_tokens: int
    admissions: tuple[AttentionAdmission, ...]


@dataclass(frozen=True)
class AttentionCapacityPlan:
    """A validated device-local admission plan that has not yet mutated state."""

    device_id: int
    required_final_kv_tokens: int
    transfer_tokens: int
    cache_hit: bool
    evicted_session_ids: tuple[int, ...]


@dataclass
class AttentionDeviceState:
    """Capacity and resident-session accounting for one attention device."""

    device_id: int
    capacity_tokens: int
    resident_sessions: dict[int, SessionState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.device_id, bool)
            or not isinstance(self.device_id, int)
            or self.device_id < 0
        ):
            raise ValueError("device_id must be a non-negative integer")
        if (
            isinstance(self.capacity_tokens, bool)
            or not isinstance(self.capacity_tokens, int)
            or self.capacity_tokens < 0
        ):
            raise ValueError("capacity_tokens must be a non-negative integer")

    @property
    def active_context_load(self) -> int:
        return sum(
            session.current_context_tokens
            for session in self.resident_sessions.values()
            if session.kv.attention.state is AttentionKVResidency.ACTIVE
        )

    @property
    def active_commitment_tokens(self) -> int:
        return sum(
            session.active_commitment_tokens()
            for session in self.resident_sessions.values()
            if session.kv.attention.state is AttentionKVResidency.ACTIVE
        )

    @property
    def resident_used_tokens(self) -> int:
        return sum(
            self._resident_snapshot_tokens(session)
            for session in self.resident_sessions.values()
        )

    @property
    def reserved_or_used_tokens(self) -> int:
        total = 0
        for session in self.resident_sessions.values():
            state = session.kv.attention.state
            if state is AttentionKVResidency.ACTIVE:
                total += session.active_commitment_tokens()
            else:
                total += self._resident_snapshot_tokens(session)
        return total

    def plan_admission(
        self,
        session: SessionState,
        protected_session_ids: frozenset[int],
    ) -> AttentionCapacityPlan | None:
        episode = session.current_episode
        if episode is None or episode.phase not in {
            DecodeEpisodePhase.PREFILLED,
            DecodeEpisodePhase.WAITING_ATTN_CAPACITY,
        }:
            raise ValueError("attention admission requires a prefilled episode")

        attention = session.kv.attention
        cache_hit = attention.state is AttentionKVResidency.INACTIVE_CLEAN
        if cache_hit:
            if attention.device_id != self.device_id:
                return None
            if self.resident_sessions.get(session.session_id) is not session:
                raise ValueError("cached attention affinity is not registered on its device")
        elif attention.state is AttentionKVResidency.EVICTED:
            if session.session_id in self.resident_sessions:
                raise ValueError("evicted attention KV cannot remain registered as resident")
        else:
            raise ValueError("attention admission requires clean inactive or evicted KV")

        required = session.current_context_tokens + episode.turn.output_tokens
        transfer_tokens = attention.restore_tokens(session.kv.current_snapshot)
        projected = self.reserved_or_used_tokens + required
        if cache_hit:
            projected -= self._resident_snapshot_tokens(session)

        victims: list[int] = []
        evictable = sorted(
            (
                resident
                for resident in self.resident_sessions.values()
                if resident.session_id != session.session_id
                and resident.session_id not in protected_session_ids
                and resident.kv.attention.state
                is AttentionKVResidency.INACTIVE_CLEAN
            ),
            key=lambda resident: resident.kv.attention.lru_key(resident.session_id),
        )
        for resident in evictable:
            if projected <= self.capacity_tokens:
                break
            self._validate_recoverable(resident)
            projected -= self._resident_snapshot_tokens(resident)
            victims.append(resident.session_id)
        if projected > self.capacity_tokens:
            return None
        return AttentionCapacityPlan(
            device_id=self.device_id,
            required_final_kv_tokens=required,
            transfer_tokens=transfer_tokens,
            cache_hit=cache_hit,
            evicted_session_ids=tuple(victims),
        )

    def evict_clean(self, session_id: int) -> SessionState:
        try:
            session = self.resident_sessions[session_id]
        except KeyError as error:
            raise KeyError(
                f"session {session_id} is not resident on attention device {self.device_id}"
            ) from error
        if session.kv.attention.state is not AttentionKVResidency.INACTIVE_CLEAN:
            raise ValueError("only clean inactive attention KV can be evicted")
        self._validate_recoverable(session)
        session.kv.attention.evict()
        del self.resident_sessions[session_id]
        return session

    def assert_within_capacity(self) -> None:
        reserved = self.reserved_or_used_tokens
        if reserved > self.capacity_tokens:
            raise RuntimeError(
                f"attention device {self.device_id} reserves {reserved} tokens "
                f"beyond capacity {self.capacity_tokens}"
            )

    def _resident_snapshot_tokens(self, session: SessionState) -> int:
        attention = session.kv.attention
        if attention.state is AttentionKVResidency.EVICTED:
            raise ValueError("evicted attention KV cannot be registered as resident")
        if attention.device_id != self.device_id or attention.snapshot is None:
            raise ValueError("resident attention KV has inconsistent device affinity")
        return attention.snapshot.context_tokens

    @staticmethod
    def _validate_recoverable(session: SessionState) -> None:
        attention_snapshot = session.kv.attention.snapshot
        if attention_snapshot is None or session.kv.pool.snapshot != attention_snapshot:
            raise ValueError(
                "cannot evict attention KV before its cached snapshot is recoverable"
            )


class AttentionAdmissionController:
    """Own confirmed V1 placement, residency, and FCFS capacity retry behavior."""

    def __init__(
        self,
        device_ids: tuple[int, ...],
        device_capacities: Mapping[int, int],
        placement: AttentionPlacement,
        residency: AttentionResidency,
    ) -> None:
        if set(device_ids) != set(device_capacities):
            raise ValueError("attention capacities must exactly match mapped device IDs")
        if not isinstance(residency, AttentionResidency):
            raise ValueError(f"unsupported attention residency policy: {residency}")
        self.device_ids = device_ids
        self.residency = residency
        self.devices = {
            device_id: AttentionDeviceState(
                device_id=device_id,
                capacity_tokens=device_capacities[device_id],
            )
            for device_id in device_ids
        }
        self.placement_policy: AttentionPlacementPolicy = (
            build_attention_placement_policy(placement, device_ids)
        )
        self._waiters: deque[CapacityWaiter] = deque()
        self._next_waiter_sequence = 0

    @classmethod
    def from_config(
        cls,
        config: DecodeAttentionMappingConfig,
        devices: Mapping[int, Device],
    ) -> AttentionAdmissionController:
        capacities: dict[int, int] = {}
        for device_id in config.device_ids:
            try:
                device = devices[device_id]
            except KeyError as error:
                raise KeyError(f"unknown attention device {device_id}") from error
            if device.attention_model is None:
                raise ValueError(f"device {device_id} has no decode_attn capability")
            if device.attention_kv_capacity_tokens is None:
                raise ValueError(f"device {device_id} has no attention KV capacity")
            capacities[device_id] = device.attention_kv_capacity_tokens
        return cls(
            device_ids=config.device_ids,
            device_capacities=capacities,
            placement=config.placement,
            residency=config.residency,
        )

    @property
    def waiters(self) -> tuple[CapacityWaiter, ...]:
        return tuple(self._waiters)

    def submit(
        self,
        session: SessionState,
        ready_time_s: float,
        *,
        defer_transfer: bool = False,
    ) -> AttentionAdmission | None:
        ready = _time(ready_time_s, "ready_time_s")
        if not isinstance(defer_transfer, bool):
            raise ValueError("defer_transfer must be a boolean")
        self._validate_new_submission(session, ready)
        if self._waiters:
            self._enqueue(session, ready, defer_transfer)
            return None
        plan = self._select_plan(session)
        if plan is None:
            self._enqueue(session, ready, defer_transfer)
            return None
        return self._commit(plan, session, ready, defer_transfer)

    def retry(self, retry_time_s: float) -> tuple[AttentionAdmission, ...]:
        retry_time = _time(retry_time_s, "retry_time_s")
        admissions: list[AttentionAdmission] = []
        while self._waiters:
            waiter = self._waiters[0]
            if retry_time < waiter.enqueued_time_s:
                raise ValueError(
                    "capacity retry cannot precede the head waiter's enqueue time"
                )
            plan = self._select_plan(waiter.session)
            if plan is None:
                break
            self._waiters.popleft()
            admissions.append(
                self._commit(
                    plan,
                    waiter.session,
                    retry_time,
                    waiter.transfer_pending,
                )
            )
        return tuple(admissions)

    def complete_pool_write(
        self,
        session: SessionState,
        completion_time_s: float,
    ) -> PoolWriteCompletion:
        completion = _time(completion_time_s, "completion_time_s")
        device_id = session.kv.attention.device_id
        if device_id is None or device_id not in self.devices:
            raise ValueError("dirty session is not assigned to a mapped attention device")
        device = self.devices[device_id]
        if device.resident_sessions.get(session.session_id) is not session:
            raise ValueError("dirty session is not registered on its attention device")
        episode = session.current_episode
        if episode is None:
            raise ValueError("dirty session has no decode episode")
        earliest_completion = episode.attention_admitted_time_s or 0.0
        if episode.token_completion_times_s:
            earliest_completion = episode.token_completion_times_s[-1]
        if completion < earliest_completion:
            raise ValueError("pool write cannot complete before decode")
        retain = self.residency is AttentionResidency.LRU_CACHE
        payload_tokens = session.complete_pool_write(retain_attention=retain)
        if not retain:
            del device.resident_sessions[session.session_id]
        self.validate_capacities()
        return PoolWriteCompletion(payload_tokens, self.retry(completion))

    def evict_clean(
        self,
        device_id: int,
        session_id: int,
        eviction_time_s: float,
    ) -> tuple[AttentionAdmission, ...]:
        eviction_time = _time(eviction_time_s, "eviction_time_s")
        if any(waiter.session.session_id == session_id for waiter in self._waiters):
            raise ValueError("cannot evict the protected affinity of a capacity waiter")
        try:
            device = self.devices[device_id]
        except KeyError as error:
            raise KeyError(f"unknown attention device {device_id}") from error
        device.evict_clean(session_id)
        self.validate_capacities()
        return self.retry(eviction_time)

    def validate_capacities(self) -> None:
        for device in self.devices.values():
            device.assert_within_capacity()

    def _validate_new_submission(
        self, session: SessionState, ready_time_s: float
    ) -> None:
        episode = session.current_episode
        if episode is None or episode.phase is not DecodeEpisodePhase.PREFILLED:
            raise ValueError("new attention submission requires a prefilled episode")
        if ready_time_s < episode.turn_ready_time_s:
            raise ValueError("attention submission cannot precede turn readiness")
        if any(waiter.session is session for waiter in self._waiters):
            raise ValueError("session is already waiting for attention capacity")
        attention = session.kv.attention
        if attention.state not in {
            AttentionKVResidency.INACTIVE_CLEAN,
            AttentionKVResidency.EVICTED,
        }:
            raise ValueError("session is already active or draining on attention")
        registered = self._registered_session(session.session_id)
        if attention.state is AttentionKVResidency.INACTIVE_CLEAN:
            if self.residency is not AttentionResidency.LRU_CACHE:
                raise ValueError("active_only cannot retain clean inactive attention KV")
            if registered is not session:
                raise ValueError("cached attention session is not registered")
        elif registered is not None:
            raise ValueError("evicted session cannot remain registered on attention")

    def _enqueue(
        self,
        session: SessionState,
        ready_time_s: float,
        transfer_pending: bool,
    ) -> None:
        session.begin_attention_capacity_wait(ready_time_s)
        self._waiters.append(
            CapacityWaiter(
                sequence_id=self._next_waiter_sequence,
                enqueued_time_s=ready_time_s,
                session=session,
                transfer_pending=transfer_pending,
            )
        )
        self._next_waiter_sequence += 1

    def _select_plan(self, session: SessionState) -> AttentionCapacityPlan | None:
        attention = session.kv.attention
        protected = self._protected_affinities()
        if attention.state is AttentionKVResidency.INACTIVE_CLEAN:
            device_id = attention.device_id
            if device_id is None or device_id not in self.devices:
                raise ValueError("cached attention affinity is outside the mapped devices")
            return self.devices[device_id].plan_admission(
                session,
                protected.get(device_id, frozenset()),
            )

        plans = {
            device_id: device.plan_admission(
                session,
                protected.get(device_id, frozenset()),
            )
            for device_id, device in self.devices.items()
        }
        feasible_ids = frozenset(
            device_id for device_id, plan in plans.items() if plan is not None
        )
        selected = self.placement_policy.select(
            feasible_ids,
            {
                device_id: device.active_context_load
                for device_id, device in self.devices.items()
            },
        )
        return plans[selected] if selected is not None else None

    def _commit(
        self,
        plan: AttentionCapacityPlan,
        session: SessionState,
        admission_time_s: float,
        transfer_pending: bool,
    ) -> AttentionAdmission:
        device = self.devices[plan.device_id]
        for victim_id in plan.evicted_session_ids:
            device._validate_recoverable(device.resident_sessions[victim_id])
        session.activate_attention(
            plan.device_id,
            admission_time_s=admission_time_s,
            transfer_tokens=plan.transfer_tokens,
            cache_hit=plan.cache_hit,
            transfer_pending=transfer_pending,
        )
        for victim_id in plan.evicted_session_ids:
            device.evict_clean(victim_id)
        device.resident_sessions[session.session_id] = session
        self.validate_capacities()
        episode = session.current_episode
        if episode is None:
            raise RuntimeError("admitted session lost its decode episode")
        return AttentionAdmission(
            session_id=session.session_id,
            device_id=plan.device_id,
            admission_time_s=admission_time_s,
            required_final_kv_tokens=plan.required_final_kv_tokens,
            transfer_tokens=plan.transfer_tokens,
            cache_hit=plan.cache_hit,
            evicted_session_ids=plan.evicted_session_ids,
            capacity_wait_s=episode.attention_capacity_wait_s,
            transfer_pending=transfer_pending,
        )

    def _protected_affinities(self) -> dict[int, frozenset[int]]:
        protected: dict[int, set[int]] = {}
        for waiter in self._waiters:
            attention = waiter.session.kv.attention
            if (
                attention.state is AttentionKVResidency.INACTIVE_CLEAN
                and attention.device_id is not None
            ):
                protected.setdefault(attention.device_id, set()).add(
                    waiter.session.session_id
                )
        return {
            device_id: frozenset(session_ids)
            for device_id, session_ids in protected.items()
        }

    def _registered_session(self, session_id: int) -> SessionState | None:
        matches = [
            device.resident_sessions[session_id]
            for device in self.devices.values()
            if session_id in device.resident_sessions
        ]
        if len(matches) > 1:
            raise ValueError(f"session {session_id} is resident on multiple attention devices")
        return matches[0] if matches else None
