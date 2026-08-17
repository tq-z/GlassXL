"""Milestone 3 session, decode-episode, and logical KV-copy state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from agentic_sim.workload import SessionSpec, TurnSpec


def _nonnegative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _time(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return result


class SessionStatus(StrEnum):
    NOT_ARRIVED = "not_arrived"
    IDLE = "idle"
    EPISODE_ACTIVE = "episode_active"
    WAITING_ATTN_CAPACITY = "waiting_attn_capacity"
    FINISHED = "finished"


class DecodeEpisodePhase(StrEnum):
    TURN_READY = "turn_ready"
    PREFILL_READY = "prefill_ready"
    PREFILLED = "prefilled"
    WAITING_ATTN_CAPACITY = "waiting_attn_capacity"
    ACTIVE = "active"
    DRAINING_DIRTY = "draining_dirty"
    POOL_SYNCED = "pool_synced"


class AttentionKVResidency(StrEnum):
    ACTIVE = "active"
    DRAINING_DIRTY = "draining_dirty"
    INACTIVE_CLEAN = "inactive_clean"
    EVICTED = "evicted"


@dataclass(frozen=True)
class KVSnapshot:
    """One logical KV copy at a token-counted version."""

    context_tokens: int
    version: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.context_tokens, "context_tokens")
        _nonnegative_integer(self.version, "version")


@dataclass
class PoolCopyState:
    """The recoverable KV-pool snapshot for one session."""

    snapshot: KVSnapshot

    def missing_tokens(self, current: KVSnapshot) -> int:
        if current.version < self.snapshot.version:
            raise ValueError("current KV version cannot precede the pool copy")
        if current.context_tokens < self.snapshot.context_tokens:
            raise ValueError("current context cannot be smaller than the pool copy")
        version_delta = current.version - self.snapshot.version
        token_delta = current.context_tokens - self.snapshot.context_tokens
        if version_delta != token_delta:
            raise ValueError("KV version and token deltas must agree")
        return token_delta

    def synchronize(self, current: KVSnapshot) -> int:
        payload_tokens = self.missing_tokens(current)
        self.snapshot = current
        return payload_tokens


@dataclass
class AttentionKVState:
    """Attention-side residency, affinity, snapshot, and deterministic LRU time."""

    state: AttentionKVResidency = AttentionKVResidency.EVICTED
    device_id: int | None = None
    snapshot: KVSnapshot | None = None
    last_access_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.state is AttentionKVResidency.EVICTED:
            if self.device_id is not None or self.snapshot is not None:
                raise ValueError("evicted attention KV cannot retain a device or snapshot")
        elif self.device_id is None or self.snapshot is None:
            raise ValueError("resident attention KV requires a device and snapshot")
        if self.device_id is not None:
            _nonnegative_integer(self.device_id, "device_id")
        if self.last_access_time_s is not None:
            _time(self.last_access_time_s, "last_access_time_s")

    @property
    def is_pinned(self) -> bool:
        return self.state in {
            AttentionKVResidency.ACTIVE,
            AttentionKVResidency.DRAINING_DIRTY,
        }

    def activate(self, device_id: int, current: KVSnapshot) -> None:
        _nonnegative_integer(device_id, "device_id")
        if self.state in {
            AttentionKVResidency.ACTIVE,
            AttentionKVResidency.DRAINING_DIRTY,
        }:
            raise ValueError(f"cannot activate attention KV from {self.state.value}")
        if (
            self.state is AttentionKVResidency.INACTIVE_CLEAN
            and self.device_id != device_id
        ):
            raise ValueError("inactive cached attention KV cannot migrate devices")
        if self.snapshot is not None:
            version_delta = current.version - self.snapshot.version
            token_delta = current.context_tokens - self.snapshot.context_tokens
            if version_delta < 0 or token_delta != version_delta:
                raise ValueError(
                    "attention KV activation requires monotonic matching token/version deltas"
                )
        self.state = AttentionKVResidency.ACTIVE
        self.device_id = device_id
        self.snapshot = current

    def restore_tokens(self, current: KVSnapshot) -> int:
        """Return the KV payload needed to restore the current attention copy."""

        if self.state is AttentionKVResidency.EVICTED:
            return current.context_tokens
        if self.state is not AttentionKVResidency.INACTIVE_CLEAN:
            raise ValueError("attention restore requires clean inactive or evicted KV")
        if self.snapshot is None:
            raise ValueError("clean inactive attention KV requires a snapshot")
        version_delta = current.version - self.snapshot.version
        token_delta = current.context_tokens - self.snapshot.context_tokens
        if version_delta < 0 or token_delta != version_delta:
            raise ValueError(
                "attention restore requires monotonic matching token/version deltas"
            )
        return token_delta

    def record_decode_token(self, current: KVSnapshot, completion_time_s: float) -> None:
        if self.state is not AttentionKVResidency.ACTIVE:
            raise ValueError("decode tokens require ACTIVE attention residency")
        completion = _time(completion_time_s, "completion_time_s")
        if self.last_access_time_s is not None and completion < self.last_access_time_s:
            raise ValueError("decode completion times must be monotonic")
        if self.snapshot is None:
            raise ValueError("ACTIVE attention residency requires a KV snapshot")
        if (
            current.version != self.snapshot.version + 1
            or current.context_tokens != self.snapshot.context_tokens + 1
        ):
            raise ValueError("each decode step must append exactly one attention KV token")
        self.snapshot = current
        self.last_access_time_s = completion

    def begin_draining(self) -> None:
        if self.state is not AttentionKVResidency.ACTIVE:
            raise ValueError("only ACTIVE attention KV can begin draining")
        self.state = AttentionKVResidency.DRAINING_DIRTY

    def mark_pool_clean(self, *, retain: bool) -> None:
        if self.state is not AttentionKVResidency.DRAINING_DIRTY:
            raise ValueError("only DRAINING_DIRTY attention KV can become clean")
        if retain:
            self.state = AttentionKVResidency.INACTIVE_CLEAN
        else:
            self._clear_evicted()

    def evict(self) -> None:
        if self.state is AttentionKVResidency.EVICTED:
            return
        if self.state is not AttentionKVResidency.INACTIVE_CLEAN:
            raise ValueError("only INACTIVE_CLEAN attention KV can be evicted")
        self._clear_evicted()

    def lru_key(self, session_id: int) -> tuple[float, int]:
        if self.state is not AttentionKVResidency.INACTIVE_CLEAN:
            raise ValueError("only INACTIVE_CLEAN attention KV participates in LRU")
        _nonnegative_integer(session_id, "session_id")
        access_time = (
            self.last_access_time_s
            if self.last_access_time_s is not None
            else float("-inf")
        )
        return access_time, session_id

    def _clear_evicted(self) -> None:
        self.state = AttentionKVResidency.EVICTED
        self.device_id = None
        self.snapshot = None
        self.last_access_time_s = None


@dataclass
class SessionKVState:
    """Canonical context and all logical KV copies for one session."""

    context_tokens: int
    version: int
    pool: PoolCopyState
    prefill_snapshot: KVSnapshot | None = None
    attention: AttentionKVState = field(default_factory=AttentionKVState)

    @classmethod
    def warm_start(cls, initial_context_tokens: int) -> SessionKVState:
        _nonnegative_integer(initial_context_tokens, "initial_context_tokens")
        initial = KVSnapshot(initial_context_tokens, version=0)
        return cls(
            context_tokens=initial_context_tokens,
            version=0,
            pool=PoolCopyState(initial),
        )

    @property
    def current_snapshot(self) -> KVSnapshot:
        return KVSnapshot(self.context_tokens, self.version)

    def append_tokens(self, tokens: int) -> KVSnapshot:
        """Advance canonical context/version exactly once per appended token."""

        _nonnegative_integer(tokens, "tokens")
        self.context_tokens += tokens
        self.version += tokens
        return self.current_snapshot

    def prepare_prefill_copy(self) -> int:
        """Synchronize a transient/retained prefill copy from the current pool."""

        current = self.current_snapshot
        if self.pool.snapshot != current:
            raise ValueError("prefill preparation requires a current recoverable pool copy")
        if self.prefill_snapshot is None:
            load_tokens = current.context_tokens
        else:
            if self.prefill_snapshot.version > current.version:
                raise ValueError("prefill KV version cannot exceed current version")
            load_tokens = current.version - self.prefill_snapshot.version
            if current.context_tokens - self.prefill_snapshot.context_tokens != load_tokens:
                raise ValueError("prefill KV version and token deltas must agree")
        self.prefill_snapshot = current
        return load_tokens

    def apply_prefill_tokens(self, tokens: int) -> KVSnapshot:
        if self.prefill_snapshot != self.current_snapshot:
            raise ValueError("prefill copy must be synchronized before prefill compute")
        current = self.append_tokens(tokens)
        self.prefill_snapshot = current
        return current

    def discard_prefill_copy(self) -> None:
        self.prefill_snapshot = None

    def unsynced_prefill_tokens(self) -> int:
        if self.prefill_snapshot is None:
            return self.context_tokens
        delta = self.version - self.prefill_snapshot.version
        if self.context_tokens - self.prefill_snapshot.context_tokens != delta:
            raise ValueError("prefill KV version and token deltas must agree")
        return delta


@dataclass
class DecodeEpisodeState:
    """Mutable state for exactly one oracle decode episode."""

    turn: TurnSpec
    turn_ready_time_s: float
    context_before_prefill: int
    phase: DecodeEpisodePhase = DecodeEpisodePhase.TURN_READY
    prefill_load_tokens: int | None = None
    context_at_decode_start: int | None = None
    generated_tokens: int = 0
    pool_write_tokens: int | None = None
    token_completion_times_s: list[float] = field(default_factory=list)
    capacity_wait_started_s: float | None = None
    attention_capacity_wait_s: float = 0.0
    attention_admitted_time_s: float | None = None
    attention_ready_time_s: float | None = None
    attention_transfer_tokens: int | None = None
    attention_cache_hit: bool | None = None

    def __post_init__(self) -> None:
        _time(self.turn_ready_time_s, "turn_ready_time_s")
        _nonnegative_integer(self.context_before_prefill, "context_before_prefill")

    @property
    def remaining_output_tokens(self) -> int:
        return self.turn.output_tokens - self.generated_tokens

    @property
    def required_final_kv_tokens(self) -> int:
        if self.context_at_decode_start is None:
            raise ValueError("decode has not started")
        return self.context_at_decode_start + self.turn.output_tokens

    @property
    def ttft_s(self) -> float:
        if not self.token_completion_times_s:
            raise ValueError("TTFT is unavailable before the first token completes")
        return self.token_completion_times_s[0] - self.turn_ready_time_s


@dataclass
class SessionState:
    """Long-lived mutable runtime state for one immutable workload session."""

    spec: SessionSpec
    status: SessionStatus = field(default=SessionStatus.NOT_ARRIVED, init=False)
    kv: SessionKVState = field(init=False)
    next_turn_index: int = field(default=0, init=False)
    current_episode: DecodeEpisodeState | None = field(default=None, init=False)
    completed_episodes: list[DecodeEpisodeState] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.kv = SessionKVState.warm_start(self.spec.initial_context_tokens)

    @property
    def session_id(self) -> int:
        return self.spec.session_id

    @property
    def current_context_tokens(self) -> int:
        return self.kv.context_tokens

    @property
    def is_finished(self) -> bool:
        return self.status is SessionStatus.FINISHED

    def arrive(self, arrival_time_s: float) -> None:
        arrival = _time(arrival_time_s, "arrival_time_s")
        if self.status is not SessionStatus.NOT_ARRIVED:
            raise ValueError("session has already arrived")
        if arrival < self.spec.arrival_time_s:
            raise ValueError("session cannot arrive before its workload timestamp")
        self.status = SessionStatus.IDLE

    def start_episode(self, turn_ready_time_s: float) -> DecodeEpisodeState:
        ready = _time(turn_ready_time_s, "turn_ready_time_s")
        if self.status is not SessionStatus.IDLE:
            raise ValueError("session must be idle to start an episode")
        if self.current_episode is not None:
            raise ValueError("session already has an active episode")
        if self.next_turn_index >= len(self.spec.turns):
            raise ValueError("session has no remaining turns")
        if ready < self.spec.arrival_time_s:
            raise ValueError("turn cannot be ready before session arrival")
        episode = DecodeEpisodeState(
            turn=self.spec.turns[self.next_turn_index],
            turn_ready_time_s=ready,
            context_before_prefill=self.current_context_tokens,
        )
        self.current_episode = episode
        self.status = SessionStatus.EPISODE_ACTIVE
        return episode

    def prepare_prefill(self) -> int:
        episode = self._episode_in(DecodeEpisodePhase.TURN_READY)
        load_tokens = self.kv.prepare_prefill_copy()
        episode.prefill_load_tokens = load_tokens
        episode.phase = DecodeEpisodePhase.PREFILL_READY
        return load_tokens

    def apply_prefill(self) -> int:
        episode = self._episode_in(DecodeEpisodePhase.PREFILL_READY)
        self.kv.apply_prefill_tokens(episode.turn.prefill_tokens)
        episode.phase = DecodeEpisodePhase.PREFILLED
        return episode.turn.prefill_tokens

    def begin_attention_capacity_wait(self, wait_started_s: float) -> None:
        episode = self._episode_in(DecodeEpisodePhase.PREFILLED)
        started = _time(wait_started_s, "wait_started_s")
        if started < episode.turn_ready_time_s:
            raise ValueError("attention capacity wait cannot precede turn readiness")
        episode.capacity_wait_started_s = started
        episode.phase = DecodeEpisodePhase.WAITING_ATTN_CAPACITY
        self.status = SessionStatus.WAITING_ATTN_CAPACITY

    def activate_attention(
        self,
        device_id: int,
        *,
        admission_time_s: float | None = None,
        transfer_tokens: int | None = None,
        cache_hit: bool | None = None,
    ) -> None:
        if self.current_episode is None:
            raise ValueError("session has no active episode")
        episode = self.current_episode
        if episode.phase not in {
            DecodeEpisodePhase.PREFILLED,
            DecodeEpisodePhase.WAITING_ATTN_CAPACITY,
        }:
            raise ValueError(
                "episode phase must be prefilled or waiting_attn_capacity, "
                f"got {episode.phase.value}"
            )
        admitted = (
            _time(admission_time_s, "admission_time_s")
            if admission_time_s is not None
            else None
        )
        if episode.phase is DecodeEpisodePhase.WAITING_ATTN_CAPACITY:
            if admitted is None or episode.capacity_wait_started_s is None:
                raise ValueError("capacity-waiting admission requires an admission time")
            if admitted < episode.capacity_wait_started_s:
                raise ValueError("admission cannot precede the capacity wait")
            episode.attention_capacity_wait_s = (
                admitted - episode.capacity_wait_started_s
            )
        if admitted is not None and admitted < episode.turn_ready_time_s:
            raise ValueError("attention admission cannot precede turn readiness")
        if transfer_tokens is not None:
            _nonnegative_integer(transfer_tokens, "transfer_tokens")
        self.kv.attention.activate(device_id, self.kv.current_snapshot)
        episode.context_at_decode_start = self.current_context_tokens
        episode.attention_admitted_time_s = admitted
        episode.attention_ready_time_s = admitted
        episode.attention_transfer_tokens = transfer_tokens
        episode.attention_cache_hit = cache_hit
        episode.phase = DecodeEpisodePhase.ACTIVE
        self.status = SessionStatus.EPISODE_ACTIVE

    def complete_attention_transfer(self, completion_time_s: float) -> None:
        episode = self._episode_in(DecodeEpisodePhase.ACTIVE)
        completion = _time(completion_time_s, "completion_time_s")
        if episode.attention_admitted_time_s is None:
            raise ValueError("attention transfer requires a timed capacity admission")
        if episode.attention_transfer_tokens is None:
            raise ValueError("attention transfer payload is unavailable")
        if completion < episode.attention_admitted_time_s:
            raise ValueError("attention transfer cannot precede capacity admission")
        episode.attention_ready_time_s = completion

    def emit_decode_token(self, completion_time_s: float) -> int:
        episode = self._episode_in(DecodeEpisodePhase.ACTIVE)
        if episode.remaining_output_tokens <= 0:
            raise ValueError("decode episode has already emitted its oracle output")
        completion = _time(completion_time_s, "completion_time_s")
        if (
            episode.attention_ready_time_s is not None
            and completion < episode.attention_ready_time_s
        ):
            raise ValueError("decode completion cannot precede attention KV readiness")
        last_access = self.kv.attention.last_access_time_s
        if last_access is not None and completion < last_access:
            raise ValueError("decode completion times must be monotonic")
        current = self.kv.append_tokens(1)
        self.kv.attention.record_decode_token(current, completion)
        episode.generated_tokens += 1
        episode.token_completion_times_s.append(completion)
        return self.current_context_tokens

    def finish_decode(self) -> None:
        episode = self._episode_in(DecodeEpisodePhase.ACTIVE)
        if episode.remaining_output_tokens != 0:
            raise ValueError("cannot finish decode before the oracle output is emitted")
        self.kv.attention.begin_draining()
        episode.phase = DecodeEpisodePhase.DRAINING_DIRTY

    def complete_pool_write(self, *, retain_attention: bool) -> int:
        episode = self._episode_in(DecodeEpisodePhase.DRAINING_DIRTY)
        current = self.kv.current_snapshot
        payload_tokens = self.kv.pool.missing_tokens(current)
        expected_tokens = episode.turn.prefill_tokens + episode.generated_tokens
        if payload_tokens != expected_tokens:
            raise ValueError(
                f"pool payload {payload_tokens} does not match unsynchronized P+O "
                f"{expected_tokens}"
            )
        self.kv.pool.synchronize(current)
        self.kv.attention.mark_pool_clean(retain=retain_attention)
        episode.pool_write_tokens = payload_tokens
        episode.phase = DecodeEpisodePhase.POOL_SYNCED
        return payload_tokens

    def complete_episode(self) -> DecodeEpisodeState:
        episode = self._episode_in(DecodeEpisodePhase.POOL_SYNCED)
        self.completed_episodes.append(episode)
        self.current_episode = None
        self.next_turn_index += 1
        self.status = (
            SessionStatus.FINISHED
            if self.next_turn_index == len(self.spec.turns)
            else SessionStatus.IDLE
        )
        return episode

    def active_commitment_tokens(self) -> int:
        episode = self._episode_in(DecodeEpisodePhase.ACTIVE)
        commitment = self.current_context_tokens + episode.remaining_output_tokens
        if commitment != episode.required_final_kv_tokens:
            raise ValueError("active oracle commitment must remain constant")
        return commitment

    def _episode_in(self, phase: DecodeEpisodePhase) -> DecodeEpisodeState:
        if self.current_episode is None:
            raise ValueError("session has no active episode")
        if self.current_episode.phase is not phase:
            raise ValueError(
                f"episode phase must be {phase.value}, got {self.current_episode.phase.value}"
            )
        return self.current_episode
