from __future__ import annotations

import pytest

from agentic_sim.runtime import (
    AttentionKVResidency,
    DecodeEpisodePhase,
    KVSnapshot,
    SessionState,
    SessionStatus,
)
from agentic_sim.workload import SessionSpec, TurnSpec


def two_turn_spec(session_id: int = 7) -> SessionSpec:
    return SessionSpec(
        session_id=session_id,
        arrival_time_s=0.5,
        initial_context_tokens=10,
        turns=(
            TurnSpec(0, prefill_tokens=3, output_tokens=2, tool_wait_s=0.1),
            TurnSpec(1, prefill_tokens=4, output_tokens=1, tool_wait_s=0.2),
        ),
    )


def start_first_episode(session: SessionState) -> None:
    session.arrive(0.5)
    session.start_episode(0.5)


def test_initial_context_is_warm_pool_state_not_prefill_compute() -> None:
    session = SessionState(two_turn_spec())

    assert session.status is SessionStatus.NOT_ARRIVED
    assert session.current_context_tokens == 10
    assert session.kv.version == 0
    assert session.kv.pool.snapshot == KVSnapshot(10, 0)
    assert session.kv.prefill_snapshot is None
    assert session.kv.attention.state is AttentionKVResidency.EVICTED

    start_first_episode(session)
    assert session.prepare_prefill() == 10
    assert session.apply_prefill() == 3
    assert session.current_context_tokens == 13
    assert session.kv.version == 3
    assert session.kv.prefill_snapshot == KVSnapshot(13, 3)
    assert session.kv.pool.snapshot == KVSnapshot(10, 0)


def test_context_and_logical_copies_advance_once_per_p_and_o_token() -> None:
    session = SessionState(two_turn_spec())
    start_first_episode(session)
    episode = session.current_episode
    assert episode is not None

    session.prepare_prefill()
    session.apply_prefill()
    session.activate_attention(device_id=1)

    assert episode.context_before_prefill == 10
    assert episode.context_at_decode_start == 13
    assert episode.required_final_kv_tokens == 15
    assert session.active_commitment_tokens() == 15
    assert session.kv.attention.snapshot == KVSnapshot(13, 3)

    assert session.emit_decode_token(1.0) == 14
    assert session.active_commitment_tokens() == 15
    assert session.kv.current_snapshot == KVSnapshot(14, 4)
    assert session.kv.attention.snapshot == KVSnapshot(14, 4)
    assert session.kv.prefill_snapshot == KVSnapshot(13, 3)
    assert session.kv.pool.snapshot == KVSnapshot(10, 0)

    assert session.emit_decode_token(1.2) == 15
    assert session.active_commitment_tokens() == 15
    assert session.kv.current_snapshot == KVSnapshot(15, 5)
    assert session.kv.attention.snapshot == KVSnapshot(15, 5)
    assert session.kv.unsynced_prefill_tokens() == 2
    assert session.kv.pool.missing_tokens(session.kv.current_snapshot) == 5


def test_dirty_to_clean_pool_sync_writes_unsynchronized_p_plus_o() -> None:
    session = SessionState(two_turn_spec())
    start_first_episode(session)
    session.prepare_prefill()
    session.apply_prefill()
    session.activate_attention(1)
    session.emit_decode_token(1.0)
    session.emit_decode_token(1.2)
    session.finish_decode()

    assert session.current_episode is not None
    assert session.current_episode.phase is DecodeEpisodePhase.DRAINING_DIRTY
    assert session.kv.attention.state is AttentionKVResidency.DRAINING_DIRTY
    assert session.kv.attention.is_pinned
    with pytest.raises(ValueError, match="only INACTIVE_CLEAN"):
        session.kv.attention.evict()

    assert session.complete_pool_write(retain_attention=True) == 5
    assert session.kv.pool.snapshot == KVSnapshot(15, 5)
    assert session.kv.attention.state is AttentionKVResidency.INACTIVE_CLEAN
    assert not session.kv.attention.is_pinned
    assert session.kv.attention.lru_key(session.session_id) == (1.2, 7)

    completed = session.complete_episode()
    assert completed.pool_write_tokens == 5
    assert session.status is SessionStatus.IDLE


def test_stateful_prefill_copy_loads_only_decode_delta_next_turn() -> None:
    session = SessionState(two_turn_spec())
    start_first_episode(session)
    session.prepare_prefill()
    session.apply_prefill()
    session.activate_attention(1)
    session.emit_decode_token(1.0)
    session.emit_decode_token(1.2)
    session.finish_decode()
    session.complete_pool_write(retain_attention=True)
    session.complete_episode()

    session.start_episode(2.0)
    assert session.prepare_prefill() == 2
    assert session.apply_prefill() == 4
    assert session.current_context_tokens == 19
    assert session.kv.prefill_snapshot == KVSnapshot(19, 9)
    assert session.kv.pool.snapshot == KVSnapshot(15, 5)

    session.activate_attention(1)
    assert session.emit_decode_token(2.5) == 20
    session.finish_decode()
    assert session.complete_pool_write(retain_attention=False) == 5
    assert session.kv.pool.snapshot == KVSnapshot(20, 10)
    assert session.kv.attention.state is AttentionKVResidency.EVICTED
    session.complete_episode()

    assert session.is_finished
    assert session.current_context_tokens == 10 + 3 + 2 + 4 + 1


def test_stateless_prefill_copy_reloads_full_context() -> None:
    session = SessionState(two_turn_spec())
    start_first_episode(session)
    session.prepare_prefill()
    session.apply_prefill()
    session.activate_attention(1)
    session.emit_decode_token(1.0)
    session.emit_decode_token(1.2)
    session.finish_decode()
    session.complete_pool_write(retain_attention=False)
    session.complete_episode()

    session.kv.discard_prefill_copy()
    session.start_episode(2.0)
    assert session.prepare_prefill() == 15


def test_lru_timestamp_refreshes_only_on_decode_token_completion() -> None:
    session = SessionState(two_turn_spec())
    start_first_episode(session)
    session.prepare_prefill()
    session.apply_prefill()
    session.activate_attention(1)

    assert session.kv.attention.last_access_time_s is None
    session.emit_decode_token(3.0)
    assert session.kv.attention.last_access_time_s == 3.0

    snapshot = session.kv.current_snapshot
    with pytest.raises(ValueError, match="monotonic"):
        session.emit_decode_token(2.9)
    assert session.kv.current_snapshot == snapshot
    assert session.current_episode is not None
    assert session.current_episode.generated_tokens == 1

    session.emit_decode_token(3.5)
    assert session.kv.attention.last_access_time_s == 3.5
    with pytest.raises(ValueError, match="monotonic"):
        session.kv.attention.record_decode_token(
            session.kv.current_snapshot,
            completion_time_s=3.4,
        )


def test_cached_attention_affinity_cannot_migrate_and_eviction_clears_it() -> None:
    session = SessionState(two_turn_spec())
    start_first_episode(session)
    session.prepare_prefill()
    session.apply_prefill()
    session.activate_attention(1)
    session.emit_decode_token(1.0)
    session.emit_decode_token(1.1)
    session.finish_decode()
    session.complete_pool_write(retain_attention=True)
    session.complete_episode()

    session.start_episode(2.0)
    session.prepare_prefill()
    session.apply_prefill()
    with pytest.raises(ValueError, match="cannot migrate"):
        session.activate_attention(2)

    session.kv.attention.evict()
    assert session.kv.attention.state is AttentionKVResidency.EVICTED
    assert session.kv.attention.device_id is None
    assert session.kv.attention.snapshot is None


def test_zero_prefill_and_zero_output_do_not_advance_version() -> None:
    spec = SessionSpec(
        session_id=0,
        arrival_time_s=0.0,
        initial_context_tokens=8,
        turns=(TurnSpec(0, 0, 0, 0.0),),
    )
    session = SessionState(spec)
    session.arrive(0.0)
    session.start_episode(0.0)

    assert session.prepare_prefill() == 8
    assert session.apply_prefill() == 0
    session.activate_attention(0)
    session.finish_decode()
    assert session.complete_pool_write(retain_attention=False) == 0
    session.complete_episode()

    assert session.current_context_tokens == 8
    assert session.kv.version == 0
    assert session.kv.pool.snapshot == KVSnapshot(8, 0)


def test_invalid_or_duplicate_state_transitions_are_rejected() -> None:
    session = SessionState(two_turn_spec())
    with pytest.raises(ValueError, match="idle"):
        session.start_episode(0.5)

    start_first_episode(session)
    with pytest.raises(ValueError, match="turn_ready"):
        session.apply_prefill()
    session.prepare_prefill()
    with pytest.raises(ValueError, match="prefill_ready"):
        session.prepare_prefill()
    session.apply_prefill()
    session.activate_attention(1)
    with pytest.raises(ValueError, match="before the oracle output"):
        session.finish_decode()
    session.emit_decode_token(1.0)
    session.emit_decode_token(1.1)
    with pytest.raises(ValueError, match="oracle output"):
        session.emit_decode_token(1.2)
