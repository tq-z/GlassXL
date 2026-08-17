from __future__ import annotations

import pytest

from agentic_sim.config import load_config
from agentic_sim.config.models import AttentionPlacement, AttentionResidency
from agentic_sim.hardware.topology import Topology
from agentic_sim.runtime import (
    AttentionAdmissionController,
    AttentionKVResidency,
    DecodeEpisodePhase,
    KVSnapshot,
    SessionState,
    SessionStatus,
)
from agentic_sim.workload import SessionSpec, TurnSpec


def controller(
    capacities: dict[int, int],
    *,
    placement: AttentionPlacement = AttentionPlacement.ROUND_ROBIN,
    residency: AttentionResidency = AttentionResidency.ACTIVE_ONLY,
) -> AttentionAdmissionController:
    return AttentionAdmissionController(
        device_ids=tuple(capacities),
        device_capacities=capacities,
        placement=placement,
        residency=residency,
    )


def session_with_turns(
    session_id: int,
    context_tokens: int,
    turns: tuple[TurnSpec, ...],
) -> SessionState:
    return SessionState(
        SessionSpec(
            session_id=session_id,
            arrival_time_s=0.0,
            initial_context_tokens=context_tokens,
            turns=turns,
        )
    )


def one_turn_session(
    session_id: int,
    context_tokens: int,
    *,
    prefill_tokens: int = 0,
    output_tokens: int = 0,
) -> SessionState:
    return session_with_turns(
        session_id,
        context_tokens,
        (TurnSpec(0, prefill_tokens, output_tokens, 0.0),),
    )


def prepare_turn(session: SessionState, ready_time_s: float = 0.0) -> None:
    if session.status is SessionStatus.NOT_ARRIVED:
        session.arrive(0.0)
    session.start_episode(ready_time_s)
    session.prepare_prefill()
    session.apply_prefill()


def finish_and_clean(
    admission: AttentionAdmissionController,
    session: SessionState,
    *,
    first_token_time_s: float,
    pool_completion_time_s: float,
) -> None:
    episode = session.current_episode
    assert episode is not None
    for offset in range(episode.turn.output_tokens):
        session.emit_decode_token(first_token_time_s + offset * 0.01)
    session.finish_decode()
    completion = admission.complete_pool_write(session, pool_completion_time_s)
    assert completion.admissions == ()
    session.complete_episode()


def test_round_robin_scans_feasible_devices_and_advances_after_success() -> None:
    admission = controller({1: 10, 2: 20})
    large = one_turn_session(0, 12)
    small = one_turn_session(1, 5)
    prepare_turn(large)
    prepare_turn(small)

    large_result = admission.submit(large, 0.0)
    small_result = admission.submit(small, 0.0)

    assert large_result is not None
    assert large_result.device_id == 2
    assert large_result.required_final_kv_tokens == 12
    assert large_result.transfer_tokens == 12
    assert small_result is not None
    assert small_result.device_id == 1
    admission.validate_capacities()


def test_least_active_context_uses_only_feasible_devices_and_ties_by_id() -> None:
    admission = controller(
        {1: 20, 2: 10},
        placement=AttentionPlacement.LEAST_ACTIVE_CONTEXT,
    )
    first = one_turn_session(0, 5)
    second = one_turn_session(1, 10)
    third = one_turn_session(2, 6)
    for session in (first, second, third):
        prepare_turn(session)

    first_result = admission.submit(first, 0.0)
    second_result = admission.submit(second, 0.0)
    third_result = admission.submit(third, 0.0)

    assert first_result is not None and first_result.device_id == 1
    assert second_result is not None and second_result.device_id == 2
    assert third_result is not None and third_result.device_id == 1


def test_capacity_wait_retries_after_dirty_release_and_is_included_in_ttft() -> None:
    admission = controller({1: 10})
    occupying = one_turn_session(0, 10)
    waiting = one_turn_session(1, 5, output_tokens=1)
    prepare_turn(occupying)
    prepare_turn(waiting, ready_time_s=1.0)
    assert admission.submit(occupying, 0.0) is not None

    assert admission.submit(waiting, 1.0) is None
    assert waiting.status is SessionStatus.WAITING_ATTN_CAPACITY
    assert waiting.current_episode is not None
    assert (
        waiting.current_episode.phase
        is DecodeEpisodePhase.WAITING_ATTN_CAPACITY
    )
    assert [item.session.session_id for item in admission.waiters] == [1]

    occupying.finish_decode()
    completion = admission.complete_pool_write(occupying, 5.0)
    assert len(completion.admissions) == 1
    retried = completion.admissions[0]
    assert retried.session_id == 1
    assert retried.capacity_wait_s == pytest.approx(4.0)
    assert waiting.status is SessionStatus.EPISODE_ACTIVE

    waiting.emit_decode_token(6.0)
    assert waiting.current_episode is not None
    assert waiting.current_episode.attention_capacity_wait_s == pytest.approx(4.0)
    assert waiting.current_episode.ttft_s == pytest.approx(5.0)


def test_oracle_commitment_stays_constant_and_prevents_capacity_overflow() -> None:
    admission = controller({1: 10})
    session = one_turn_session(0, 5, output_tokens=5)
    prepare_turn(session)
    result = admission.submit(session, 0.0)

    assert result is not None
    assert result.required_final_kv_tokens == 10
    device = admission.devices[1]
    for step in range(5):
        session.emit_decode_token(1.0 + step)
        assert device.active_commitment_tokens == 10
        assert device.reserved_or_used_tokens == 10
        assert device.resident_used_tokens == 6 + step
        admission.validate_capacities()


def test_active_and_dirty_sessions_are_never_lru_eviction_candidates() -> None:
    admission = controller({1: 10}, residency=AttentionResidency.LRU_CACHE)
    pinned = one_turn_session(0, 10)
    blocked = one_turn_session(1, 1)
    prepare_turn(pinned)
    prepare_turn(blocked)
    assert admission.submit(pinned, 0.0) is not None

    assert admission.submit(blocked, 0.0) is None
    assert pinned.kv.attention.state is AttentionKVResidency.ACTIVE
    with pytest.raises(ValueError, match="only clean inactive"):
        admission.evict_clean(1, pinned.session_id, 0.5)

    pinned.finish_decode()
    assert pinned.kv.attention.state is AttentionKVResidency.DRAINING_DIRTY
    assert admission.retry(0.5) == ()
    assert blocked.status is SessionStatus.WAITING_ATTN_CAPACITY


def test_lru_prunes_only_clean_recoverable_entries_with_session_id_ties() -> None:
    admission = controller({1: 12}, residency=AttentionResidency.LRU_CACHE)
    higher_id = one_turn_session(2, 3, output_tokens=1)
    lower_id = one_turn_session(1, 3, output_tokens=1)

    for session in (higher_id, lower_id):
        prepare_turn(session)
        assert admission.submit(session, 0.0) is not None
        finish_and_clean(
            admission,
            session,
            first_token_time_s=1.0,
            pool_completion_time_s=1.1,
        )

    incoming = one_turn_session(3, 6)
    prepare_turn(incoming, 2.0)
    result = admission.submit(incoming, 2.0)

    assert result is not None
    assert result.evicted_session_ids == (1,)
    assert lower_id.kv.attention.state is AttentionKVResidency.EVICTED
    assert higher_id.kv.attention.state is AttentionKVResidency.INACTIVE_CLEAN
    assert lower_id.kv.pool.snapshot == lower_id.kv.current_snapshot
    admission.validate_capacities()


def test_lru_cache_hit_keeps_affinity_and_restores_only_new_prefill_tokens() -> None:
    admission = controller(
        {1: 20, 2: 20},
        residency=AttentionResidency.LRU_CACHE,
    )
    session = session_with_turns(
        0,
        3,
        (
            TurnSpec(0, 1, 1, 0.0),
            TurnSpec(1, 2, 1, 0.0),
        ),
    )
    prepare_turn(session)
    first = admission.submit(session, 0.0)
    assert first is not None and first.device_id == 1
    assert not first.cache_hit and first.transfer_tokens == 4
    finish_and_clean(
        admission,
        session,
        first_token_time_s=1.0,
        pool_completion_time_s=1.1,
    )

    prepare_turn(session, 2.0)
    second = admission.submit(session, 2.0)

    assert second is not None
    assert second.cache_hit
    assert second.device_id == 1
    assert second.transfer_tokens == 2
    assert second.required_final_kv_tokens == 8
    assert session.kv.attention.state is AttentionKVResidency.ACTIVE


def test_lru_affinity_waits_without_migrating_when_commitment_no_longer_fits() -> None:
    admission = controller(
        {1: 8, 2: 100},
        residency=AttentionResidency.LRU_CACHE,
    )
    session = session_with_turns(
        0,
        8,
        (
            TurnSpec(0, 0, 0, 0.0),
            TurnSpec(1, 0, 1, 0.0),
        ),
    )
    prepare_turn(session)
    first = admission.submit(session, 0.0)
    assert first is not None and first.device_id == 1
    finish_and_clean(
        admission,
        session,
        first_token_time_s=0.0,
        pool_completion_time_s=0.1,
    )

    prepare_turn(session, 1.0)
    assert admission.submit(session, 1.0) is None
    assert session.kv.attention.device_id == 1
    assert session.kv.attention.state is AttentionKVResidency.INACTIVE_CLEAN
    assert admission.retry(2.0) == ()
    with pytest.raises(ValueError, match="protected affinity"):
        admission.evict_clean(1, session.session_id, 2.0)


def test_active_only_releases_after_pool_write_and_reruns_full_restore_placement() -> None:
    admission = controller({1: 20, 2: 20})
    session = session_with_turns(
        0,
        3,
        (
            TurnSpec(0, 1, 0, 0.0),
            TurnSpec(1, 2, 0, 0.0),
        ),
    )
    prepare_turn(session)
    first = admission.submit(session, 0.0)
    assert first is not None and first.device_id == 1

    session.finish_decode()
    admission.complete_pool_write(session, 1.0)
    assert session.kv.attention.state is AttentionKVResidency.EVICTED
    assert session.session_id not in admission.devices[1].resident_sessions
    session.complete_episode()

    prepare_turn(session, 2.0)
    second = admission.submit(session, 2.0)

    assert second is not None
    assert second.device_id == 2
    assert not second.cache_hit
    assert second.transfer_tokens == session.current_context_tokens == 6


def test_lru_eviction_clears_affinity_and_later_requires_full_context_restore() -> None:
    admission = controller(
        {1: 20, 2: 20},
        residency=AttentionResidency.LRU_CACHE,
    )
    session = session_with_turns(
        0,
        3,
        (
            TurnSpec(0, 1, 0, 0.0),
            TurnSpec(1, 2, 0, 0.0),
        ),
    )
    prepare_turn(session)
    first = admission.submit(session, 0.0)
    assert first is not None and first.device_id == 1
    session.finish_decode()
    admission.complete_pool_write(session, 0.5)
    session.complete_episode()

    assert admission.evict_clean(1, session.session_id, 1.0) == ()
    assert session.kv.attention.state is AttentionKVResidency.EVICTED
    assert session.kv.attention.device_id is None

    prepare_turn(session, 2.0)
    second = admission.submit(session, 2.0)

    assert second is not None
    assert not second.cache_hit
    assert second.device_id == 2
    assert second.transfer_tokens == session.current_context_tokens == 6


def test_pruning_rejects_a_cached_copy_that_is_not_recoverable() -> None:
    admission = controller({1: 10}, residency=AttentionResidency.LRU_CACHE)
    session = one_turn_session(0, 3)
    prepare_turn(session)
    assert admission.submit(session, 0.0) is not None
    session.finish_decode()
    admission.complete_pool_write(session, 0.5)

    session.kv.pool.snapshot = KVSnapshot(0, 0)
    with pytest.raises(ValueError, match="before its cached snapshot is recoverable"):
        admission.evict_clean(1, session.session_id, 1.0)
    assert session.kv.attention.state is AttentionKVResidency.INACTIVE_CLEAN
    assert session.session_id in admission.devices[1].resident_sessions


def test_invalid_admission_times_do_not_advance_cursor_or_drop_waiters() -> None:
    admission = controller({1: 10, 2: 10})
    feasible = one_turn_session(0, 5)
    prepare_turn(feasible, 1.0)

    with pytest.raises(ValueError, match="cannot precede turn readiness"):
        admission.submit(feasible, 0.5)
    result = admission.submit(feasible, 1.0)
    assert result is not None and result.device_id == 1

    blocked = one_turn_session(1, 20)
    prepare_turn(blocked, 2.0)
    assert admission.submit(blocked, 2.0) is None
    with pytest.raises(ValueError, match="cannot precede.*enqueue"):
        admission.retry(1.5)
    assert [waiter.session.session_id for waiter in admission.waiters] == [1]
    assert blocked.status is SessionStatus.WAITING_ATTN_CAPACITY


def test_capacity_wait_queue_is_strictly_fcfs() -> None:
    admission = controller({1: 10})
    oversized_head = one_turn_session(0, 20)
    feasible_later = one_turn_session(1, 1)
    prepare_turn(oversized_head)
    prepare_turn(feasible_later)

    assert admission.submit(oversized_head, 0.0) is None
    assert admission.submit(feasible_later, 0.0) is None
    assert admission.retry(1.0) == ()
    assert [
        (waiter.sequence_id, waiter.session.session_id)
        for waiter in admission.waiters
    ] == [
        (0, 0),
        (1, 1),
    ]


def test_controller_builds_from_typed_mapping_and_hardware_devices() -> None:
    config = load_config(
        "configs/example/hardware.yaml",
        "configs/example/workload.yaml",
        "configs/example/mapping.yaml",
    )
    topology = Topology.from_config(config.hardware)

    admission = AttentionAdmissionController.from_config(
        config.mapping.decode_attn,
        topology.devices,
    )

    assert admission.device_ids == (1, 2)
    assert admission.residency is AttentionResidency.LRU_CACHE
    assert {
        device_id: device.capacity_tokens
        for device_id, device in admission.devices.items()
    } == {
        1: 262144,
        2: 262144,
    }
