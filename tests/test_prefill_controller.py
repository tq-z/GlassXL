from __future__ import annotations

import pytest

from agentic_sim.config import load_config
from agentic_sim.config.models import (
    AttentionPlacement,
    AttentionResidency,
    PrefillMappingConfig,
    PrefillPlacement,
    PrefillState,
)
from agentic_sim.hardware.compute_models import (
    AggregatePrefillLatencyModel,
    LinearAttentionLatencyModel,
)
from agentic_sim.hardware.device import Device, DeviceOperation
from agentic_sim.hardware.interconnect import (
    FFNAllToAllInterconnect,
    KVP2PInterconnect,
    StartupMinPackageCollectiveModel,
    StartupThroughputP2PModel,
)
from agentic_sim.hardware.kv_pool import KVPool, SymmetricKVPoolLatencyModel
from agentic_sim.hardware.topology import Topology
from agentic_sim.mapping.prefill import (
    RoundRobinPrefillPlacement,
    StickyRoundRobinPrefillPlacement,
)
from agentic_sim.runtime import (
    AttentionAdmissionController,
    AttentionKVResidency,
    KVSnapshot,
    PrefillController,
    SessionState,
    SessionStatus,
)
from agentic_sim.workload import SessionSpec, TurnSpec


def make_session(
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


def start_turn(session: SessionState, ready_time_s: float) -> None:
    if session.status is SessionStatus.NOT_ARRIVED:
        session.arrive(0.0)
    session.start_episode(ready_time_s)


def make_controller(
    state: PrefillState,
    *,
    residency: AttentionResidency,
    prefill_ids: tuple[int, ...] = (0,),
    attention_capacities: dict[int, int] | None = None,
) -> tuple[PrefillController, AttentionAdmissionController, Topology]:
    capacities = attention_capacities or {1: 100}
    devices = {
        device_id: Device(
            device_id,
            prefill_model=AggregatePrefillLatencyModel(10.0),
        )
        for device_id in prefill_ids
    }
    devices.update(
        {
            device_id: Device(
                device_id,
                attention_model=LinearAttentionLatencyModel(10.0),
                attention_kv_capacity_tokens=capacity,
            )
            for device_id, capacity in capacities.items()
        }
    )
    topology = Topology(
        devices=devices,
        ffn_all_to_all=FFNAllToAllInterconnect(
            StartupMinPackageCollectiveModel(0.0, 0, 10.0)
        ),
        kv_p2p=KVP2PInterconnect(
            StartupThroughputP2PModel(0.5, 10.0)
        ),
        kv_pool=KVPool(SymmetricKVPoolLatencyModel(1.0, 10.0)),
    )
    attention = AttentionAdmissionController(
        device_ids=tuple(capacities),
        device_capacities=capacities,
        placement=AttentionPlacement.ROUND_ROBIN,
        residency=residency,
    )
    placement = (
        PrefillPlacement.STICKY_ROUND_ROBIN
        if state is PrefillState.STATEFUL
        else PrefillPlacement.ROUND_ROBIN
    )
    prefill = PrefillController(
        topology,
        PrefillMappingConfig(prefill_ids, state, placement),
        attention,
    )
    return prefill, attention, topology


def finish_episode(
    attention: AttentionAdmissionController,
    session: SessionState,
    token_times: tuple[float, ...],
    pool_completion_time_s: float,
) -> None:
    for token_time in token_times:
        session.emit_decode_token(token_time)
    session.finish_decode()
    completion = attention.complete_pool_write(session, pool_completion_time_s)
    assert completion.admissions == ()
    session.complete_episode()


def test_stateful_prefill_loads_warm_context_then_only_decode_delta() -> None:
    prefill, attention, _ = make_controller(
        PrefillState.STATEFUL,
        residency=AttentionResidency.LRU_CACHE,
    )
    session = make_session(
        0,
        10,
        (
            TurnSpec(0, 4, 3, 0.0),
            TurnSpec(1, 2, 0, 0.0),
        ),
    )
    start_turn(session, 0.0)
    assert prefill.enqueue(session, 0.0) == 0

    first = prefill.process_ready(0.0)[0]

    assert len(first.sessions) == 1
    assert first.sessions[0].context_tokens == 10
    assert first.sessions[0].load_tokens == 10
    assert first.sessions[0].unsynced_decode_tokens == 0
    assert first.sessions[0].prefill_tokens == 4
    assert first.pool_load.start_time == 0.0
    assert first.pool_load.finish_time == pytest.approx(2.0)
    assert first.compute.start_time == pytest.approx(2.0)
    assert first.compute.finish_time == pytest.approx(2.4)
    assert first.compute.service_time == pytest.approx(0.4)
    assert first.transfers[0].total_tokens == 14
    assert first.transfers[0].reservation.start_time == pytest.approx(2.4)
    assert first.transfers[0].reservation.finish_time == pytest.approx(4.3)
    assert session.current_episode is not None
    assert session.current_episode.attention_ready_time_s == pytest.approx(4.3)
    assert session.kv.prefill_snapshot == KVSnapshot(14, 4)
    assert session.kv.pool.snapshot == KVSnapshot(10, 0)

    with pytest.raises(ValueError, match="KV readiness"):
        session.emit_decode_token(4.2)
    finish_episode(attention, session, (5.0, 6.0, 7.0), 8.0)

    start_turn(session, 9.0)
    assert prefill.enqueue(session, 9.0) == 0
    second = prefill.process_ready(9.0)[0]

    assert second.sessions[0].context_tokens == 17
    assert second.sessions[0].load_tokens == 3
    assert second.sessions[0].unsynced_decode_tokens == 3
    assert second.sessions[0].prefill_tokens == 2
    assert second.pool_load.service_time == pytest.approx(1.3)
    assert second.compute.service_time == pytest.approx(0.2)
    assert second.transfers[0].total_tokens == 2
    assert second.transfers[0].reservation.service_time == pytest.approx(0.7)
    assert second.transfers[0].admissions[0].cache_hit
    assert session.kv.prefill_snapshot == KVSnapshot(19, 9)
    assert session.kv.pool.snapshot == KVSnapshot(17, 7)
    assert prefill.retained_stateful_tokens == 19


def test_stateless_prefill_reloads_and_retransfers_full_context_each_turn() -> None:
    prefill, attention, _ = make_controller(
        PrefillState.STATELESS,
        residency=AttentionResidency.ACTIVE_ONLY,
    )
    session = make_session(
        0,
        10,
        (
            TurnSpec(0, 4, 3, 0.0),
            TurnSpec(1, 2, 0, 0.0),
        ),
    )
    start_turn(session, 0.0)
    prefill.enqueue(session, 0.0)
    first = prefill.process_ready(0.0)[0]

    assert first.sessions[0].load_tokens == 10
    assert first.transfers[0].total_tokens == 14
    assert session.kv.prefill_snapshot is None
    finish_episode(attention, session, (5.0, 6.0, 7.0), 8.0)

    start_turn(session, 9.0)
    prefill.enqueue(session, 9.0)
    second = prefill.process_ready(9.0)[0]

    assert second.sessions[0].context_tokens == 17
    assert second.sessions[0].load_tokens == 17
    assert second.sessions[0].unsynced_decode_tokens == 0
    assert second.pool_load.service_time == pytest.approx(2.7)
    assert second.compute.service_time == pytest.approx(0.2)
    assert second.transfers[0].total_tokens == 19
    assert second.transfers[0].reservation.service_time == pytest.approx(2.4)
    assert not second.transfers[0].admissions[0].cache_hit
    assert session.kv.prefill_snapshot is None


def test_fcfs_greedy_batch_charges_one_load_compute_and_multidestination_transfer() -> None:
    prefill, _, topology = make_controller(
        PrefillState.STATEFUL,
        residency=AttentionResidency.ACTIVE_ONLY,
        attention_capacities={1: 100, 2: 100},
    )
    first = make_session(0, 5, (TurnSpec(0, 2, 0, 0.0),))
    second = make_session(1, 7, (TurnSpec(0, 3, 0, 0.0),))
    for session in (first, second):
        start_turn(session, 0.0)
        prefill.enqueue(session, 0.0)

    result = prefill.process_ready(0.0)[0]

    assert [request.session.session_id for request in result.batch.requests] == [0, 1]
    assert [item.load_tokens for item in result.sessions] == [5, 7]
    assert result.pool_load.service_time == pytest.approx(2.2)
    assert result.compute.service_time == pytest.approx(0.5)
    assert len(result.transfers) == 1
    transfer = result.transfers[0]
    assert transfer.destination_device_ids == (1, 2)
    assert transfer.total_tokens == 17
    assert transfer.reservation.service_time == pytest.approx(2.2)
    assert len(transfer.reservation.endpoint_reservations) == 3
    assert len(topology.kv_pool.queue.reservations) == 1
    assert len(topology.kv_p2p.queue.reservations) == 1
    assert [item.operation for item in topology.devices[0].reservations] == [
        DeviceOperation.KV_POOL_LOAD,
        DeviceOperation.PREFILL,
        DeviceOperation.KV_P2P,
    ]


def test_stateless_copy_is_held_during_capacity_wait_then_discarded_after_transfer() -> None:
    prefill, attention, _ = make_controller(
        PrefillState.STATELESS,
        residency=AttentionResidency.ACTIVE_ONLY,
        attention_capacities={1: 10},
    )
    occupying = make_session(0, 10, (TurnSpec(0, 0, 0, 0.0),))
    start_turn(occupying, 0.0)
    occupying.prepare_prefill()
    occupying.apply_prefill()
    assert attention.submit(occupying, 0.0) is not None

    waiting = make_session(1, 5, (TurnSpec(0, 1, 0, 0.0),))
    start_turn(waiting, 0.0)
    prefill.enqueue(waiting, 0.0)
    result = prefill.process_ready(0.0)[0]

    assert result.blocked_session_ids == (1,)
    assert result.transfers == ()
    assert prefill.pending_copy(1) is not None
    assert waiting.kv.prefill_snapshot == KVSnapshot(6, 1)
    assert waiting.status is SessionStatus.WAITING_ATTN_CAPACITY

    occupying.finish_decode()
    released = attention.complete_pool_write(occupying, 10.0)
    assert len(released.admissions) == 1
    transfer = prefill.complete_attention_admissions(released.admissions)[0]

    assert transfer.total_tokens == 6
    assert transfer.reservation.start_time == pytest.approx(10.0)
    assert prefill.pending_copy(1) is None
    assert waiting.kv.prefill_snapshot is None
    assert waiting.kv.attention.state is AttentionKVResidency.ACTIVE
    assert waiting.current_episode is not None
    assert waiting.current_episode.attention_ready_time_s == pytest.approx(
        transfer.reservation.finish_time
    )


def test_prefill_placement_is_sticky_only_for_stateful_sessions() -> None:
    stateful = StickyRoundRobinPrefillPlacement((0, 3))
    stateless = RoundRobinPrefillPlacement((0, 3))

    assert [stateful.select(session_id) for session_id in (7, 8, 7, 8)] == [
        0,
        3,
        0,
        3,
    ]
    assert stateful.affinity_for(7) == 0
    assert [stateless.select(7) for _ in range(4)] == [0, 3, 0, 3]
    assert stateless.affinity_for(7) is None


def test_invalid_enqueue_does_not_advance_stateless_placement_cursor() -> None:
    prefill, _, _ = make_controller(
        PrefillState.STATELESS,
        residency=AttentionResidency.ACTIVE_ONLY,
        prefill_ids=(0, 3),
    )
    session = make_session(0, 1, (TurnSpec(0, 1, 0, 0.0),))
    start_turn(session, 1.0)

    with pytest.raises(ValueError, match="cannot precede turn readiness"):
        prefill.enqueue(session, 0.5)

    assert prefill.enqueue(session, 1.0) == 0
    assert prefill.capacity_tokens is None


def test_busy_device_defers_batch_formation_and_keeps_fcfs_order() -> None:
    prefill, _, topology = make_controller(
        PrefillState.STATEFUL,
        residency=AttentionResidency.ACTIVE_ONLY,
    )
    first = make_session(0, 5, (TurnSpec(0, 2, 0, 0.0),))
    later = make_session(1, 5, (TurnSpec(0, 2, 0, 0.0),))
    start_turn(first, 0.0)
    start_turn(later, 1.0)
    prefill.enqueue(first, 0.0)
    prefill.enqueue(later, 1.0)

    first_result = prefill.process_ready(0.0)[0]
    assert [item.session.session_id for item in prefill.batcher.waiting()] == [1]
    assert prefill.process_ready(1.0) == ()
    expected_launch = topology.devices[0].queue.available_time
    assert prefill.next_launch_time_s() == pytest.approx(expected_launch)

    later_result = prefill.process_ready(expected_launch)[0]
    assert [request.session.session_id for request in first_result.batch.requests] == [0]
    assert [request.session.session_id for request in later_result.batch.requests] == [1]


def test_controller_assembles_from_the_typed_example_configuration() -> None:
    config = load_config(
        "configs/example/hardware.yaml",
        "configs/example/workload.yaml",
        "configs/example/mapping.yaml",
    )
    topology = Topology.from_config(config.hardware)
    attention = AttentionAdmissionController.from_config(
        config.mapping.decode_attn,
        topology.devices,
    )

    prefill = PrefillController(topology, config.mapping.prefill, attention)

    assert prefill.mapping.state is PrefillState.STATEFUL
    assert prefill.mapping.placement is PrefillPlacement.STICKY_ROUND_ROBIN
    assert prefill.mapping.device_ids == (0,)
