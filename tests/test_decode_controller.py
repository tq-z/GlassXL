from __future__ import annotations

from pathlib import Path

import pytest

from agentic_sim.config.models import (
    AttentionPlacement,
    AttentionResidency,
    CapacityReservationConfig,
    DecodeAttentionMappingConfig,
    DecodeFFNMappingConfig,
    ExpertPlacementConfig,
    MappingConfig,
    ModelShapeConfig,
    PrefillMappingConfig,
    PrefillPlacement,
    PrefillState,
)
from agentic_sim.hardware.compute_models import (
    LinearAttentionLatencyModel,
    LinearExpertBatchLatencyModel,
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
from agentic_sim.mapping import ExpertMapping
from agentic_sim.runtime import (
    AttentionAdmissionController,
    AttentionKVResidency,
    DecodeController,
    DecodeEpisodePhase,
    SessionState,
)
from agentic_sim.workload import SessionSpec, TurnSpec


def mapping_config(
    attention_ids: tuple[int, ...],
    experts_by_device: dict[int, tuple[int, ...]],
    *,
    num_experts: int,
    top_k: int,
    num_layers: int = 32,
) -> MappingConfig:
    ffn_ids = tuple(experts_by_device)
    return MappingConfig(
        schema_version=1,
        granularity="decode_step",
        model=ModelShapeConfig(num_layers, num_experts, top_k),
        prefill=PrefillMappingConfig(
            (99,),
            PrefillState.STATELESS,
            PrefillPlacement.ROUND_ROBIN,
        ),
        decode_attn=DecodeAttentionMappingConfig(
            attention_ids,
            AttentionPlacement.ROUND_ROBIN,
            AttentionResidency.ACTIVE_ONLY,
            CapacityReservationConfig("oracle_output"),
        ),
        decode_ffn=DecodeFFNMappingConfig(
            ffn_ids,
            tuple(
                ExpertPlacementConfig(device_id, expert_ids)
                for device_id, expert_ids in experts_by_device.items()
            ),
        ),
        source_path=Path("mapping.yaml"),
    )


def topology(
    attention_rates: dict[int, float],
    expert_rates: dict[int, float],
    *,
    collective_start_s: float = 0.1,
    collective_min_tokens: int = 1,
    collective_rate: float = 10.0,
    pool_start_s: float = 0.2,
    pool_rate: float = 10.0,
) -> Topology:
    device_ids = set(attention_rates) | set(expert_rates)
    devices = {
        device_id: Device(
            device_id,
            attention_model=(
                LinearAttentionLatencyModel(attention_rates[device_id])
                if device_id in attention_rates
                else None
            ),
            expert_model=(
                LinearExpertBatchLatencyModel(expert_rates[device_id])
                if device_id in expert_rates
                else None
            ),
            attention_kv_capacity_tokens=(
                1_000 if device_id in attention_rates else None
            ),
        )
        for device_id in device_ids
    }
    return Topology(
        devices,
        FFNAllToAllInterconnect(
            StartupMinPackageCollectiveModel(
                collective_start_s,
                collective_min_tokens,
                collective_rate,
            )
        ),
        KVP2PInterconnect(StartupThroughputP2PModel(0.0, 100.0)),
        KVPool(SymmetricKVPoolLatencyModel(pool_start_s, pool_rate)),
    )


def session(
    session_id: int,
    context_tokens: int,
    *,
    prefill_tokens: int = 0,
    output_tokens: int = 2,
) -> SessionState:
    result = SessionState(
        SessionSpec(
            session_id,
            0.0,
            context_tokens,
            (TurnSpec(0, prefill_tokens, output_tokens, 0.0),),
        )
    )
    result.arrive(0.0)
    result.start_episode(0.0)
    result.prepare_prefill()
    result.apply_prefill()
    return result


def controller(
    hardware: Topology,
    config: MappingConfig,
    *,
    residency: AttentionResidency = AttentionResidency.ACTIVE_ONLY,
) -> tuple[DecodeController, AttentionAdmissionController]:
    capacities = {
        device_id: hardware.get_device(device_id).attention_kv_capacity_tokens
        for device_id in config.decode_attn.device_ids
    }
    assert all(value is not None for value in capacities.values())
    attention = AttentionAdmissionController(
        config.decode_attn.device_ids,
        {key: int(value) for key, value in capacities.items()},
        config.decode_attn.placement,
        residency,
    )
    return DecodeController(hardware, config, attention), attention


def admit(attention: AttentionAdmissionController, *sessions: SessionState) -> None:
    for item in sessions:
        assert attention.submit(item, 0.0) is not None


def test_exact_split_stage_order_barriers_and_one_token_per_session() -> None:
    hardware = topology({1: 10.0, 2: 20.0}, {3: 2.0, 4: 4.0})
    config = mapping_config(
        (1, 2),
        {3: (0, 1), 4: (2, 3)},
        num_experts=4,
        top_k=2,
    )
    decode, attention = controller(hardware, config)
    first = session(0, 10)
    second = session(1, 20)
    admit(attention, first, second)

    result = decode.run_step(0.0)

    assert tuple(tuple(work.session_id for work in batch.work) for batch in result.batch.local_batches) == ((0,), (1,))
    assert tuple(item.service_time for item in result.attention) == pytest.approx((1.0, 1.0))
    assert result.attention_barrier_s == pytest.approx(1.0)
    assert result.dispatch.start_time == pytest.approx(1.0)
    assert result.dispatch.finish_time == pytest.approx(1.3)
    assert result.activation_tokens == 4
    assert result.send_tokens == 2
    assert tuple(work.activation_tokens for work in result.expert_loads) == (1, 1, 1, 1)
    assert tuple(item.service_time for item in result.ffn) == pytest.approx((1.0, 0.5))
    assert result.ffn_barrier_s == pytest.approx(2.3)
    assert result.combine.start_time == pytest.approx(2.3)
    assert result.combine.finish_time == pytest.approx(2.6)
    assert result.dispatch.service_time == result.combine.service_time
    assert result.emitted_context_tokens == ((0, 11), (1, 21))
    assert first.current_episode is not None
    assert second.current_episode is not None
    assert first.current_episode.generated_tokens == 1
    assert second.current_episode.generated_tokens == 1
    assert first.kv.attention.device_id == 1
    assert second.kv.attention.device_id == 2


def test_context_increments_before_next_step_and_num_layers_does_not_scale() -> None:
    hardware = topology({1: 10.0, 2: 20.0}, {3: 2.0, 4: 4.0})
    config = mapping_config(
        (1, 2),
        {3: (0, 1), 4: (2, 3)},
        num_experts=4,
        top_k=2,
        num_layers=96,
    )
    decode, attention = controller(hardware, config)
    first = session(0, 10)
    second = session(1, 20)
    admit(attention, first, second)

    previous = decode.run_step(0.0)
    current = decode.run_step(previous.token_completion_time_s)

    assert tuple(work.context_tokens for work in current.batch.local_batches[0].work) == (11,)
    assert tuple(work.context_tokens for work in current.batch.local_batches[1].work) == (21,)
    assert tuple(item.service_time for item in current.attention) == pytest.approx((1.1, 1.05))
    assert current.activation_tokens == 4
    assert current.send_tokens == 2
    assert current.decode_step == 1


def test_routing_conserves_compute_when_activation_count_is_below_ep_width() -> None:
    hardware = topology({1: 10.0}, {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    config = mapping_config(
        (1,),
        {2: (0,), 3: (1,), 4: (2,), 5: (3,)},
        num_experts=4,
        top_k=1,
    )
    decode, attention = controller(hardware, config)
    active = session(0, 1)
    admit(attention, active)

    result = decode.run_step(0.0)

    assert result.activation_tokens == 1
    assert result.send_tokens == 1
    assert tuple(work.activation_tokens for work in result.expert_loads) == (1, 0, 0, 0)
    assert sum(work.activation_tokens for work in result.expert_loads) == 1
    assert tuple(batch.device_id for batch in result.expert_batches) == (2, 3, 4, 5)
    assert len(result.ffn) == 4
    assert tuple(item.service_time for item in result.ffn) == (1.0, 0.0, 0.0, 0.0)


def test_expert_mapping_uses_configured_ownership_and_quotient_remainder() -> None:
    config = mapping_config(
        (1,),
        {7: (0, 3), 8: (1, 2)},
        num_experts=4,
        top_k=2,
    )
    experts = ExpertMapping.from_config(config.decode_ffn, 4)

    loads = experts.equal_loads(6)
    batches = experts.device_batches(6)

    assert tuple((work.expert_id, work.activation_tokens, work.device_id) for work in loads) == (
        (0, 2, 7),
        (1, 2, 8),
        (2, 1, 8),
        (3, 1, 7),
    )
    assert tuple(batch.activation_tokens for batch in batches) == (3, 3)


def test_all_dp_devices_join_attention_barrier_even_with_empty_local_batch() -> None:
    hardware = topology({1: 10.0, 2: 10.0}, {3: 10.0})
    config = mapping_config((1, 2), {3: (0,)}, num_experts=1, top_k=1)
    decode, attention = controller(hardware, config)
    active = session(0, 10)
    admit(attention, active)
    hardware.get_device(2).reserve_external(DeviceOperation.KV_P2P, 0.0, 2.0)

    result = decode.run_step(0.0)

    assert result.batch.local_batches[1].work == ()
    assert result.attention[0].finish_time == pytest.approx(1.0)
    assert result.attention[1].start_time == pytest.approx(2.0)
    assert result.attention_barrier_s == pytest.approx(2.0)
    assert result.dispatch.start_time == pytest.approx(2.0)


def test_colocated_mapping_still_charges_full_collectives_on_shared_queue() -> None:
    hardware = topology(
        {1: 10.0},
        {1: 4.0},
        collective_start_s=0.4,
        collective_rate=10.0,
    )
    config = mapping_config((1,), {1: (0,)}, num_experts=1, top_k=1)
    decode, attention = controller(hardware, config)
    active = session(0, 10)
    admit(attention, active)

    result = decode.run_step(0.0)

    assert decode.participant_ids == (1,)
    assert result.attention[0].finish_time == pytest.approx(1.0)
    assert result.dispatch.service_time == pytest.approx(0.5)
    assert result.dispatch.start_time == pytest.approx(1.0)
    assert result.ffn[0].start_time == pytest.approx(1.5)
    assert result.ffn[0].finish_time == pytest.approx(1.75)
    assert result.combine.start_time == pytest.approx(1.75)
    assert result.combine.finish_time == pytest.approx(2.25)
    operations = tuple(item.operation for item in hardware.get_device(1).reservations)
    assert operations[:4] == (
        DeviceOperation.DECODE_ATTN,
        DeviceOperation.FFN_ALL_TO_ALL,
        DeviceOperation.DECODE_FFN,
        DeviceOperation.FFN_ALL_TO_ALL,
    )


@pytest.mark.parametrize(
    ("residency", "clean_state", "remains_resident"),
    (
        (AttentionResidency.ACTIVE_ONLY, AttentionKVResidency.EVICTED, False),
        (AttentionResidency.LRU_CACHE, AttentionKVResidency.INACTIVE_CLEAN, True),
    ),
)
def test_episode_pool_write_is_p_plus_o_dirty_until_modeled_completion(
    residency: AttentionResidency,
    clean_state: AttentionKVResidency,
    remains_resident: bool,
) -> None:
    hardware = topology({1: 10.0}, {2: 10.0})
    config = mapping_config((1,), {2: (0,)}, num_experts=1, top_k=1)
    decode, attention = controller(hardware, config, residency=residency)
    active = session(0, 5, prefill_tokens=3, output_tokens=1)
    admit(attention, active)
    old_pool = active.kv.pool.snapshot

    result = decode.run_step(0.0)
    pending = result.pool_writes[0]

    assert pending.payload_tokens == 4
    assert pending.reservation.service_time == pytest.approx(0.6)
    assert active.current_episode is not None
    assert active.current_episode.phase is DecodeEpisodePhase.DRAINING_DIRTY
    assert active.kv.attention.state is AttentionKVResidency.DRAINING_DIRTY
    assert active.kv.pool.snapshot == old_pool
    assert attention.devices[1].reserved_or_used_tokens == 9

    completed = decode.complete_pool_write(active.session_id)

    assert completed.completion.pool_write_tokens == 4
    assert active.current_episode.phase is DecodeEpisodePhase.POOL_SYNCED
    assert active.kv.pool.snapshot == active.kv.current_snapshot
    assert active.kv.attention.state is clean_state
    assert (active.session_id in attention.devices[1].resident_sessions) is remains_resident
    assert decode.pending_pool_writes == ()


def test_zero_output_episode_flushes_without_decode_or_ffn_work() -> None:
    hardware = topology({1: 10.0}, {2: 10.0})
    config = mapping_config((1,), {2: (0,)}, num_experts=1, top_k=1)
    decode, attention = controller(hardware, config)
    active = session(0, 5, prefill_tokens=3, output_tokens=0)
    admit(attention, active)

    pending = decode.flush_zero_output_sessions(0.0)

    assert len(pending) == 1
    assert pending[0].payload_tokens == 3
    assert active.current_episode is not None
    assert active.current_episode.generated_tokens == 0
    assert active.current_episode.phase is DecodeEpisodePhase.DRAINING_DIRTY
    assert decode.next_decode_step == 0
    decode.complete_pool_write(active.session_id)
    assert active.kv.pool.snapshot.context_tokens == 8


def test_pending_attention_transfer_is_not_decode_ready() -> None:
    hardware = topology({1: 10.0}, {2: 10.0})
    config = mapping_config((1,), {2: (0,)}, num_experts=1, top_k=1)
    decode, attention = controller(hardware, config)
    active = session(0, 5)
    admission = attention.submit(active, 0.0, defer_transfer=True)
    assert admission is not None
    assert admission.transfer_pending

    with pytest.raises(ValueError, match="at least one ready session"):
        decode.run_step(0.0)

    active.complete_attention_transfer(1.0)
    with pytest.raises(ValueError, match="at least one ready session"):
        decode.run_step(0.5)
    result = decode.run_step(1.0)
    assert result.batch.sessions == (active,)


def test_capacity_retry_preserves_pending_transfer_readiness() -> None:
    hardware = topology({1: 10.0}, {2: 10.0})
    hardware.get_device(1).attention_kv_capacity_tokens = 6
    config = mapping_config((1,), {2: (0,)}, num_experts=1, top_k=1)
    decode, attention = controller(hardware, config)
    occupying = session(0, 5, output_tokens=1)
    waiting = session(1, 5, output_tokens=1)
    admit(attention, occupying)
    assert attention.submit(waiting, 0.0, defer_transfer=True) is None

    step = decode.run_step(0.0)
    cleanup = decode.complete_pool_write(occupying.session_id)

    assert len(cleanup.completion.admissions) == 1
    retried = cleanup.completion.admissions[0]
    assert retried.session_id == waiting.session_id
    assert retried.transfer_pending
    assert waiting.current_episode is not None
    assert waiting.current_episode.attention_ready_time_s is None
    with pytest.raises(ValueError, match="at least one ready session"):
        decode.run_step(step.pool_writes[0].reservation.finish_time)

    transfer_completion = step.pool_writes[0].reservation.finish_time + 0.5
    waiting.complete_attention_transfer(transfer_completion)
    result = decode.run_step(transfer_completion)
    assert result.batch.sessions == (waiting,)
