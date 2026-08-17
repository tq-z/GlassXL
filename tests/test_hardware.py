from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agentic_sim.config import load_config
from agentic_sim.hardware.compute_models import (
    AggregatePrefillLatencyModel,
    AttentionWork,
    ExpertWork,
    LinearAttentionLatencyModel,
    LinearExpertBatchLatencyModel,
    PrefillWork,
    StageContext,
)
from agentic_sim.hardware.device import Device, DeviceOperation
from agentic_sim.hardware.interconnect import (
    FFNAllToAllInterconnect,
    KVP2PInterconnect,
    StartupMinPackageCollectiveModel,
    StartupThroughputP2PModel,
)
from agentic_sim.hardware.kv_pool import KVPool, KVPoolOperation, SymmetricKVPoolLatencyModel
from agentic_sim.hardware.topology import Topology


CONTEXT = StageContext(0)


def multi_capability_device(device_id: int = 0) -> Device:
    return Device(
        device_id,
        prefill_model=AggregatePrefillLatencyModel(10.0),
        attention_model=LinearAttentionLatencyModel(20.0),
        expert_model=LinearExpertBatchLatencyModel(5.0),
        attention_kv_capacity_tokens=100,
    )


def prefill_device(device_id: int) -> Device:
    return Device(device_id, prefill_model=AggregatePrefillLatencyModel(10.0))


def test_prefill_attention_and_ffn_share_one_physical_queue() -> None:
    device = multi_capability_device()

    prefill = device.submit_prefill((PrefillWork(0, 20, 10, 0),), CONTEXT, ready_time=0.0)
    attention = device.submit_attention((AttentionWork(0, 20, 0),), CONTEXT, ready_time=0.0)
    experts = device.submit_experts(
        (ExpertWork(0, 2, 0), ExpertWork(1, 3, 0)),
        CONTEXT,
        ready_time=0.0,
    )

    assert prefill.start_time == 0.0
    assert prefill.finish_time == pytest.approx(1.0)
    assert attention.start_time == pytest.approx(1.0)
    assert attention.finish_time == pytest.approx(2.0)
    assert experts.start_time == pytest.approx(2.0)
    assert experts.finish_time == pytest.approx(3.0)
    assert [item.operation for item in device.reservations] == [
        DeviceOperation.PREFILL,
        DeviceOperation.DECODE_ATTN,
        DeviceOperation.DECODE_FFN,
    ]


def test_all_resident_experts_form_one_atomic_device_job() -> None:
    device = multi_capability_device()

    experts = device.submit_experts(
        (ExpertWork(0, 4, 0), ExpertWork(1, 6, 0)),
        CONTEXT,
        ready_time=0.0,
    )
    later_prefill = device.submit_prefill(
        (PrefillWork(1, 0, 5, 0),),
        CONTEXT,
        ready_time=0.5,
    )

    assert experts.service_time == pytest.approx(2.0)
    assert len(device.queue.reservations) == 2
    assert later_prefill.start_time == pytest.approx(2.0)


def test_different_physical_devices_execute_concurrently() -> None:
    first = prefill_device(0)
    second = prefill_device(1)

    first_job = first.submit_prefill((PrefillWork(0, 0, 10, 0),), CONTEXT, 0.0)
    second_job = second.submit_prefill((PrefillWork(1, 0, 10, 0),), CONTEXT, 0.0)

    assert first_job.start_time == second_job.start_time == 0.0
    assert first_job.finish_time == second_job.finish_time == pytest.approx(1.0)


def test_work_must_target_the_submitting_device() -> None:
    device = multi_capability_device(2)

    with pytest.raises(ValueError, match="attention work must target device 2"):
        device.submit_attention((AttentionWork(0, 1, 3),), CONTEXT, 0.0)
    with pytest.raises(ValueError, match="expert work must target device 2"):
        device.submit_experts((ExpertWork(0, 1, 3),), CONTEXT, 0.0)


def test_all_to_all_waits_for_endpoints_and_occupies_every_queue() -> None:
    first = prefill_device(0)
    second = prefill_device(1)
    first.submit_prefill((PrefillWork(0, 0, 20, 0),), CONTEXT, 0.0)
    interconnect = FFNAllToAllInterconnect(
        StartupMinPackageCollectiveModel(0.0, 0, 10.0)
    )

    collective = interconnect.submit(10, (first, second), ready_time=0.0)

    assert collective.start_time == pytest.approx(2.0)
    assert collective.finish_time == pytest.approx(3.0)
    assert collective.queue_delay == pytest.approx(2.0)
    assert [item.start_time for item in collective.endpoint_reservations] == [2.0, 2.0]
    assert [item.operation for item in collective.endpoint_reservations] == [
        DeviceOperation.FFN_ALL_TO_ALL,
        DeviceOperation.FFN_ALL_TO_ALL,
    ]


def test_global_p2p_queue_serializes_otherwise_disjoint_endpoints() -> None:
    devices = [prefill_device(index) for index in range(4)]
    interconnect = KVP2PInterconnect(StartupThroughputP2PModel(0.0, 10.0))

    first = interconnect.submit(10, devices[0], devices[1], ready_time=0.0)
    second = interconnect.submit(10, devices[2], devices[3], ready_time=0.0)

    assert first.start_time == 0.0
    assert first.finish_time == pytest.approx(1.0)
    assert second.start_time == pytest.approx(1.0)
    assert second.finish_time == pytest.approx(2.0)


def test_pool_load_and_offload_share_queue_and_occupy_device() -> None:
    first = prefill_device(0)
    second = prefill_device(1)
    first.submit_prefill((PrefillWork(0, 0, 10, 0),), CONTEXT, 0.0)
    pool = KVPool(SymmetricKVPoolLatencyModel(0.0, 10.0))

    load = pool.submit_load(10, first, ready_time=0.0)
    offload = pool.submit_offload(10, second, ready_time=0.0)

    assert load.operation is KVPoolOperation.LOAD
    assert load.start_time == pytest.approx(1.0)
    assert load.finish_time == pytest.approx(2.0)
    assert load.endpoint_reservation.operation is DeviceOperation.KV_POOL_LOAD
    assert offload.operation is KVPoolOperation.OFFLOAD
    assert offload.start_time == pytest.approx(2.0)
    assert offload.finish_time == pytest.approx(3.0)
    assert offload.endpoint_reservation.operation is DeviceOperation.KV_POOL_OFFLOAD


def test_v1_pool_capacity_is_explicitly_unbounded() -> None:
    pool = KVPool(SymmetricKVPoolLatencyModel(0.0, 1.0))

    assert pool.capacity_tokens is None
    assert pool.can_fit(10**30)
    with pytest.raises(ValueError, match="unbounded"):
        KVPool(SymmetricKVPoolLatencyModel(0.0, 1.0), capacity_tokens=1)


def test_topology_builds_only_typed_token_unit_resources() -> None:
    config = load_config(
        "configs/example/hardware.yaml",
        "configs/example/workload.yaml",
        "configs/example/mapping.yaml",
    )
    topology = Topology.from_config(config.hardware)

    assert set(topology.devices) == {0, 1, 2, 3, 4}
    assert topology.devices[0].capabilities == {DeviceOperation.PREFILL}
    assert topology.devices[1].attention_kv_capacity_tokens == 262144
    assert topology.kv_pool.capacity_tokens is None
    p2p = topology.submit_kv_p2p(10, 0, 1, ready_time=0.0)
    assert p2p.service_time == pytest.approx(5e-6 + 10 / 1_000_000)


@dataclass
class RecordingAttentionModel:
    calls: list[StageContext] = field(default_factory=list)

    def latency_s(self, batch, context: StageContext) -> float:
        self.calls.append(context)
        return 0.25 * len(batch)


def test_future_two_layer_plan_reuses_interface_without_scaling_v1_summary() -> None:
    summary_device = Device(
        0,
        attention_model=LinearAttentionLatencyModel(100.0),
        attention_kv_capacity_tokens=100,
    )
    summary = summary_device.submit_attention(
        (AttentionWork(0, 10, 0),),
        StageContext(0, layer_id=None),
        ready_time=0.0,
    )
    assert summary.service_time == pytest.approx(0.1)
    assert len(summary_device.reservations) == 1

    layer_model = RecordingAttentionModel()
    layer_device = Device(
        1,
        attention_model=layer_model,
        attention_kv_capacity_tokens=100,
    )
    first = layer_device.submit_attention(
        (AttentionWork(0, 10, 1),),
        StageContext(0, layer_id=0),
        ready_time=0.0,
    )
    second = layer_device.submit_attention(
        (AttentionWork(0, 10, 1),),
        StageContext(0, layer_id=1),
        ready_time=first.finish_time,
    )

    assert [context.layer_id for context in layer_model.calls] == [0, 1]
    assert second.finish_time == pytest.approx(0.5)
