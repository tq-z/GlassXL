from __future__ import annotations

import math

import pytest

from agentic_sim.hardware.compute_models import (
    AggregatePrefillLatencyModel,
    AttentionLatencyModel,
    AttentionWork,
    BatchLatencyCurveExpertModel,
    ExpertComputeLatencyModel,
    ExpertWork,
    LinearAttentionLatencyModel,
    LinearExpertBatchLatencyModel,
    PrefillLatencyModel,
    PrefillWork,
    RooflineContextPrefillLatencyModel,
    StageContext,
)
from agentic_sim.hardware.interconnect import (
    StartupMinPackageCollectiveModel,
    StartupThroughputP2PModel,
)
from agentic_sim.hardware.kv_pool import SymmetricKVPoolLatencyModel


CONTEXT = StageContext(decode_step=3)


def test_exact_prefill_sum_of_incremental_tokens() -> None:
    model = AggregatePrefillLatencyModel(tokens_per_second=100.0)
    batch = (
        PrefillWork(0, context_tokens=1000, prefill_tokens=10, unsynced_decode_tokens=7),
        PrefillWork(1, context_tokens=9000, prefill_tokens=20, unsynced_decode_tokens=2),
    )

    assert model.latency_s(batch, CONTEXT) == pytest.approx(0.3)


def roofline_prefill(**overrides) -> RooflineContextPrefillLatencyModel:
    values = dict(
        flops_per_second=100.0,
        memory_bandwidth_bytes_per_second=1000.0,
        flops_per_token=2.0,
        flops_per_attention_pair=1.0,
        fixed_weight_bytes=50.0,
        routed_weight_bytes=400.0,
        routed_expert_fraction=0.5,
        context_bytes_per_token=3.0,
        bytes_per_prefill_token=10.0,
    )
    values.update(overrides)
    return RooflineContextPrefillLatencyModel(**values)


def test_exact_roofline_prefill_compute_bound() -> None:
    model = roofline_prefill()
    batch = (
        PrefillWork(0, context_tokens=100, prefill_tokens=3, unsynced_decode_tokens=0),
        PrefillWork(1, context_tokens=40, prefill_tokens=1, unsynced_decode_tokens=5),
    )
    # T=4; pairs = 3*100 + 6 + 1*40 + 1 = 347; flops = 8 + 347 = 355 -> 3.55 s.
    # bytes = 50 + 400*(1-0.5^4) + 3*140 + 10*4 = 885 -> 0.885 s.
    assert model.latency_s(batch, CONTEXT) == pytest.approx(3.55)


def test_exact_roofline_prefill_memory_bound_and_context_rules() -> None:
    model = roofline_prefill(flops_per_second=1e9)
    batch = (
        PrefillWork(0, context_tokens=100, prefill_tokens=2, unsynced_decode_tokens=0),
        PrefillWork(1, context_tokens=7000, prefill_tokens=0, unsynced_decode_tokens=0),
    )
    # A session with no new tokens contributes no context read.
    expected_bytes = 50.0 + 400.0 * (1.0 - 0.5**2) + 3.0 * 100 + 10.0 * 2
    assert model.latency_s(batch, CONTEXT) == pytest.approx(expected_bytes / 1000.0)
    assert model.latency_s(batch[1:], CONTEXT) == 0.0
    assert model.expert_coverage(0) == 0.0
    assert roofline_prefill(routed_expert_fraction=1.0).expert_coverage(3) == 1.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"flops_per_second": 0.0}, "flops_per_second"),
        ({"memory_bandwidth_bytes_per_second": -1.0}, "memory_bandwidth"),
        ({"fixed_weight_bytes": -1.0}, "fixed_weight_bytes"),
        ({"routed_expert_fraction": 1.5}, "routed_expert_fraction"),
    ],
)
def test_roofline_prefill_rejects_invalid_parameters(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        roofline_prefill(**overrides)


def test_exact_attention_sum_of_per_session_context_work() -> None:
    model = LinearAttentionLatencyModel(context_tokens_per_second=1000.0)
    batch = (
        AttentionWork(0, context_tokens=100, device_id=2),
        AttentionWork(1, context_tokens=50, device_id=2),
    )

    assert model.latency_s(batch, CONTEXT) == pytest.approx(0.15)


def test_exact_atomic_per_device_expert_batch_rate() -> None:
    model = LinearExpertBatchLatencyModel(tokens_per_second=4.0)
    batches = (
        ExpertWork(expert_id=0, activation_tokens=3, device_id=4),
        ExpertWork(expert_id=3, activation_tokens=5, device_id=4),
    )

    assert model.latency_s(batches, CONTEXT) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "model",
    [
        AggregatePrefillLatencyModel(10.0),
        LinearAttentionLatencyModel(10.0),
        LinearExpertBatchLatencyModel(10.0),
    ],
)
def test_empty_compute_batch_has_zero_latency(model) -> None:
    assert model.latency_s((), CONTEXT) == 0.0


def test_models_satisfy_replaceable_protocols() -> None:
    assert isinstance(AggregatePrefillLatencyModel(1.0), PrefillLatencyModel)
    assert isinstance(roofline_prefill(), PrefillLatencyModel)
    assert isinstance(LinearAttentionLatencyModel(1.0), AttentionLatencyModel)
    assert isinstance(LinearExpertBatchLatencyModel(1.0), ExpertComputeLatencyModel)


@pytest.mark.parametrize("send_tokens, expected", [(0, 0.11), (9, 0.11), (10, 0.11), (11, 0.12)])
def test_all_to_all_max_floor_below_at_and_above_boundary(
    send_tokens: int,
    expected: float,
) -> None:
    model = StartupMinPackageCollectiveModel(
        start_latency_s=0.01,
        min_package_tokens=10,
        throughput_tokens_per_second=100.0,
    )

    assert model.latency_s(send_tokens, participants=4) == pytest.approx(expected)


def test_exact_p2p_has_startup_and_no_payload_floor() -> None:
    model = StartupThroughputP2PModel(
        start_latency_s=0.01,
        throughput_tokens_per_second=100.0,
    )

    assert model.latency_s(0, src=0, dst=1) == pytest.approx(0.01)
    assert model.latency_s(10, src=0, dst=1) == pytest.approx(0.11)


def test_exact_symmetric_pool_equation() -> None:
    model = SymmetricKVPoolLatencyModel(
        start_latency_s=0.005,
        tokens_per_second=100.0,
    )

    assert model.latency_s(0) == pytest.approx(0.005)
    assert model.latency_s(10) == pytest.approx(0.105)


@pytest.mark.parametrize("rate", [0.0, -1.0, math.inf, math.nan])
@pytest.mark.parametrize(
    "factory",
    [
        AggregatePrefillLatencyModel,
        LinearAttentionLatencyModel,
        LinearExpertBatchLatencyModel,
    ],
)
def test_compute_models_reject_invalid_rates(factory, rate: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        factory(rate)


@pytest.mark.parametrize(
    "constructor, arguments, message",
    [
        (PrefillWork, (0, 0, -1, 0), "prefill_tokens"),
        (AttentionWork, (0, -1, 0), "context_tokens"),
        (ExpertWork, (0, -1, 0), "activation_tokens"),
        (StageContext, (-1,), "decode_step"),
    ],
)
def test_work_records_reject_negative_values(constructor, arguments: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        constructor(*arguments)


def test_transfer_models_reject_invalid_inputs() -> None:
    collective = StartupMinPackageCollectiveModel(0.0, 1, 1.0)
    p2p = StartupThroughputP2PModel(0.0, 1.0)
    pool = SymmetricKVPoolLatencyModel(0.0, 1.0)

    with pytest.raises(ValueError, match="send_tokens"):
        collective.latency_s(-1, 1)
    with pytest.raises(ValueError, match="participants"):
        collective.latency_s(1, 0)
    with pytest.raises(ValueError, match="tokens"):
        p2p.latency_s(-1, 0, 1)
    with pytest.raises(ValueError, match="tokens"):
        pool.latency_s(-1)

def test_batch_latency_curve_exact_interpolated_and_extrapolated() -> None:
    model = BatchLatencyCurveExpertModel(((0, 0.0), (2, 0.01), (4, 0.03)))

    def work(tokens: int) -> tuple[ExpertWork, ...]:
        return (ExpertWork(expert_id=0, activation_tokens=tokens, device_id=0),)

    assert model.latency_s((), CONTEXT) == 0.0
    assert model.latency_s(work(2), CONTEXT) == pytest.approx(0.01)
    assert model.latency_s(work(3), CONTEXT) == pytest.approx(0.02)
    assert model.latency_s(work(6), CONTEXT) == pytest.approx(0.05)
    assert isinstance(model, ExpertComputeLatencyModel)


@pytest.mark.parametrize(
    "points",
    [
        ((0, 0.0),),
        ((1, 0.0), (2, 1.0)),
        ((0, 0.0), (0, 1.0)),
        ((0, 0.0), (2, 1.0), (3, 0.5)),
        ((0, 0.0), (2, 1.0), (3, 1.0)),
    ],
)
def test_batch_latency_curve_rejects_invalid_points(
    points: tuple[tuple[int, float], ...],
) -> None:
    with pytest.raises(ValueError):
        BatchLatencyCurveExpertModel(points)
