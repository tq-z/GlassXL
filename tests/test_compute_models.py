from __future__ import annotations

import math

import pytest

from agentic_sim.hardware.compute_models import (
    AggregatePrefillLatencyModel,
    AttentionLatencyModel,
    AttentionWork,
    ExpertComputeLatencyModel,
    ExpertWork,
    LinearAttentionLatencyModel,
    LinearExpertBatchLatencyModel,
    PrefillLatencyModel,
    PrefillWork,
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
