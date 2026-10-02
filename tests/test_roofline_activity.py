from __future__ import annotations

import math

import pytest

from roofline_model import (
    DeploymentSpec,
    HardwareSpec,
    PrecisionSpec,
    estimate_decode_attention,
    kv_token_size,
    resolve_model,
    supported_models,
)
from roofline_model.activity import (
    FULL_ATTENTION_KINDS,
    PrefillContextCoefficients,
    amortized_attention_rate,
    decode_attention_activity,
    ffn_layer_activity,
    ffn_layer_time,
    kv_append_bytes_per_token,
    moe_layer_count,
)
from roofline_model.estimator import _attention_layer, _ffn_layer_work, _prefill_layer

HARDWARE = HardwareSpec(1.0e15, 3.0e12)
PRECISION = PrecisionSpec()
FULL_ATTENTION_MODELS = tuple(
    name
    for name in supported_models()
    if all(
        group.attention.kind in FULL_ATTENTION_KINDS
        for group in resolve_model(name).layer_groups
    )
)


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


@pytest.mark.parametrize("name", supported_models())
@pytest.mark.parametrize("context", [1, 17, 4096, 131072])
def test_single_session_attention_matches_estimator(name: str, context: int) -> None:
    spec = resolve_model(name)
    activity = decode_attention_activity(spec, PRECISION, [context])
    expected_flops = sum(
        group.count * _attention_layer(spec, group, HARDWARE, PRECISION, context).flops
        for group in spec.layer_groups
    )
    expected_bytes = sum(
        group.count * _attention_layer(spec, group, HARDWARE, PRECISION, context).bytes_moved
        for group in spec.layer_groups
    )
    append = kv_append_bytes_per_token(spec, PRECISION)
    assert close(activity.flops, expected_flops)
    assert close(activity.bytes_moved - append, expected_bytes)


@pytest.mark.parametrize("name", supported_models())
def test_batched_attention_reads_projection_weights_once(name: str) -> None:
    spec = resolve_model(name)
    contexts = [1024, 8192, 65536]
    batched = decode_attention_activity(spec, PRECISION, contexts)
    singles = [decode_attention_activity(spec, PRECISION, [c]) for c in contexts]
    weights = sum(
        group.count * group.attention.projection_params * PRECISION.weight_bytes
        for group in spec.layer_groups
    )
    assert close(batched.flops, sum(item.flops for item in singles))
    assert close(batched.weight_rd, weights)
    assert close(sum(item.weight_rd for item in singles), 3 * weights)
    assert close(batched.kv_rd, sum(item.kv_rd for item in singles))
    assert decode_attention_activity(spec, PRECISION, []).bytes_moved == 0.0


@pytest.mark.parametrize("name", supported_models())
def test_kv_append_matches_growing_kv_size(name: str) -> None:
    spec = resolve_model(name)
    growing, _, _ = kv_token_size(spec, PRECISION, 4096)
    assert close(kv_append_bytes_per_token(spec, PRECISION), growing * PRECISION.kv_bytes)


@pytest.mark.parametrize("name", supported_models())
@pytest.mark.parametrize(("tokens", "owned"), [(0, 2), (1, 2), (2, 2), (5, 3), (300, 3)])
def test_ffn_layer_matches_estimator(name: str, tokens: int, owned: int) -> None:
    spec = resolve_model(name)
    deployment = DeploymentSpec(dp=4, ep=4)
    for group in spec.layer_groups:
        active = min(owned, tokens)
        expected = _ffn_layer_work(spec, group, HARDWARE, PRECISION, deployment, owned, tokens)
        activity = ffn_layer_activity(
            spec, group, PRECISION, deployment, tokens, max(active, 1) if tokens else 0
        )
        assert close(activity.flops, expected.flops)
        assert close(activity.bytes_moved, expected.bytes_moved)
        assert close(
            ffn_layer_time(spec, group, HARDWARE, PRECISION, deployment, tokens, active),
            expected.latency_s,
        )


def test_ffn_nonrouted_tokens_decouple_shared_work_from_routing() -> None:
    spec = resolve_model("glm-4.5")
    moe = next(group for group in spec.layer_groups if group.ffn_kind == "moe")
    deployment = DeploymentSpec(dp=8, ep=8)
    skewed = ffn_layer_activity(spec, moe, PRECISION, deployment, 64, 5, nonrouted_tokens=16)
    equal = ffn_layer_activity(spec, moe, PRECISION, deployment, 16, 5, nonrouted_tokens=16)
    matrix = spec.routed_width * spec.expert_intermediate_size
    assert close(skewed.flops_gemm - equal.flops_gemm, 6.0 * 48 * matrix)
    assert close(skewed.weight_rd, equal.weight_rd)
    assert moe_layer_count(spec) == 89


@pytest.mark.parametrize("name", FULL_ATTENTION_MODELS)
@pytest.mark.parametrize(("prefill", "context"), [(1, 0), (181, 107_000), (4096, 131_072)])
def test_prefill_flops_match_estimator(name: str, prefill: int, context: int) -> None:
    spec = resolve_model(name)
    coefficients = PrefillContextCoefficients.from_model(spec, PRECISION)
    expected = sum(
        group.count * _prefill_layer(spec, group, HARDWARE, PRECISION, prefill, context).flops
        for group in spec.layer_groups
    )
    assert math.isclose(coefficients.activity([(prefill, context)]).flops, expected, rel_tol=1e-12)


def test_prefill_rejects_non_full_attention() -> None:
    with pytest.raises(ValueError, match="full attention only"):
        PrefillContextCoefficients.from_model("kimi-k3", PRECISION)


def test_qwen3_prefill_coefficients_and_batch_semantics() -> None:
    coefficients = PrefillContextCoefficients.from_model("qwen3-235b", PRECISION)
    projection = 71_303_168
    expert = 3 * 4096 * 1536
    assert coefficients.flops_per_token == 94 * (2 * projection + 2 * 8 * expert)
    assert coefficients.flops_per_attention_pair == 94 * 4 * 64 * 128
    assert coefficients.fixed_weight_bytes == 94 * projection * 2
    assert coefficients.routed_weight_bytes == 94 * 128 * expert * 2
    assert coefficients.routed_expert_fraction == 8 / 128
    assert coefficients.context_bytes_per_token == 192_512
    assert coefficients.bytes_per_prefill_token == 192_512 + 94 * 6 * 4096 * 2

    one = coefficients.activity([(181, 107_000)])
    assert one.kv_rd == 192_512 * 107_000
    assert one.kv_wr == 192_512 * 181
    # A session with no new tokens adds no context read; weights are read once per batch.
    two = coefficients.activity([(181, 107_000), (0, 50_000)])
    assert two == one
    both = coefficients.activity([(100, 10_000), (81, 20_000)])
    assert close(both.weight_rd, coefficients.fixed_weight_bytes
                 + coefficients.routed_weight_bytes * coefficients.expert_coverage(181))
    assert coefficients.expert_coverage(0) == 0.0
    assert close(coefficients.expert_coverage(1), 8 / 128)
    assert coefficients.activity([]) == coefficients.activity([(0, 5)])


def test_simulator_prefill_model_matches_activity_latency() -> None:
    from agentic_sim.hardware.compute_models import (
        PrefillWork,
        RooflineContextPrefillLatencyModel,
        StageContext,
    )

    coefficients = PrefillContextCoefficients.from_model("qwen3-235b", PRECISION)
    for hardware in (HardwareSpec(4 * 1.4625e15, 4 * 6e12), HardwareSpec(1e18, 4 * 6e12), HardwareSpec(1e14, 1e16)):
        document = coefficients.capability_document(hardware)
        assert document.pop("model") == "roofline_context"
        model = RooflineContextPrefillLatencyModel(**document)
        for works in ([(181, 107_000)], [(181, 107_000), (0, 50_000), (4096, 131_072)], [(1, 0)], []):
            batch = tuple(
                PrefillWork(index, context_tokens=context, prefill_tokens=prefill, unsynced_decode_tokens=0)
                for index, (prefill, context) in enumerate(works)
            )
            assert math.isclose(
                model.latency_s(batch, StageContext(0)),
                coefficients.latency_s(works, hardware),
                rel_tol=1e-12,
                abs_tol=0.0,
            )


@pytest.mark.parametrize("name", supported_models())
def test_amortized_attention_rate_reduces_to_estimator_at_batch_one(name: str) -> None:
    spec = resolve_model(name)
    _, aggregate = estimate_decode_attention(spec, HARDWARE, PRECISION, 32768)
    assert amortized_attention_rate(spec, HARDWARE, PRECISION, 32768, 1) == aggregate.tokens_per_second
    assert amortized_attention_rate(spec, HARDWARE, PRECISION, 32768, 8) >= aggregate.tokens_per_second


def test_amortized_rate_matches_batched_roofline() -> None:
    spec = resolve_model("qwen3-235b")
    hardware = HardwareSpec(1.048576e15, 6.4e13)
    batch, context = 8, 131_072
    rate = amortized_attention_rate(spec, hardware, PRECISION, context, batch)
    work = decode_attention_activity(spec, PRECISION, [context] * batch)
    batched_time = max(
        work.flops / hardware.flops_per_second,
        (work.bytes_moved - batch * kv_append_bytes_per_token(spec, PRECISION))
        / hardware.memory_bandwidth_bytes_per_second,
    )
    assert math.isclose(batch * context / rate, batched_time, rel_tol=1e-9)
