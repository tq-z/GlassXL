from __future__ import annotations

import math

import pytest

from roofline_model import (
    DeploymentSpec,
    HardwareSpec,
    OperatingPoint,
    PrecisionSpec,
    build_ffn_curve,
    estimate_decode_attention,
    estimate_model,
    estimate_prefill,
    kv_token_size,
    resolve_model,
    roofline_time,
    supported_models,
)


EXPECTED = {
    "glm-5.2": (78, (3, 18, 57), 256, 8),
    "deepseek-v4": (61, (31, 30), 384, 6),
    "kimi-k3": (93, (1, 68, 24), 896, 16),
    "qwen3.8-2.4t": (92, (69, 23), 512, 10),
    "glm-4.5": (92, (3, 89), 160, 8),
    "deepseek-v3": (61, (3, 58), 256, 8),
    "kimi-k2": (61, (1, 60), 384, 8),
    "qwen3-235b": (94, (94,), 128, 8),
}


@pytest.mark.parametrize("name", supported_models())
def test_registry_shape_and_grouped_layers(name: str) -> None:
    spec = resolve_model(name)
    layers, groups, experts, top_k = EXPECTED[name]
    assert spec.num_layers == layers
    assert tuple(group.count for group in spec.layer_groups) == groups
    assert sum(group.count for group in spec.layer_groups) == layers
    assert spec.num_experts == experts
    assert spec.top_k == top_k
    assert spec.source_urls


def test_aliases_and_explicit_v4_pro_identity() -> None:
    assert resolve_model("GLM5.2").key == "glm-5.2"
    assert resolve_model("QWEN3-235B").key == "qwen3-235b"
    assert resolve_model("DeepSeek-V4").display_name == "DeepSeek-V4-Pro-1.6T"
    assert resolve_model("Qwen3.8-2.4T").num_layers == 92


def test_roofline_selects_compute_and_memory_limits() -> None:
    hardware = HardwareSpec(100.0, 10.0)
    assert roofline_time(100.0, 1.0, hardware) == (1.0, 1.0, 0.1, "compute")
    latency, compute, memory, bottleneck = roofline_time(1.0, 100.0, hardware)
    assert (latency, compute, memory, bottleneck) == (10.0, 0.01, 10.0, "memory")


def test_prefill_uses_compute_latency_and_includes_causal_growth() -> None:
    hardware = HardwareSpec(1e15, 3e12)
    precision = PrecisionSpec()
    _, short = estimate_prefill("qwen3-235b", hardware, precision, 64)
    _, long = estimate_prefill("qwen3-235b", hardware, precision, 128)
    assert short.latency_s == short.compute_s
    assert short.bottleneck == "compute_assumed"
    assert long.flops > 2.0 * short.flops



def test_prefill_accounts_for_existing_context() -> None:
    hardware = HardwareSpec(1e15, 3e12)
    precision = PrecisionSpec()
    tokens = 64
    context = 1024
    groups_zero, aggregate_zero = estimate_prefill(
        "qwen3-235b", hardware, precision, tokens, 0
    )
    groups_context, aggregate_context = estimate_prefill(
        "qwen3-235b", hardware, precision, tokens, context
    )
    spec = resolve_model("qwen3-235b")
    expected_extra_flops = (
        spec.num_layers
        * spec.layer_groups[0].attention.pair_flops
        * tokens
        * context
    )
    assert aggregate_context.flops - aggregate_zero.flops == expected_extra_flops
    assert aggregate_context.tokens_per_second < aggregate_zero.tokens_per_second

    kimi_zero, _ = estimate_prefill("kimi-k3", hardware, precision, tokens, 0)
    kimi_context, _ = estimate_prefill(
        "kimi-k3", hardware, precision, tokens, context
    )
    assert kimi_context[0][1].flops == kimi_zero[0][1].flops
    assert kimi_context[2][1].flops > kimi_zero[2][1].flops

def test_linear_state_is_context_independent_but_full_attention_is_not() -> None:
    hardware = HardwareSpec(1e15, 3e12)
    precision = PrecisionSpec()
    kimi_short, _ = estimate_decode_attention("kimi-k3", hardware, precision, 1024)
    kimi_long, _ = estimate_decode_attention("kimi-k3", hardware, precision, 8192)
    assert kimi_short[0][1].latency_s == kimi_long[0][1].latency_s
    qwen_short, _ = estimate_decode_attention("qwen3-235b", hardware, precision, 1024)
    qwen_long, _ = estimate_decode_attention("qwen3-235b", hardware, precision, 8192)
    assert qwen_long[0][1].bytes_moved > qwen_short[0][1].bytes_moved


def test_attention_aggregate_sums_group_latencies() -> None:
    groups, aggregate = estimate_decode_attention(
        "glm-5.2", HardwareSpec(1e15, 3e12), PrecisionSpec(), 4096
    )
    expected = sum(group.count * estimate.latency_s for group, estimate in groups)
    assert math.isclose(aggregate.latency_s, expected)
    assert math.isclose(aggregate.tokens_per_second, 4096 / expected)


def test_fixed_recurrent_state_is_amortized_for_kv_conversion() -> None:
    precision = PrecisionSpec()
    growing_short, fixed_short, bytes_short = kv_token_size("kimi-k3", precision, 1024)
    growing_long, fixed_long, bytes_long = kv_token_size("kimi-k3", precision, 8192)
    assert growing_short == growing_long
    assert fixed_short == fixed_long > 0
    assert bytes_short > bytes_long
    _, fixed_mla, _ = kv_token_size("kimi-k2", precision, 1024)
    assert fixed_mla == 0


def test_ffn_curve_has_active_expert_transitions_and_stable_tail() -> None:
    curve = build_ffn_curve(
        "qwen3-235b",
        HardwareSpec(1e15, 3e12),
        PrecisionSpec(),
        DeploymentSpec(2, 8),
        16,
    )
    token_values = [point.activation_tokens for point in curve.points]
    latencies = [point.latency_s for point in curve.points]
    assert token_values[:17] == list(range(17))
    assert latencies[0] == 0
    assert all(right >= left for left, right in zip(latencies, latencies[1:]))
    assert curve.points[-1].latency_s > curve.points[-2].latency_s
    assert curve.points[-1].activation_tokens >= 2 * curve.points[-2].activation_tokens


def test_estimator_rejects_more_ep_ranks_than_experts() -> None:
    with pytest.raises(ValueError, match="exceeds routed expert count"):
        estimate_model(
            "qwen3-235b",
            HardwareSpec(1e15, 3e12),
            PrecisionSpec(),
            DeploymentSpec(1, 129),
            OperatingPoint(1024, 4096),
        )
