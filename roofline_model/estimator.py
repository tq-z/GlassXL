"""Coarse analytical roofline estimates for simulator token-rate calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .specs import AttentionProfile, LayerGroup, ModelSpec, resolve_model


@dataclass(frozen=True)
class HardwareSpec:
    flops_per_second: float
    memory_bandwidth_bytes_per_second: float

    def __post_init__(self) -> None:
        _positive(self.flops_per_second, "flops_per_second")
        _positive(
            self.memory_bandwidth_bytes_per_second,
            "memory_bandwidth_bytes_per_second",
        )


@dataclass(frozen=True)
class PrecisionSpec:
    weight_bytes: float = 2.0
    activation_bytes: float = 2.0
    kv_bytes: float = 2.0

    def __post_init__(self) -> None:
        _positive(self.weight_bytes, "weight_bytes")
        _positive(self.activation_bytes, "activation_bytes")
        _positive(self.kv_bytes, "kv_bytes")


@dataclass(frozen=True)
class DeploymentSpec:
    dp: int
    ep: int

    def __post_init__(self) -> None:
        _positive_integer(self.dp, "dp")
        _positive_integer(self.ep, "ep")


@dataclass(frozen=True)
class OperatingPoint:
    prefill_tokens: int
    decode_context_tokens: int

    def __post_init__(self) -> None:
        _positive_integer(self.prefill_tokens, "prefill_tokens")
        _positive_integer(self.decode_context_tokens, "decode_context_tokens")


@dataclass(frozen=True)
class PhaseEstimate:
    flops: float
    bytes_moved: float
    compute_s: float
    memory_s: float
    latency_s: float
    tokens_per_second: float
    bottleneck: str


@dataclass(frozen=True)
class GroupEstimate:
    group: LayerGroup
    prefill: PhaseEstimate
    decode_attention: PhaseEstimate


@dataclass(frozen=True)
class CurvePoint:
    activation_tokens: int
    latency_s: float

    @property
    def tokens_per_second(self) -> float | None:
        if self.activation_tokens == 0 or self.latency_s == 0.0:
            return None
        return self.activation_tokens / self.latency_s


@dataclass(frozen=True)
class FFNCurve:
    local_experts: int
    points: tuple[CurvePoint, ...]


@dataclass(frozen=True)
class FFNCurveSet:
    local_experts: int
    all_layers: FFNCurve
    layer_groups: tuple[tuple[str, FFNCurve], ...]


@dataclass(frozen=True)
class ModelEstimate:
    model: ModelSpec
    hardware: HardwareSpec
    precision: PrecisionSpec
    deployment: DeploymentSpec
    operating_point: OperatingPoint
    layer_groups: tuple[GroupEstimate, ...]
    prefill_all_layers: PhaseEstimate
    decode_attention_all_layers: PhaseEstimate
    ffn_curve_sets: tuple[FFNCurveSet, ...]
    routed_activation_bytes: float
    growing_kv_elements_per_token: float
    fixed_state_elements_per_session: float
    effective_kv_bytes_per_token: float


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def roofline_time(
    flops: float,
    bytes_moved: float,
    hardware: HardwareSpec,
) -> tuple[float, float, float, str]:
    """Return roofline latency, compute time, memory time, and limiting side."""

    if flops < 0.0 or bytes_moved < 0.0:
        raise ValueError("roofline work must be non-negative")
    compute_s = flops / hardware.flops_per_second
    memory_s = bytes_moved / hardware.memory_bandwidth_bytes_per_second
    bottleneck = "compute" if compute_s >= memory_s else "memory"
    return max(compute_s, memory_s), compute_s, memory_s, bottleneck


def _sum_minimum(length: int, limit: int) -> int:
    prefix = min(length, limit)
    return prefix * (prefix + 1) // 2 + (length - prefix) * limit


def _sum_ceil_div(length: int, divisor: int) -> int:
    quotient, remainder = divmod(length, divisor)
    return divisor * quotient * (quotient + 1) // 2 + remainder * (quotient + 1)


def _attended_items(profile: AttentionProfile, context: int) -> int:
    if profile.kind in {"kda", "delta_net"}:
        return 0
    if profile.kind in {"dsa_fresh", "dsa_shared"}:
        return min(context, profile.sparse_top_k)
    if profile.kind == "hca":
        return min(context, profile.window) + math.ceil(context / profile.compression)
    if profile.kind == "csa":
        return min(context, profile.window) + min(
            profile.sparse_top_k,
            math.ceil(context / profile.compression),
        )
    return context


def _prefill_attended_items(
    profile: AttentionProfile,
    tokens: int,
    context: int,
) -> int:
    if profile.kind in {"kda", "delta_net"}:
        return 0
    if profile.kind in {"dsa_fresh", "dsa_shared"}:
        return _sum_minimum(
            context + tokens,
            profile.sparse_top_k,
        ) - _sum_minimum(context, profile.sparse_top_k)
    if profile.kind == "hca":
        return (
            _sum_minimum(context + tokens, profile.window)
            - _sum_minimum(context, profile.window)
            + _sum_ceil_div(context + tokens, profile.compression)
            - _sum_ceil_div(context, profile.compression)
        )
    if profile.kind == "csa":
        compressed_limit = profile.sparse_top_k * profile.compression
        total_end = min(context + tokens, compressed_limit)
        total_start = min(context, compressed_limit)
        compressed = _sum_ceil_div(total_end, profile.compression)
        compressed -= _sum_ceil_div(total_start, profile.compression)
        compressed += (
            max(0, context + tokens - compressed_limit) * profile.sparse_top_k
        )
        compressed -= max(0, context - compressed_limit) * profile.sparse_top_k
        return (
            _sum_minimum(context + tokens, profile.window)
            - _sum_minimum(context, profile.window)
            + compressed
        )
    return tokens * context + tokens * (tokens + 1) // 2


def _index_scan_items(profile: AttentionProfile, context: int) -> int:
    if profile.kind == "dsa_fresh":
        return context
    if profile.kind == "csa":
        return math.ceil(context / profile.compression)
    return 0


def _prefill_index_scan_items(
    profile: AttentionProfile,
    tokens: int,
    context: int,
) -> int:
    if profile.kind == "dsa_fresh":
        return tokens * context + tokens * (tokens + 1) // 2
    if profile.kind == "csa":
        return _sum_ceil_div(
            context + tokens,
            profile.compression,
        ) - _sum_ceil_div(context, profile.compression)
    return 0


def _ffn_active_params(model: ModelSpec, group: LayerGroup) -> int:
    if group.ffn_kind == "dense":
        return model.dense_active_params
    return model.moe_active_params


def _prefill_layer(
    model: ModelSpec,
    group: LayerGroup,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    tokens: int,
    context: int,
) -> PhaseEstimate:
    profile = group.attention
    attended = _prefill_attended_items(profile, tokens, context)
    scanned = _prefill_index_scan_items(profile, tokens, context)
    attention_flops = (
        2.0 * profile.projection_params * tokens
        + profile.pair_flops * attended
        + profile.index_scan_flops_per_item * scanned
        + profile.fixed_state_flops * tokens
    )
    active_ffn_params = _ffn_active_params(model, group)
    ffn_flops = 2.0 * active_ffn_params * tokens
    flops = attention_flops + ffn_flops
    bytes_moved = (
        (profile.projection_params + active_ffn_params) * precision.weight_bytes
        + 6.0 * tokens * model.hidden_size * precision.activation_bytes
        + attended * profile.kv_elements_per_token * precision.kv_bytes
        + scanned * profile.index_elements_per_token * precision.kv_bytes
    )
    compute_s = flops / hardware.flops_per_second
    memory_s = bytes_moved / hardware.memory_bandwidth_bytes_per_second
    return PhaseEstimate(
        flops=flops,
        bytes_moved=bytes_moved,
        compute_s=compute_s,
        memory_s=memory_s,
        latency_s=compute_s,
        tokens_per_second=tokens / compute_s,
        bottleneck="compute_assumed",
    )


def _attention_layer(
    model: ModelSpec,
    group: LayerGroup,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    context: int,
) -> PhaseEstimate:
    profile = group.attention
    attended = _attended_items(profile, context)
    scanned = _index_scan_items(profile, context)
    flops = (
        2.0 * profile.projection_params
        + profile.pair_flops * attended
        + profile.index_scan_flops_per_item * scanned
        + profile.fixed_state_flops
    )
    bytes_moved = (
        profile.projection_params * precision.weight_bytes
        + attended * profile.kv_elements_per_token * precision.kv_bytes
        + scanned * profile.index_elements_per_token * precision.kv_bytes
        + 2.0 * profile.fixed_state_elements * precision.kv_bytes
        + 4.0 * model.hidden_size * precision.activation_bytes
    )
    latency_s, compute_s, memory_s, bottleneck = roofline_time(
        flops,
        bytes_moved,
        hardware,
    )
    return PhaseEstimate(
        flops=flops,
        bytes_moved=bytes_moved,
        compute_s=compute_s,
        memory_s=memory_s,
        latency_s=latency_s,
        tokens_per_second=context / latency_s,
        bottleneck=bottleneck,
    )


def _aggregate_phase(
    estimates: tuple[tuple[int, PhaseEstimate], ...],
    token_numerator: int,
    *,
    compute_assumed: bool = False,
) -> PhaseEstimate:
    flops = sum(count * item.flops for count, item in estimates)
    bytes_moved = sum(count * item.bytes_moved for count, item in estimates)
    compute_s = sum(count * item.compute_s for count, item in estimates)
    memory_s = sum(count * item.memory_s for count, item in estimates)
    latency_s = sum(count * item.latency_s for count, item in estimates)
    if compute_assumed:
        bottleneck = "compute_assumed"
    else:
        bottleneck = "compute" if compute_s >= memory_s else "memory"
    return PhaseEstimate(
        flops=flops,
        bytes_moved=bytes_moved,
        compute_s=compute_s,
        memory_s=memory_s,
        latency_s=latency_s,
        tokens_per_second=token_numerator / latency_s,
        bottleneck=bottleneck,
    )


def estimate_prefill(
    model: str | ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    input_tokens: int,
    context_tokens: int = 0,
) -> tuple[tuple[tuple[LayerGroup, PhaseEstimate], ...], PhaseEstimate]:
    _positive_integer(input_tokens, "input_tokens")
    if (
        isinstance(context_tokens, bool)
        or not isinstance(context_tokens, int)
        or context_tokens < 0
    ):
        raise ValueError("context_tokens must be a non-negative integer")
    spec = resolve_model(model) if isinstance(model, str) else model
    groups = tuple(
        (
            group,
            _prefill_layer(
                spec,
                group,
                hardware,
                precision,
                input_tokens,
                context_tokens,
            ),
        )
        for group in spec.layer_groups
    )
    aggregate = _aggregate_phase(
        tuple((group.count, estimate) for group, estimate in groups),
        input_tokens,
        compute_assumed=True,
    )
    return groups, aggregate


def estimate_decode_attention(
    model: str | ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    context_tokens: int,
) -> tuple[tuple[tuple[LayerGroup, PhaseEstimate], ...], PhaseEstimate]:
    _positive_integer(context_tokens, "context_tokens")
    spec = resolve_model(model) if isinstance(model, str) else model
    groups = tuple(
        (group, _attention_layer(spec, group, hardware, precision, context_tokens))
        for group in spec.layer_groups
    )
    aggregate = _aggregate_phase(
        tuple((group.count, estimate) for group, estimate in groups),
        context_tokens,
    )
    return groups, aggregate


@dataclass(frozen=True)
class _FFNWork:
    flops: float
    bytes_moved: float
    latency_s: float


def _single_roofline_work(
    flops: float,
    bytes_moved: float,
    hardware: HardwareSpec,
) -> _FFNWork:
    latency_s, _, _, _ = roofline_time(flops, bytes_moved, hardware)
    return _FFNWork(flops, bytes_moved, latency_s)


def _ffn_layer_work(
    model: ModelSpec,
    group: LayerGroup,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    deployment: DeploymentSpec,
    local_experts: int,
    activation_tokens: int,
) -> _FFNWork:
    if activation_tokens == 0:
        return _FFNWork(0.0, 0.0, 0.0)
    original_tokens = activation_tokens / model.top_k
    works: list[_FFNWork] = []
    if group.ffn_kind == "dense":
        params = model.dense_active_params
        works.append(
            _single_roofline_work(
                2.0 * original_tokens * params,
                params * precision.weight_bytes / deployment.ep
                + original_tokens
                * 3.0
                * (model.hidden_size + model.dense_intermediate_size)
                * precision.activation_bytes,
                hardware,
            )
        )
    else:
        active_experts = min(local_experts, activation_tokens)
        matrix_params = model.routed_width * model.expert_intermediate_size
        for _ in range(3):
            works.append(
                _single_roofline_work(
                    2.0 * activation_tokens * matrix_params,
                    active_experts * matrix_params * precision.weight_bytes
                    + activation_tokens
                    * (model.routed_width + model.expert_intermediate_size)
                    * precision.activation_bytes,
                    hardware,
                )
            )
        if model.shared_active_params:
            works.append(
                _single_roofline_work(
                    2.0 * original_tokens * model.shared_active_params,
                    model.shared_active_params
                    * precision.weight_bytes
                    / deployment.ep
                    + original_tokens
                    * model.shared_experts
                    * 3.0
                    * (model.hidden_size + model.shared_intermediate_size)
                    * precision.activation_bytes,
                    hardware,
                )
            )
        if model.moe_common_active_params:
            works.append(
                _single_roofline_work(
                    2.0 * original_tokens * model.moe_common_active_params,
                    model.moe_common_active_params
                    * precision.weight_bytes
                    / deployment.ep
                    + original_tokens
                    * 4.0
                    * model.hidden_size
                    * precision.activation_bytes,
                    hardware,
                )
            )
    return _FFNWork(
        flops=sum(work.flops for work in works),
        bytes_moved=sum(work.bytes_moved for work in works),
        latency_s=sum(work.latency_s for work in works),
    )


def _crossover_candidates(
    model: ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    deployment: DeploymentSpec,
    local_experts: int,
) -> set[int]:
    components: list[tuple[float, float, float]] = []
    routed_matrix = model.routed_width * model.expert_intermediate_size
    components.append(
        (
            local_experts * routed_matrix * precision.weight_bytes,
            2.0 * routed_matrix,
            (model.routed_width + model.expert_intermediate_size)
            * precision.activation_bytes,
        )
    )
    if model.dense_active_params:
        components.append(
            (
                model.dense_active_params * precision.weight_bytes / deployment.ep,
                2.0 * model.dense_active_params / model.top_k,
                3.0
                * (model.hidden_size + model.dense_intermediate_size)
                * precision.activation_bytes
                / model.top_k,
            )
        )
    if model.shared_active_params:
        components.append(
            (
                model.shared_active_params * precision.weight_bytes / deployment.ep,
                2.0 * model.shared_active_params / model.top_k,
                model.shared_experts
                * 3.0
                * (model.hidden_size + model.shared_intermediate_size)
                * precision.activation_bytes
                / model.top_k,
            )
        )
    if model.moe_common_active_params:
        components.append(
            (
                model.moe_common_active_params * precision.weight_bytes / deployment.ep,
                2.0 * model.moe_common_active_params / model.top_k,
                4.0
                * model.hidden_size
                * precision.activation_bytes
                / model.top_k,
            )
        )

    result: set[int] = set()
    for fixed_bytes, flops_per_token, bytes_per_token in components:
        difference = (
            flops_per_token / hardware.flops_per_second
            - bytes_per_token / hardware.memory_bandwidth_bytes_per_second
        )
        if difference <= 0.0:
            continue
        crossing = (
            fixed_bytes / hardware.memory_bandwidth_bytes_per_second
        ) / difference
        if not math.isfinite(crossing) or crossing <= 0.0 or crossing > 1_000_000_000:
            continue
        center = int(math.ceil(crossing))
        result.update(token for token in (center - 1, center, center + 1) if token > 0)
    return result


def build_ffn_curve(
    model: str | ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    deployment: DeploymentSpec,
    local_experts: int,
    *,
    layer_group: LayerGroup | None = None,
) -> FFNCurve:
    _positive_integer(local_experts, "local_experts")
    spec = resolve_model(model) if isinstance(model, str) else model
    if local_experts > spec.num_experts:
        raise ValueError("local_experts cannot exceed the model expert count")
    groups = (layer_group,) if layer_group is not None else spec.layer_groups
    token_points = set(range(0, local_experts + 1))
    crossings = _crossover_candidates(
        spec,
        hardware,
        precision,
        deployment,
        local_experts,
    )
    token_points.update(crossings)
    largest = max(token_points)
    tail = max(1024, 2 * largest, 2 * local_experts)
    token_points.update((tail, 2 * tail))
    points: list[CurvePoint] = []
    previous_latency = 0.0
    for tokens in sorted(token_points):
        latency = 0.0
        for group in groups:
            work = _ffn_layer_work(
                spec,
                group,
                hardware,
                precision,
                deployment,
                local_experts,
                tokens,
            )
            latency += group.count * work.latency_s
        latency = max(previous_latency, latency)
        points.append(CurvePoint(tokens, latency))
        previous_latency = latency
    if points[-1].latency_s <= points[-2].latency_s:
        extra_tokens = points[-1].activation_tokens * 2
        work_latency = 0.0
        for group in groups:
            work_latency += group.count * _ffn_layer_work(
                spec,
                group,
                hardware,
                precision,
                deployment,
                local_experts,
                extra_tokens,
            ).latency_s
        points.append(CurvePoint(extra_tokens, max(work_latency, points[-1].latency_s + 1e-15)))
    return FFNCurve(local_experts, tuple(points))


def kv_token_size(
    model: str | ModelSpec,
    precision: PrecisionSpec,
    context_tokens: int,
) -> tuple[float, float, float]:
    _positive_integer(context_tokens, "context_tokens")
    spec = resolve_model(model) if isinstance(model, str) else model
    growing_elements = sum(
        group.count
        * (
            group.attention.kv_elements_per_token
            + group.attention.index_elements_per_token
        )
        for group in spec.layer_groups
    )
    fixed_elements = sum(
        group.count * group.attention.fixed_state_elements
        for group in spec.layer_groups
    )
    effective_bytes = (
        growing_elements * precision.kv_bytes
        + fixed_elements * precision.kv_bytes / context_tokens
    )
    return growing_elements, fixed_elements, effective_bytes


def estimate_model(
    model: str | ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    deployment: DeploymentSpec,
    operating_point: OperatingPoint,
) -> ModelEstimate:
    spec = resolve_model(model) if isinstance(model, str) else model
    if deployment.ep > spec.num_experts:
        raise ValueError(
            f"ep={deployment.ep} exceeds routed expert count {spec.num_experts}"
        )
    prefill_groups, prefill_all = estimate_prefill(
        spec,
        hardware,
        precision,
        operating_point.prefill_tokens,
        operating_point.decode_context_tokens,
    )
    attention_groups, attention_all = estimate_decode_attention(
        spec,
        hardware,
        precision,
        operating_point.decode_context_tokens,
    )
    attention_by_name = {group.name: estimate for group, estimate in attention_groups}
    grouped = tuple(
        GroupEstimate(group, prefill, attention_by_name[group.name])
        for group, prefill in prefill_groups
    )

    quotient, remainder = divmod(spec.num_experts, deployment.ep)
    local_counts = tuple(
        quotient + (index < remainder)
        for index in range(deployment.ep)
    )
    curve_sets: list[FFNCurveSet] = []
    for local_experts in sorted(set(local_counts)):
        curve_sets.append(
            FFNCurveSet(
                local_experts=local_experts,
                all_layers=build_ffn_curve(
                    spec,
                    hardware,
                    precision,
                    deployment,
                    local_experts,
                ),
                layer_groups=tuple(
                    (
                        group.name,
                        build_ffn_curve(
                            spec,
                            hardware,
                            precision,
                            deployment,
                            local_experts,
                            layer_group=group,
                        ),
                    )
                    for group in spec.layer_groups
                ),
            )
        )

    growing, fixed, effective = kv_token_size(
        spec,
        precision,
        operating_point.decode_context_tokens,
    )
    return ModelEstimate(
        model=spec,
        hardware=hardware,
        precision=precision,
        deployment=deployment,
        operating_point=operating_point,
        layer_groups=grouped,
        prefill_all_layers=prefill_all,
        decode_attention_all_layers=attention_all,
        ffn_curve_sets=tuple(curve_sets),
        routed_activation_bytes=spec.routed_activation_width
        * precision.activation_bytes,
        growing_kv_elements_per_token=growing,
        fixed_state_elements_per_session=fixed,
        effective_kv_bytes_per_token=effective,
    )

