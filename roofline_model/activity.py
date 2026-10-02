"""Decomposed activity counters for energy accounting.

The roofline estimator reports one byte total per phase, which is enough for
latency but not for energy: weight and KV bytes come from DRAM/HBM, activations
move through on-chip buffers, and GEMM and attention FLOPs load the datapath
differently. These counters reuse the estimator's model math (attended items,
GEMM shapes, KV sizes) and keep each kind separate.

Two intentional differences from the estimator's decode-attention bytes:

* projection weights are read once per device per step, not once per session
  (the estimator's per-session charge is only right at batch 1);
* each decode step appends one token of KV per session (``kv_wr``).

Prefill bytes are modelled once per batch (weights) and once per session
(context KV, flash-attention style). ``_prefill_layer`` charges KV per
query-key pair, which is harmless for its compute-only latency but is not a
byte count, so it is deliberately not reused here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import astuple, dataclass

from .estimator import (
    DeploymentSpec,
    HardwareSpec,
    PrecisionSpec,
    _attended_items,
    _ffn_active_params,
    _ffn_layer_work,
    _index_scan_items,
    roofline_time,
)
from .specs import LayerGroup, ModelSpec, resolve_model

FULL_ATTENTION_KINDS = frozenset({"gqa_full", "gated_full", "mla_full"})


@dataclass(frozen=True)
class Activity:
    """FLOPs and bytes by kind for one unit of work.

    FLOPs count multiplies and adds separately (2 per MAC). ``weight_rd``,
    ``kv_rd`` and ``kv_wr`` are main-memory (DRAM/HBM) bytes; ``act`` is the
    estimator's activation traffic term (4*hidden per decode layer, 6*hidden per
    prefill token-layer). power_model prices it as on-chip SRAM on an NMP node
    and as HBM traffic on a GPU, where large prefill batches spill activations.
    """

    flops_gemm: float = 0.0
    flops_attn: float = 0.0
    weight_rd: float = 0.0
    kv_rd: float = 0.0
    kv_wr: float = 0.0
    act: float = 0.0

    def __add__(self, other: Activity) -> Activity:
        return Activity(*(a + b for a, b in zip(astuple(self), astuple(other))))

    def scaled(self, factor: float) -> Activity:
        return Activity(*(value * factor for value in astuple(self)))

    @property
    def flops(self) -> float:
        return self.flops_gemm + self.flops_attn

    @property
    def mem_rd(self) -> float:
        return self.weight_rd + self.kv_rd

    @property
    def mem_wr(self) -> float:
        return self.kv_wr

    @property
    def bytes_moved(self) -> float:
        return self.weight_rd + self.kv_rd + self.kv_wr + self.act


def _spec(model: str | ModelSpec) -> ModelSpec:
    return resolve_model(model) if isinstance(model, str) else model


def kv_append_bytes_per_token(model: str | ModelSpec, precision: PrecisionSpec) -> float:
    """KV bytes written when one token joins a session's context (all layers)."""

    spec = _spec(model)
    return precision.kv_bytes * sum(
        group.count
        * (group.attention.kv_elements_per_token + group.attention.index_elements_per_token)
        for group in spec.layer_groups
    )


@dataclass(frozen=True)
class DecodeAttentionCoefficients:
    """Per-device decode-attention activity for one whole-model step.

    Full-attention groups are linear in (sessions, summed context); other
    attention kinds are evaluated per session with the estimator's helpers.
    """

    per_session: Activity
    per_context_token: Activity
    per_step_weight_bytes: float
    nonlinear_groups: tuple[LayerGroup, ...]
    kv_bytes: float

    @classmethod
    def from_model(
        cls,
        model: str | ModelSpec,
        precision: PrecisionSpec,
    ) -> DecodeAttentionCoefficients:
        spec = _spec(model)
        per_session = Activity(
            flops_gemm=sum(
                group.count * 2.0 * group.attention.projection_params
                for group in spec.layer_groups
            ),
            kv_wr=kv_append_bytes_per_token(spec, precision),
            act=sum(
                group.count * 4.0 * spec.hidden_size * precision.activation_bytes
                for group in spec.layer_groups
            ),
        )
        full = tuple(
            group for group in spec.layer_groups
            if group.attention.kind in FULL_ATTENTION_KINDS
        )
        per_context_token = Activity(
            flops_attn=sum(group.count * group.attention.pair_flops for group in full),
            kv_rd=sum(
                group.count * group.attention.kv_elements_per_token * precision.kv_bytes
                for group in full
            ),
        )
        return cls(
            per_session=per_session,
            per_context_token=per_context_token,
            per_step_weight_bytes=sum(
                group.count * group.attention.projection_params * precision.weight_bytes
                for group in spec.layer_groups
            ),
            nonlinear_groups=tuple(
                group for group in spec.layer_groups
                if group.attention.kind not in FULL_ATTENTION_KINDS
            ),
            kv_bytes=precision.kv_bytes,
        )

    def activity(self, contexts: Sequence[int]) -> Activity:
        """Activity for one device serving ``contexts`` (pre-step context lengths)."""

        sessions = len(contexts)
        if sessions == 0:
            return Activity()
        total = (
            self.per_session.scaled(sessions)
            + self.per_context_token.scaled(float(sum(contexts)))
            + Activity(weight_rd=self.per_step_weight_bytes)
        )
        for group in self.nonlinear_groups:
            profile = group.attention
            attended = sum(_attended_items(profile, context) for context in contexts)
            scanned = sum(_index_scan_items(profile, context) for context in contexts)
            total = total + Activity(
                flops_attn=group.count * (
                    profile.pair_flops * attended
                    + profile.index_scan_flops_per_item * scanned
                    + profile.fixed_state_flops * sessions
                ),
                kv_rd=group.count * self.kv_bytes * (
                    attended * profile.kv_elements_per_token
                    + scanned * profile.index_elements_per_token
                    + profile.fixed_state_elements * sessions
                ),
                kv_wr=group.count * self.kv_bytes * profile.fixed_state_elements * sessions,
            )
        return total


def decode_attention_activity(
    model: str | ModelSpec,
    precision: PrecisionSpec,
    contexts: Sequence[int],
) -> Activity:
    """Decode-attention activity of one device for one step."""

    return DecodeAttentionCoefficients.from_model(model, precision).activity(contexts)


def amortized_attention_rate(
    model: str | ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    context_tokens: int,
    reference_batch: int = 1,
) -> float:
    """Context tokens/s for the simulator's linear attention model.

    A device serving ``reference_batch`` sessions per step reads the projection
    weights once for the whole batch, so each session is charged
    ``1/reference_batch`` of them. ``reference_batch=1`` reproduces
    ``estimate_decode_attention`` exactly.
    """

    if isinstance(reference_batch, bool) or not isinstance(reference_batch, int) or reference_batch <= 0:
        raise ValueError("reference_batch must be a positive integer")
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or context_tokens <= 0:
        raise ValueError("context_tokens must be a positive integer")
    spec = _spec(model)
    latency_s = 0.0
    for group in spec.layer_groups:
        profile = group.attention
        attended = _attended_items(profile, context_tokens)
        scanned = _index_scan_items(profile, context_tokens)
        flops = (
            2.0 * profile.projection_params
            + profile.pair_flops * attended
            + profile.index_scan_flops_per_item * scanned
            + profile.fixed_state_flops
        )
        bytes_moved = (
            profile.projection_params * precision.weight_bytes / reference_batch
            + attended * profile.kv_elements_per_token * precision.kv_bytes
            + scanned * profile.index_elements_per_token * precision.kv_bytes
            + 2.0 * profile.fixed_state_elements * precision.kv_bytes
            + 4.0 * spec.hidden_size * precision.activation_bytes
        )
        layer_latency, _, _, _ = roofline_time(flops, bytes_moved, hardware)
        latency_s += group.count * layer_latency
    return context_tokens / latency_s


def ffn_layer_activity(
    model: str | ModelSpec,
    group: LayerGroup,
    precision: PrecisionSpec,
    deployment: DeploymentSpec,
    activation_tokens: float,
    active_experts: int,
    nonrouted_tokens: float | None = None,
) -> Activity:
    """FFN activity of one EP device for one layer.

    ``activation_tokens`` routed (token, expert) copies land on
    ``active_experts`` distinct local experts. Dense layers and shared/common
    parameters use ``nonrouted_tokens / top_k`` original tokens with weights
    sharded evenly over EP, exactly as ``_ffn_layer_work`` does; by default
    ``nonrouted_tokens`` equals ``activation_tokens``, which reproduces the
    estimator. Pass the device's equal share when routing is real, so shared
    work does not follow routing skew.
    """

    spec = _spec(model)
    shared_basis = activation_tokens if nonrouted_tokens is None else nonrouted_tokens
    original_tokens = shared_basis / spec.top_k
    wb = precision.weight_bytes
    ab = precision.activation_bytes
    total = Activity()
    if group.ffn_kind == "dense":
        if shared_basis <= 0:
            return total
        params = spec.dense_active_params
        return Activity(
            flops_gemm=2.0 * original_tokens * params,
            weight_rd=params * wb / deployment.ep,
            act=original_tokens * 3.0 * (spec.hidden_size + spec.dense_intermediate_size) * ab,
        )
    if activation_tokens > 0:
        if active_experts <= 0:
            raise ValueError("routed tokens require at least one active expert")
        active = min(active_experts, activation_tokens)
        matrix = spec.routed_width * spec.expert_intermediate_size
        total = total + Activity(
            flops_gemm=3.0 * 2.0 * activation_tokens * matrix,
            weight_rd=3.0 * active * matrix * wb,
            act=3.0 * activation_tokens * (spec.routed_width + spec.expert_intermediate_size) * ab,
        )
    if shared_basis > 0:
        if spec.shared_active_params:
            total = total + Activity(
                flops_gemm=2.0 * original_tokens * spec.shared_active_params,
                weight_rd=spec.shared_active_params * wb / deployment.ep,
                act=original_tokens * spec.shared_experts * 3.0
                * (spec.hidden_size + spec.shared_intermediate_size) * ab,
            )
        if spec.moe_common_active_params:
            total = total + Activity(
                flops_gemm=2.0 * original_tokens * spec.moe_common_active_params,
                weight_rd=spec.moe_common_active_params * wb / deployment.ep,
                act=original_tokens * 4.0 * spec.hidden_size * ab,
            )
    return total


def ffn_layer_time(
    model: str | ModelSpec,
    group: LayerGroup,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    deployment: DeploymentSpec,
    activation_tokens: int,
    active_experts: int,
) -> float:
    """Roofline FFN time of one EP device for one layer with real routing.

    ``_ffn_layer_work`` only uses its local-expert count as ``min(local, x)``,
    so passing the distinct experts actually hit gives exact weight traffic.
    """

    if activation_tokens == 0:
        return 0.0
    return _ffn_layer_work(
        _spec(model),
        group,
        hardware,
        precision,
        deployment,
        active_experts,
        activation_tokens,
    ).latency_s


def moe_layer_count(model: str | ModelSpec) -> int:
    return sum(group.count for group in _spec(model).layer_groups if group.ffn_kind == "moe")


@dataclass(frozen=True)
class PrefillContextCoefficients:
    """Context-aware prefill work for a batch of (new tokens P_i, context C_i).

    flops = a_tok*T + a_pair*sum(P_i*C_i + P_i*(P_i+1)/2)
    bytes = W_fix + W_rt*coverage(T) + b_ctx*sum_{P_i>0} C_i + b_tok*T

    T = sum(P_i). Weights are read once per batch, routed experts in expectation
    under uniform routing (coverage = 1 - (1 - top_k/E)^T); each session's
    context KV is read once. All coefficients are independent of tensor
    parallelism; a TP group only scales the rates.
    """

    flops_per_token: float
    flops_per_attention_pair: float
    fixed_weight_bytes: float
    routed_weight_bytes: float
    routed_expert_fraction: float
    context_bytes_per_token: float
    bytes_per_prefill_token: float

    @classmethod
    def from_model(
        cls,
        model: str | ModelSpec,
        precision: PrecisionSpec,
    ) -> PrefillContextCoefficients:
        spec = _spec(model)
        unsupported = sorted(
            {
                group.attention.kind
                for group in spec.layer_groups
                if group.attention.kind not in FULL_ATTENTION_KINDS
            }
        )
        if unsupported:
            raise ValueError(
                f"{spec.display_name}: context-aware prefill supports full attention only, "
                f"not {', '.join(unsupported)}"
            )
        flops_per_token = 0.0
        pair_flops = 0.0
        fixed_weights = 0.0
        routed_weights = 0.0
        context_bytes = 0.0
        activation_bytes = 0.0
        for group in spec.layer_groups:
            profile = group.attention
            flops_per_token += group.count * (
                2.0 * profile.projection_params
                + profile.fixed_state_flops
                + 2.0 * _ffn_active_params(spec, group)
            )
            pair_flops += group.count * profile.pair_flops
            if group.ffn_kind == "dense":
                nonrouted = spec.dense_active_params
            else:
                nonrouted = spec.shared_active_params + spec.moe_common_active_params
                routed_weights += (
                    group.count * spec.num_experts * spec.routed_expert_params
                    * precision.weight_bytes
                )
            fixed_weights += group.count * (profile.projection_params + nonrouted) * precision.weight_bytes
            context_bytes += group.count * (
                profile.kv_elements_per_token + profile.index_elements_per_token
            ) * precision.kv_bytes
            activation_bytes += group.count * 6.0 * spec.hidden_size * precision.activation_bytes
        return cls(
            flops_per_token=flops_per_token,
            flops_per_attention_pair=pair_flops,
            fixed_weight_bytes=fixed_weights,
            routed_weight_bytes=routed_weights,
            routed_expert_fraction=spec.top_k / spec.num_experts,
            context_bytes_per_token=context_bytes,
            bytes_per_prefill_token=context_bytes + activation_bytes,
        )

    def expert_coverage(self, tokens: int) -> float:
        if tokens <= 0:
            return 0.0
        if self.routed_expert_fraction >= 1.0:
            return 1.0
        return -math.expm1(tokens * math.log1p(-self.routed_expert_fraction))

    def activity(self, works: Sequence[tuple[int, int]]) -> Activity:
        """``works`` holds (prefill_tokens, context_tokens) per session."""

        tokens = sum(prefill for prefill, _ in works)
        if tokens == 0:
            return Activity()
        pairs = sum(
            prefill * context + prefill * (prefill + 1) / 2.0
            for prefill, context in works
        )
        return Activity(
            flops_gemm=self.flops_per_token * tokens,
            flops_attn=self.flops_per_attention_pair * pairs,
            weight_rd=self.fixed_weight_bytes + self.routed_weight_bytes * self.expert_coverage(tokens),
            kv_rd=self.context_bytes_per_token * sum(
                context for prefill, context in works if prefill > 0
            ),
            kv_wr=self.context_bytes_per_token * tokens,
            act=(self.bytes_per_prefill_token - self.context_bytes_per_token) * tokens,
        )

    def latency_s(self, works: Sequence[tuple[int, int]], hardware: HardwareSpec) -> float:
        work = self.activity(works)
        if work.flops == 0.0 and work.bytes_moved == 0.0:
            return 0.0
        latency, _, _, _ = roofline_time(work.flops, work.bytes_moved, hardware)
        return latency

    def capability_document(self, hardware: HardwareSpec) -> dict[str, float | str]:
        """The simulator's ``roofline_context`` prefill capability for these coefficients."""

        return {
            "model": "roofline_context",
            "flops_per_second": hardware.flops_per_second,
            "memory_bandwidth_bytes_per_second": hardware.memory_bandwidth_bytes_per_second,
            "flops_per_token": self.flops_per_token,
            "flops_per_attention_pair": self.flops_per_attention_pair,
            "fixed_weight_bytes": self.fixed_weight_bytes,
            "routed_weight_bytes": self.routed_weight_bytes,
            "routed_expert_fraction": self.routed_expert_fraction,
            "context_bytes_per_token": self.context_bytes_per_token,
            "bytes_per_prefill_token": self.bytes_per_prefill_token,
        }


def prefill_context_coefficients(
    model: str | ModelSpec,
    precision: PrecisionSpec,
) -> PrefillContextCoefficients:
    return PrefillContextCoefficients.from_model(model, precision)


def prefill_batch_activity(
    coefficients: PrefillContextCoefficients,
    works: Sequence[tuple[int, int]],
) -> Activity:
    return coefficients.activity(works)
