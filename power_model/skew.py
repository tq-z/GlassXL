"""Skew-aware FFN latency curves for the simulator, from real routing traces.

The simulator splits a step's A = B*top_k routed activations evenly
(``ExpertMapping.equal_loads``) and takes the slowest device's curve value as
the FFN barrier. Real routing is skewed per layer, and each layer has its own
barrier, so the real FFN time is sum_l max_d t(x_ld, k_ld). This module
measures that time from i.i.d. real decode rows and writes it into per-device
curves so the simulator's barrier reproduces it, without changing the simulator.

Construction: adding one session adds top_k activations, which raise the local
count x_d of at least one device. Give each device the curve
c_d(x) = T_real(first batch at which device d reaches x). With T_real
nondecreasing, every other device's value is at most T_real(B), and the device
that just incremented returns exactly T_real(B), so max_d c_d(x_d(B)) = T_real(B)
at every batch size.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from roofline_model import DeploymentSpec, HardwareSpec, ModelSpec, PrecisionSpec
from roofline_model.activity import moe_layer_count

from .routing import DecodeRowPool, layer_device_loads, layer_folded_time


def batch_grid(max_batch: int) -> tuple[int, ...]:
    values = list(range(1, min(32, max_batch) + 1))
    batch = 32.0
    while batch < max_batch:
        batch *= 1.25
        values.append(min(max_batch, int(round(batch))))
    return tuple(sorted(set(values)))


def equal_local_tokens(activation_tokens: int, experts_by_device: Mapping[int, Sequence[int]], num_experts: int) -> dict[int, int]:
    """Per-device local activations exactly as ``ExpertMapping.equal_loads`` assigns them."""

    quotient, remainder = divmod(activation_tokens, num_experts)
    return {
        device: sum(quotient + (expert < remainder) for expert in experts)
        for device, experts in experts_by_device.items()
    }


@dataclass(frozen=True)
class SkewRow:
    batch: int
    activation_tokens: int
    real_p5_s: float
    real_p50_s: float
    real_p95_s: float
    real_mean_s: float


def measure_real_ffn_time(
    pool: DecodeRowPool,
    experts_by_device: Mapping[int, Sequence[int]],
    model: ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    deployment: DeploymentSpec,
    *,
    batches: Sequence[int],
    samples: int,
    seed: int,
) -> list[SkewRow]:
    devices = sorted(experts_by_device)
    column = {device: index for index, device in enumerate(devices)}
    owner = np.full(model.num_experts, -1, dtype=np.int64)
    for device, experts in experts_by_device.items():
        owner[list(experts)] = column[device]
    if (owner < 0).any():
        raise ValueError("every expert needs an owner")
    if pool.num_layers != moe_layer_count(model):
        raise ValueError("routing traces and model disagree on routed layer count")
    rows = pool.all_rows()
    rng = np.random.default_rng(seed)
    result: list[SkewRow] = []
    for batch in batches:
        share = batch * model.top_k / len(devices)
        times = np.array(
            [
                layer_folded_time(
                    *layer_device_loads(rows[rng.integers(0, len(rows), batch)], owner, len(devices)),
                    share, model, hardware, precision, deployment,
                )
                for _ in range(samples)
            ]
        )
        result.append(
            SkewRow(
                batch=batch,
                activation_tokens=batch * model.top_k,
                real_p5_s=float(np.percentile(times, 5)),
                real_p50_s=float(np.percentile(times, 50)),
                real_p95_s=float(np.percentile(times, 95)),
                real_mean_s=float(times.mean()),
            )
        )
    return result


def real_time_by_batch(rows: Sequence[SkewRow]) -> np.ndarray:
    """T_real for every batch 1..max, interpolated between measured sizes, nondecreasing."""

    batches = np.array([row.batch for row in rows], dtype=float)
    times = np.maximum.accumulate(np.array([row.real_mean_s for row in rows]))
    every = np.arange(1, int(batches[-1]) + 1, dtype=float)
    return np.maximum.accumulate(np.interp(every, batches, times))


def skew_aware_device_curves(
    rows: Sequence[SkewRow],
    experts_by_device: Mapping[int, Sequence[int]],
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    exact_up_to_batch: int = 512,
) -> dict[int, list[tuple[int, float]]]:
    """Per FFN device, curve points (local activations, seconds).

    Every increment is kept up to ``exact_up_to_batch``; beyond it only
    increments landing on multiples of the device's expert count are kept and
    the curve interpolates between them, where T_real is smooth.
    """

    real = real_time_by_batch(rows)
    counts = {device: len(experts) for device, experts in experts_by_device.items()}
    curves: dict[int, list[tuple[int, float]]] = {device: [(0, 0.0)] for device in experts_by_device}
    reached = {device: 0 for device in experts_by_device}
    last_batch = len(real)
    for batch in range(1, last_batch + 1):
        local = equal_local_tokens(batch * model.top_k, experts_by_device, model.num_experts)
        for device, tokens in local.items():
            if tokens <= reached[device]:
                continue
            reached[device] = tokens
            if batch <= exact_up_to_batch or tokens % counts[device] == 0 or batch == last_batch:
                curves[device].append((tokens, float(real[batch - 1])))
    # Past the measured range: at least compute-bound routed GEMMs per local activation
    # over all MoE layers, and never below the device's own last measured slope, which
    # carries the per-layer straggler (the balanced slope alone underestimates it).
    balanced = moe_layer_count(model) * 6.0 * model.routed_width * model.expert_intermediate_size / hardware.flops_per_second
    for points in curves.values():
        last_x, last_t = points[-1]
        slope = balanced
        if len(points) >= 3:
            (prev_x, prev_t), _ = points[-2], points[-1]
            if last_x > prev_x:
                slope = max(balanced, (last_t - prev_t) / (last_x - prev_x))
        points.append((2 * max(last_x, 1), last_t + slope * max(last_x, 1)))
    return curves


def curve_latency(points: Sequence[tuple[int, float]], tokens: int) -> float:
    if tokens <= 0:
        return 0.0
    for (left_x, left_t), (right_x, right_t) in zip(points, points[1:]):
        if tokens <= right_x:
            return left_t + (tokens - left_x) / (right_x - left_x) * (right_t - left_t)
    (left_x, left_t), (right_x, right_t) = points[-2], points[-1]
    return right_t + (tokens - right_x) * (right_t - left_t) / (right_x - left_x)


def simulator_barrier(
    activation_tokens: int,
    experts_by_device: Mapping[int, Sequence[int]],
    num_experts: int,
    curve_for_device,
) -> float:
    locals_ = equal_local_tokens(activation_tokens, experts_by_device, num_experts)
    return max(curve_for_device(device, tokens) for device, tokens in locals_.items())


def validation_table(
    rows: Sequence[SkewRow],
    experts_by_device: Mapping[int, Sequence[int]],
    num_experts: int,
    before,
    after,
) -> list[dict[str, float]]:
    """Simulator barrier (balanced vs skew-aware curves) against the measured real time.

    ``skew_aware_over_monotone`` checks the construction (the curves encode the
    running-max real time); ``skew_aware_over_real`` also includes sampling noise.
    """

    monotone = real_time_by_batch(rows)
    table = []
    for row in rows:
        des_before = simulator_barrier(row.activation_tokens, experts_by_device, num_experts, before)
        des_after = simulator_barrier(row.activation_tokens, experts_by_device, num_experts, after)
        target = float(monotone[row.batch - 1])
        table.append(
            {
                "batch": row.batch,
                "activation_tokens": row.activation_tokens,
                "real_p5_s": row.real_p5_s,
                "real_p50_s": row.real_p50_s,
                "real_p95_s": row.real_p95_s,
                "real_mean_s": row.real_mean_s,
                "real_monotone_s": target,
                "simulator_balanced_s": des_before,
                "simulator_skew_aware_s": des_after,
                "skew_aware_over_monotone": des_after / target if target > 0 else math.nan,
                "skew_aware_over_real": des_after / row.real_mean_s if row.real_mean_s > 0 else math.nan,
                "balanced_over_real": des_before / row.real_mean_s if row.real_mean_s > 0 else math.nan,
            }
        )
    return table
