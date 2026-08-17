#!/usr/bin/env python3
"""
FFN/MoE latency roofline model for NVIDIA B100.

Breaks one MoE FFN layer into:
  1. dispatch: local GPU packing / permutation of routed token activations
  2. expert GEMM: grouped SwiGLU expert GEMMs
  3. combine: local GPU weighted reduction / unpermutation

The default model dimensions are taken from public model configurations:
  - DeepSeek-V4-Pro
  - Kimi-K2 1T
  - Qwen3.5-397B-A17B
  - GLM-5.2

Important interpretation:
  batch_tokens = tokens initially resident on EACH GPU before MoE routing.
  With balanced routing and symmetric expert parallelism, each GPU sends and
  receives batch_tokens * top_k assignments. Therefore, the local expert shard
  also processes batch_tokens * top_k token-expert assignments in total.

This is an analytical lower-bound model, not a kernel simulator. Tune the
efficiency, HBM-copy, launch-overhead, and EP parameters against measurements.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_size: int
    expert_intermediate_size: int
    total_routed_experts: int
    top_k: int
    shared_experts: int = 0
    weight_bytes: float = 2.0
    activation_bytes: float = 2.0


@dataclass(frozen=True)
class HardwareConfig:
    name: str = "NVIDIA B100"
    # Editable assumptions. B100 product disclosures vary by system/SKU.
    bf16_peak_tflops: float = 1750.0
    hbm_bandwidth_tbps: float = 8.0
    gemm_efficiency: float = 0.65
    hbm_efficiency: float = 0.75
    kernel_launch_us: float = 3.0
    grouped_gemm_launches: int = 3
    alltoall_fixed_latency_us: float = 55.0
    alltoall_min_payload_bytes: float = 64e6
    alltoall_datalink_bytes_per_s: float = 630e9


MODELS: Dict[str, ModelConfig] = {
    "deepseek-v4-pro": ModelConfig(
        name="DeepSeek-V4-Pro-1.6T",
        hidden_size=7168,
        expert_intermediate_size=3072,
        total_routed_experts=384,
        top_k=6,
        shared_experts=1,
        # weight_bytes=1.0,  # public checkpoint uses quantized expert weights
    ),
    "kimi-k2-1t": ModelConfig(
        name="Kimi-K2-1T",
        hidden_size=7168,
        expert_intermediate_size=2048,
        total_routed_experts=384,
        top_k=8,
        shared_experts=1,
    ),
    "qwen3.5-397b-a17b": ModelConfig(
        name="Qwen3.5-397B-A17B",
        hidden_size=2048,
        expert_intermediate_size=512,
        total_routed_experts=256,
        top_k=8,
        shared_experts=1,
    ),
    "glm-5.2": ModelConfig(
        name="GLM-5.2-754B",
        hidden_size=6144,
        expert_intermediate_size=2048,
        total_routed_experts=256,
        top_k=8,
        shared_experts=1,
    ),
}


# DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_BATCHES = [4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def roofline_time_s(
    flops: float,
    bytes_moved: float,
    hw: HardwareConfig,
) -> tuple[float, float, float, str]:
    compute_s = flops / (hw.bf16_peak_tflops * 1e12 * hw.gemm_efficiency)
    memory_s = bytes_moved / (hw.hbm_bandwidth_tbps * 1e12 * hw.hbm_efficiency)
    bottleneck = "compute" if compute_s >= memory_s else "memory"
    return max(compute_s, memory_s), compute_s, memory_s, bottleneck


def active_local_experts(
    token_expert_assignments: int,
    local_experts: int,
) -> int:
    # Under perfectly even routing, no more experts can be active than assignments.
    return min(local_experts, token_expert_assignments)


def grouped_swiglu_gemm(
    model: ModelConfig,
    hw: HardwareConfig,
    batch_tokens: int,
    expert_parallel_size: int,
) -> dict:
    if expert_parallel_size < 1:
        raise ValueError("expert_parallel_size must be >= 1")
    if expert_parallel_size > model.total_routed_experts:
        raise ValueError("expert_parallel_size cannot exceed total routed experts")

    local_experts = ceil_div(model.total_routed_experts, expert_parallel_size)
    global_tokens = batch_tokens * expert_parallel_size
    assignments = global_tokens * model.top_k / expert_parallel_size
    n_active = active_local_experts(assignments, local_experts)

    # Equal activation assumption. Fractional M is valid for an analytical model.
    m_per_active_expert = assignments / n_active if n_active else 0.0
    d = model.hidden_size
    h = model.expert_intermediate_size
    ab = model.activation_bytes
    wb = model.weight_bytes

    # SwiGLU is modeled as three grouped GEMMs:
    # gate: [M,D] x [D,H]
    # up:   [M,D] x [D,H]
    # down: [M,H] x [H,D]
    # Each matrix multiply counts 2*M*K*N FLOPs.
    gate_flops = 2.0 * assignments * d * h
    up_flops = gate_flops
    down_flops = 2.0 * assignments * h * d

    # Weights are loaded for each active local expert. Activations are counted at
    # HBM boundaries. Fused activation arithmetic is neglected.
    gate_bytes = n_active * d * h * wb + assignments * (d + h) * ab
    up_bytes = n_active * d * h * wb + assignments * (d + h) * ab
    down_bytes = n_active * h * d * wb + assignments * (h + d) * ab

    rows = []
    for op, flops, moved in (
        ("gate_proj", gate_flops, gate_bytes),
        ("up_proj", up_flops, up_bytes),
        ("down_proj", down_flops, down_bytes),
    ):
        t, tc, tm, bound = roofline_time_s(flops, moved, hw)
        rows.append(
            {
                "op": op,
                "flops": flops,
                "bytes": moved,
                "compute_s": tc,
                "memory_s": tm,
                "roofline_s": t,
                "bound": bound,
                "arithmetic_intensity_flop_per_byte": flops / moved if moved else math.inf,
            }
        )

    gemm_s = sum(r["roofline_s"] for r in rows)
    gemm_s += hw.grouped_gemm_launches * hw.kernel_launch_us * 1e-6

    return {
        "local_experts": local_experts,
        "active_local_experts": n_active,
        "assignments": assignments,
        "tokens_per_active_expert": m_per_active_expert,
        "gemm_s": gemm_s,
        "gemm_rows": rows,
    }


def alltoall_time_s(
    model: ModelConfig,
    hw: HardwareConfig,
    batch_tokens_per_gpu: int,
    expert_parallel_size: int,
    gpus_per_node: int,
) -> tuple[float, float, int, float, float]:
    """Model one dispatch/combine all-to-all at the node level."""
    if gpus_per_node < 1:
        raise ValueError("gpus_per_node must be >= 1")
    if expert_parallel_size < 1 or expert_parallel_size % gpus_per_node:
        raise ValueError(
            "expert_parallel_size must be positive and divisible by gpus_per_node"
        )

    system_concurrency = batch_tokens_per_gpu * expert_parallel_size
    assignments_per_gpu = system_concurrency * model.top_k / expert_parallel_size
    tokens_per_node = batch_tokens_per_gpu * gpus_per_node
    assignments_per_node = tokens_per_node * model.top_k
    payload_bytes = assignments_per_node * model.hidden_size * model.activation_bytes
    nodes = expert_parallel_size // gpus_per_node
    transfer_s = (
        max(payload_bytes, hw.alltoall_min_payload_bytes)
        / hw.alltoall_datalink_bytes_per_s
    )
    latency_s = hw.alltoall_fixed_latency_us * 1e-6 + transfer_s
    return (
        latency_s,
        payload_bytes,
        nodes,
        system_concurrency,
        assignments_per_gpu,
    )


def dispatch_time_s(
    model: ModelConfig,
    hw: HardwareConfig,
    batch_tokens_per_gpu: int,
    expert_parallel_size: int,
    gpus_per_node: int,
) -> tuple[float, float, int, float, float]:
    return alltoall_time_s(
        model, hw, batch_tokens_per_gpu, expert_parallel_size, gpus_per_node
    )


def combine_time_s(
    model: ModelConfig,
    hw: HardwareConfig,
    batch_tokens_per_gpu: int,
    expert_parallel_size: int,
    gpus_per_node: int,
) -> tuple[float, float, int, float, float]:
    return alltoall_time_s(
        model, hw, batch_tokens_per_gpu, expert_parallel_size, gpus_per_node
    )


def model_one_layer(
    model: ModelConfig,
    hw: HardwareConfig,
    batch_tokens_per_gpu: int,
    expert_parallel_size: int,
    gpus_per_node: int,
) -> dict:
    dispatch_s, dispatch_bytes, nodes, system_concurrency, assignments_per_gpu = (
        dispatch_time_s(
            model, hw, batch_tokens_per_gpu, expert_parallel_size, gpus_per_node
        )
    )
    gemm = grouped_swiglu_gemm(
        model, hw, batch_tokens_per_gpu, expert_parallel_size
    )
    combine_s, combine_bytes, _, _, _ = combine_time_s(
        model, hw, batch_tokens_per_gpu, expert_parallel_size, gpus_per_node
    )

    total_s = dispatch_s + gemm["gemm_s"] + combine_s
    return {
        "model": model.name,
        "batch_tokens_per_gpu": batch_tokens_per_gpu,
        "system_concurrency_tokens": system_concurrency,
        "assignments_per_gpu": assignments_per_gpu,
        "ep": expert_parallel_size,
        "gpus_per_node": gpus_per_node,
        "nodes": nodes,
        "local_experts": gemm["local_experts"],
        "active_local_experts": gemm["active_local_experts"],
        "tokens_per_active_expert": gemm["tokens_per_active_expert"],
        "dispatch_ms": dispatch_s * 1e3,
        "gemm_ms": gemm["gemm_s"] * 1e3,
        "combine_ms": combine_s * 1e3,
        "total_ms": total_s * 1e3,
        "dispatch_pct": 100.0 * dispatch_s / total_s,
        "gemm_pct": 100.0 * gemm["gemm_s"] / total_s,
        "combine_pct": 100.0 * combine_s / total_s,
        "dispatch_GB": dispatch_bytes / 1e9,
        "combine_GB": combine_bytes / 1e9,
        "gemm_detail": gemm["gemm_rows"],
    }


def parse_batches(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("batches must be positive comma-separated integers")
    return values


def plot_breakdown(
    df: pd.DataFrame, out_dir: Path, plot_mode: str = "latency"
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, group in df.groupby("model", sort=False):
        group = group.sort_values("batch_tokens_per_gpu")
        x = np.arange(len(group))
        width = 0.72

        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        if plot_mode == "percentage":
            dispatch = group["dispatch_pct"].to_numpy()
            gemm = group["gemm_pct"].to_numpy()
            combine = group["combine_pct"].to_numpy()
            ylabel = "Latency breakdown (%)"
            suffix = "_ffn_breakdown_pct.png"
        else:
            dispatch = group["dispatch_ms"].to_numpy()
            gemm = group["gemm_ms"].to_numpy()
            combine = group["combine_ms"].to_numpy()
            ylabel = "Latency per MoE FFN layer (ms)"
            suffix = "_ffn_breakdown.png"

        ax.bar(x, dispatch, width, label="Dispatch")
        ax.bar(x, gemm, width, bottom=dispatch, label="Expert GEMM")
        ax.bar(x, combine, width, bottom=dispatch + gemm, label="Combine")
        ax.set_xticks(x, group["system_concurrency_tokens"].astype(str))
        ax.set_xlabel("System concurrency (tokens across EP GPUs)")
        ax.set_ylabel(ylabel)
        title_suffix = "percentage" if plot_mode == "percentage" else "latency"
        ax.set_title(f"{model_name}: B100 FFN {title_suffix} breakdown")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()

        safe = (
            model_name.lower()
            .replace(" ", "_")
            .replace(".", "_")
            .replace("-", "_")
        )
        fig.savefig(out_dir / f"{safe}{suffix}", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B100 roofline latency breakdown for MoE FFN layers."
    )
    parser.add_argument(
        "--models",
        default="all",
        help=f"Comma-separated: {','.join(MODELS)} or 'all'",
    )
    parser.add_argument(
        "--batches",
        type=parse_batches,
        default=DEFAULT_BATCHES,
        help="Comma-separated tokens per GPU before routing.",
    )
    parser.add_argument("--ep", type=int, default=64, help="Expert-parallel GPU count.")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="GPUs in each node.")
    parser.add_argument(
        "--plot-mode",
        choices=("latency", "percentage"),
        default="percentage",
        help="Plot absolute latency or normalized percentage breakdown.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("ffn_roofline_out"))
    args = parser.parse_args()

    selected = list(MODELS)
    if args.models != "all":
        selected = [x.strip().lower() for x in args.models.split(",") if x.strip()]
        unknown = sorted(set(selected) - set(MODELS))
        if unknown:
            parser.error(f"unknown model(s): {', '.join(unknown)}")

    hw = HardwareConfig()

    records = []
    gemm_records = []
    for key in selected:
        model = MODELS[key]
        for batch in args.batches:
            result = model_one_layer(model, hw, batch, args.ep, args.gpus_per_node)
            detail = result.pop("gemm_detail")
            records.append(result)
            for row in detail:
                gemm_records.append(
                    {
                        "model": model.name,
                        "batch_tokens_per_gpu": batch,
                        "system_concurrency_tokens": batch * args.ep,
                        "ep": args.ep,
                        **row,
                    }
                )

    df = pd.DataFrame(records)
    gemm_df = pd.DataFrame(gemm_records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "ffn_latency_breakdown.csv", index=False)
    gemm_df.to_csv(args.out_dir / "ffn_gemm_roofline_detail.csv", index=False)
    plot_breakdown(df, args.out_dir, args.plot_mode)

    cols = [
        "model",
        "batch_tokens_per_gpu",
        "system_concurrency_tokens",
        "assignments_per_gpu",
        "ep",
        "local_experts",
        "tokens_per_active_expert",
        "dispatch_ms",
        "gemm_ms",
        "combine_ms",
        "total_ms",
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSaved CSVs and plots to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()