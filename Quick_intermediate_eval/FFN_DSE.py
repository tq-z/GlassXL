#!/usr/bin/env python3
"""Design-space exploration for the FFN/MoE latency roofline model.

Each figure scans one hardware/communication parameter. Models are grouped on
the x-axis, scan points are neighboring bars, and the y-axis reports speedup
relative to the default HardwareConfig value. Decode and prefill use one fixed
token count each rather than sweeping token counts.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from FFN_roofline import HardwareConfig, MODELS, ModelConfig, model_one_layer


FIXED_LATENCY_US = [0.0, 5.0, 10.0, 20.0, 25.0, 55.0][::-1]
MIN_PAYLOAD_BYTES = [64*64, 64e3, 4*64e3, 64*64e3, 256*64e3, 64e6][::-1]
HBM_BANDWIDTH_TBPS = [8.0, 16.0, 32.0, 150.0, 0.5]
ALLTOALL_DATALINK_BYTES_PER_S = [630e9, 1e12, 1.5e12, 32e12, 64e16]
BATCH_SCAN_DEFAULTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def human_bytes(value: float) -> str:
    for scale, suffix in (
        (1e9, "GB"),
        (1e6, "MB"),
        (1e3, "KB"),
    ):
        if value >= scale:
            amount = value / scale
            return f"{amount:g}{suffix}"
    return f"{value:g}B"


def evaluate_scan(
    parameter: str,
    values: Iterable[float],
    formatter: Callable[[float], str],
    small_decode_tokens: int,
    large_decode_tokens: int,
    prefill_tokens: int,
    ep: int,
    gpus_per_node: int,
) -> pd.DataFrame:
    """Evaluate one parameter scan and normalize against its default value."""
    default_hw = HardwareConfig()
    default_value = getattr(default_hw, parameter)
    phases = (
        ("Small-batch Decode", small_decode_tokens),
        ("Large-batch Decode", large_decode_tokens),
        ("Prefill", prefill_tokens),
    )
    records: list[dict] = []

    for model in MODELS.values():
        baseline_by_phase = {
            phase: model_one_layer(model, default_hw, tokens, ep, gpus_per_node)[
                "total_ms"
            ]
            for phase, tokens in phases
        }

        for value in values:
            candidate_hw = replace(default_hw, **{parameter: value})
            for phase, tokens in phases:
                result = model_one_layer(
                    model, candidate_hw, tokens, ep, gpus_per_node
                )
                latency_ms = result["total_ms"]
                records.append(
                    {
                        "parameter": parameter,
                        "parameter_value": value,
                        "parameter_label": formatter(value),
                        "is_default": value == default_value,
                        "model": model.name,
                        "phase": phase,
                        "tokens_per_gpu": tokens,
                        "system_concurrency_tokens": tokens * ep,
                        "latency_ms": latency_ms,
                        "baseline_latency_ms": baseline_by_phase[phase],
                        "speedup": baseline_by_phase[phase] / latency_ms,
                    }
                )

    return pd.DataFrame(records)


def plot_scan(
    df: pd.DataFrame,
    parameter_title: str,
    output_path: Path,
) -> None:
    """Plot decode and prefill speedups as model clusters of neighboring bars."""
    phases = ["Small-batch Decode", "Large-batch Decode", "Prefill"]
    models = list(MODELS.values())
    labels = list(dict.fromkeys(df["parameter_label"]))
    x = np.arange(len(models), dtype=float)
    group_width = 0.82
    bar_width = group_width / len(labels)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5), sharey=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(labels)))

    for ax, phase in zip(axes, phases):
        phase_df = df[df["phase"] == phase]
        for index, (label, color) in enumerate(zip(labels, colors)):
            values = []
            default_flags = []
            for model in models:
                row = phase_df[
                    (phase_df["model"] == model.name)
                    & (phase_df["parameter_label"] == label)
                ].iloc[0]
                values.append(row["speedup"])
                default_flags.append(bool(row["is_default"]))

            positions = x - group_width / 2 + (index + 0.5) * bar_width
            bars = ax.bar(
                positions,
                values,
                width=bar_width * 0.92,
                label=f"{label}{' (default)' if any(default_flags) else ''}",
                color=color,
                edgecolor="black" if any(default_flags) else "none",
                linewidth=1.2 if any(default_flags) else 0.0,
            )
            for bar, speedup in zip(bars, values):
                ax.annotate(
                    f"{speedup:.2f}×",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xticks(x, [model.name for model in models], rotation=15, ha="right")
        ax.set_title(phase)
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Speedup vs. default total FFN latency")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(labels),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(parameter_title, y=1.04, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_curves(df: pd.DataFrame, parameter: str, parameter_title: str, output_path: Path) -> None:
    """Plot model curves normalized to the default NVLink baseline."""
    phases = ["Small-batch Decode", "Prefill"]
    models = list(MODELS.values())
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(models)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, phase in zip(axes, phases):
        phase_df = df[df["phase"] == phase]
        for model, color in zip(models, colors):
            curve = phase_df[phase_df["model"] == model.name].sort_values("parameter_value")
            ax.plot(curve["parameter_value"], curve["speedup"], marker="o", markersize=4, linewidth=1.8, label=model.name, color=color)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(phase)
        ax.set_xlabel(parameter_title.replace(" DSE", " scan"))
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Speedup vs. NVLink baseline")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), fontsize=8, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"{parameter_title}: model speedup curves", y=1.04, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_batch_scan(parameter, values, formatter, batches, ep, gpus_per_node):
    """Evaluate FFN latency versus system batch size for one HW parameter."""
    default_hw = HardwareConfig()
    records = []
    for model in MODELS.values():
        for value in values:
            candidate_hw = replace(default_hw, **{parameter: value})
            for batch_tokens_per_gpu in batches:
                result = model_one_layer(model, candidate_hw, batch_tokens_per_gpu, ep, gpus_per_node)
                records.append({
                    "parameter": parameter, "parameter_value": value,
                    "parameter_label": formatter(value),
                    "is_default": value == getattr(default_hw, parameter),
                    "model": model.name, "batch_tokens_per_gpu": batch_tokens_per_gpu,
                    "system_batch_size": batch_tokens_per_gpu * ep,
                    "latency_ms": result["total_ms"],
                    "system_tqps": batch_tokens_per_gpu * ep * 1000.0 / result["total_ms"], "dispatch_ms": result["dispatch_ms"],
                    "gemm_ms": result["gemm_ms"], "combine_ms": result["combine_ms"],
                })
    return pd.DataFrame(records)


def plot_batch_scan(df, parameter_title, output_path):
    """Plot one latency curve per hardware configuration for each model."""
    models = list(MODELS.values())
    labels = list(dict.fromkeys(df["parameter_label"]))
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(labels)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    for ax, model in zip(axes.flat, models):
        model_df = df[df["model"] == model.name]
        for label, color in zip(labels, colors):
            curve = model_df[model_df["parameter_label"] == label].sort_values("system_tqps")
            default = bool(curve["is_default"].any())
            curve_label = label + (" (default)" if default else "")
            ax.plot(curve["system_tqps"], curve["latency_ms"], marker="o", markersize=3.5,
                    linewidth=1.6 if default else 1.1, label=curve_label, color=color)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_title(model.name); ax.set_xlabel("System TQPS (tokens/s)")
        ax.set_ylabel("FFN layer latency (ms)"); ax.grid(True, which="both", alpha=0.25)
    handles, legend_labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(parameter_title, y=1.02, fontsize=13); fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=220, bbox_inches="tight"); plt.close(fig)


def parse_positive_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("value must be a comma-separated list of positive integers")
    return values


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FFN design-space exploration with fixed decode/prefill sizes."
    )
    parser.add_argument("--decode-tokens", type=positive_int, default=1)
    parser.add_argument("--large-decode-tokens", type=positive_int, default=256)
    parser.add_argument("--prefill-tokens", type=positive_int, default=65536)
    parser.add_argument("--ep", type=positive_int, default=64)
    parser.add_argument("--gpus-per-node", type=positive_int, default=1)
    parser.add_argument("--batch-scan", action="store_true", default=False, help="Also scan system batch size and plot latency curves per HW config.")
    parser.add_argument("--curve-plot", action="store_true",default=True, help="Create normalized speedup curve figures for decode and prefill.")
    parser.add_argument("--batch-sizes", type=parse_positive_int_list, default=BATCH_SCAN_DEFAULTS, help="Comma-separated batch_tokens_per_gpu values; x-axis is derived system TQPS.")
    parser.add_argument("--out-dir", type=Path, default=Path("ffn_dse_out"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scans = (
        (
            "alltoall_fixed_latency_us",
            FIXED_LATENCY_US,
            lambda value: f"{value:g}µs",
            "All-to-All Fixed-Latency DSE",
        ),
        (
            "alltoall_min_payload_bytes",
            MIN_PAYLOAD_BYTES,
            lambda value: f"{human_bytes(value / args.ep)}",
            "All-to-All Minimum Saturation-Payload DSE",
        ),
        (
            "hbm_bandwidth_tbps",
            HBM_BANDWIDTH_TBPS,
            lambda value: f"{value:g} TB/s",
            "HBM Bandwidth DSE",
        ),
        (
            "alltoall_datalink_bytes_per_s",
            ALLTOALL_DATALINK_BYTES_PER_S,
            lambda value: (
                f"{value / 1e9:g} GB/s"
                if value < 1e12
                else f"{value / 1e12:g} TB/s"
            ),
            "All-to-All Datalink Bandwidth DSE",
        ),
    )

    all_results = []
    for parameter, values, formatter, title in scans:
        result = evaluate_scan(
            parameter,
            values,
            formatter,
            args.decode_tokens,
            args.large_decode_tokens,
            args.prefill_tokens,
            args.ep,
            args.gpus_per_node,
        )
        all_results.append(result)
        plot_scan(result, title, args.out_dir / f"{parameter}_speedup.png")
        if args.curve_plot:
            plot_speedup_curves(result, parameter, title, args.out_dir / f"{parameter}_speedup_curve.png")
        if args.batch_scan:
            batch_result = evaluate_batch_scan(parameter, values, formatter, args.batch_sizes, args.ep, args.gpus_per_node)
            batch_result.to_csv(args.out_dir / f"{parameter}_batch_latency.csv", index=False)
            plot_batch_scan(batch_result, f"{title}: latency vs. system batch size", args.out_dir / f"{parameter}_batch_latency.png")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.out_dir / "ffn_dse_results.csv", index=False)
    print(f"Saved DSE CSV and figures to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
