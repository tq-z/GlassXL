#!/usr/bin/env python3
"""GlassML parameter DSE normalized against a fixed NVLink baseline.

Only the selected GlassML hardware parameter changes during a scan. All other
GlassML parameters remain unchanged. NVLink is evaluated with the fixed
NVLINK_HW configuration imported from FFN_compare_glassml_nvlink.py.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from FFN_compare_glassml_nvlink import GLASSML_HW, NVLINK_HW
from FFN_roofline import MODELS, model_one_layer, parse_batches


SCANS = {
    "alltoall_fixed_latency_us": (
        [0.0, 5.0, 10.0, 20.0, 25.0],
        lambda value: f"{value:g} us",
    ),
    "alltoall_min_payload_bytes": (
        [64e3, 4e6, 64e6],
        lambda value: f"{value:g} B",
    ),
    "hbm_bandwidth_tbps": (
        [8.0, 16.0, 32.0, 150.0],
        lambda value: f"{value:g} TB/s",
    ),
    "alltoall_datalink_bytes_per_s": (
        [2e12, 4e12, 8e12, 16e12, 32e12],
        lambda value: f"{value / 1e12:g} TB/s",
    ),
}


def human_bytes(value: float) -> str:
    for scale, suffix in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if value >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}B"


def axis_name(parameter: str) -> str:
    return {"alltoall_fixed_latency_us": "All-to-All fixed latency (µs)", "alltoall_min_payload_bytes": "All-to-All minimum payload", "hbm_bandwidth_tbps": "Weight access bandwidth (TB/s)", "alltoall_datalink_bytes_per_s": "Waveguides per processing unit (#)"}[parameter]


def axis_value(parameter: str, value: float) -> float:
    return value / (2e12/8) if parameter == "alltoall_datalink_bytes_per_s" else value


def axis_tick_label(parameter: str, value: float) -> str:
    if parameter == "alltoall_min_payload_bytes":
        return human_bytes(value)
    if parameter == "alltoall_datalink_bytes_per_s":
        return f"{axis_value(parameter, value):g}"
    return f"{value:g}"


def evaluate_scan(parameter, values, decode_tokens, prefill_tokens, ep, gpus_per_node):
    phases = (("Decode", decode_tokens), ("Prefill", prefill_tokens))
    records = []

    for model in MODELS.values():
        for phase, tokens in phases:
            nvlink_latency = model_one_layer(
                model, NVLINK_HW, tokens, ep, gpus_per_node
            )["total_ms"]
            for value in values:
                glassml_hw = replace(GLASSML_HW, **{parameter: value})
                glassml_latency = model_one_layer(
                    model, glassml_hw, tokens, ep, gpus_per_node
                )["total_ms"]
                records.append(
                    {
                        "parameter": parameter,
                        "parameter_value": value,
                        "model": model.name,
                        "phase": phase,
                        "tokens_per_gpu": tokens,
                        "glassml_latency_ms": glassml_latency,
                        "nvlink_baseline_latency_ms": nvlink_latency,
                        "speedup_vs_nvlink": nvlink_latency / glassml_latency,
                    }
                )

    return pd.DataFrame(records)


def plot_scan(df: pd.DataFrame, parameter: str, output_path: Path) -> None:
    models = list(MODELS.values())
    phases = ["Decode", "Prefill"]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(models)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax, phase in zip(axes, phases):
        phase_df = df[df["phase"] == phase]
        selected_value = getattr(GLASSML_HW, parameter)
        for model, color in zip(models, colors):
            curve = phase_df[phase_df["model"] == model.name].sort_values(
                "parameter_value"
            )
            x_values = curve["parameter_value"].map(lambda value: axis_value(parameter, value))
            ax.plot(
                x_values,
                curve["speedup_vs_nvlink"],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=model.name,
                color=color,
            )
            selected = curve[curve["parameter_value"] == selected_value]
            if not selected.empty:
                ax.scatter(selected["parameter_value"].map(lambda value: axis_value(parameter, value)), selected["speedup_vs_nvlink"], marker="*", s=150, color=color, edgecolor="black", linewidth=0.7, zorder=5, label="Selected GlassML setting" if model is models[0] else "_nolegend_")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(phase)
        scan_values = sorted(phase_df["parameter_value"].unique())
        ax.set_xticks([axis_value(parameter, value) for value in scan_values])
        ax.set_xticklabels([axis_tick_label(parameter, value) for value in scan_values], rotation=30, ha="right")
        ax.set_xlabel(axis_name(parameter))
        ax.grid(True, alpha=0.25)

    for ax in axes:
        ax.set_ylabel("Speedup vs. NVLink baseline")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), fontsize=8, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", default="all", help="Comma-separated scan names or 'all'.")
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--prefill-tokens", type=int, default=65536)
    parser.add_argument("--ep", type=int, default=64)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("ffn_glassml_dse_out"))
    args = parser.parse_args()

    selected = list(SCANS)
    if args.parameters != "all":
        selected = [item.strip() for item in args.parameters.split(",") if item.strip()]
        unknown = sorted(set(selected) - set(SCANS))
        if unknown:
            parser.error(f"unknown parameter(s): {', '.join(unknown)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for parameter in selected:
        values, _ = SCANS[parameter]
        result = evaluate_scan(
            parameter,
            values,
            args.decode_tokens,
            args.prefill_tokens,
            args.ep,
            args.gpus_per_node,
        )
        all_results.append(result)
        result.to_csv(args.out_dir / f"{parameter}_glassml_vs_nvlink.csv", index=False)
        plot_scan(result, parameter, args.out_dir / f"{parameter}_glassml_vs_nvlink.png")

    pd.concat(all_results, ignore_index=True).to_csv(
        args.out_dir / "glassml_dse_vs_nvlink.csv", index=False
    )
    print(f"Saved GlassML DSE figures and CSVs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
