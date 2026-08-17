#!/usr/bin/env python3
"""Compare GlassML and NVLink FFN latency versus system token batch size."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from FFN_roofline import HardwareConfig, MODELS, model_one_layer, parse_batches


# Replace these values with the two hardware parameter sets when available.
GLASSML_HW = replace(
    HardwareConfig(name="GlassML"),
    alltoall_fixed_latency_us=20.0,
    alltoall_min_payload_bytes=64e6,
    alltoall_datalink_bytes_per_s=16e12,
)
NVLINK_HW = replace(
    HardwareConfig(name="NVLink"),
    alltoall_fixed_latency_us=55.0,
    alltoall_min_payload_bytes=64e6,
    alltoall_datalink_bytes_per_s=630e9,
)

DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def evaluate_comparison(batches, ep, gpus_per_node):
    records = []
    for model in MODELS.values():
        for tag, hw in (("GlassML", GLASSML_HW), ("NVLink", NVLINK_HW)):
            for batch_tokens_per_gpu in batches:
                result = model_one_layer(
                    model, hw, batch_tokens_per_gpu, ep, gpus_per_node
                )
                records.append(
                    {
                        "config": tag,
                        "model": model.name,
                        "batch_tokens_per_gpu": batch_tokens_per_gpu,
                        "system_token_batch_size": batch_tokens_per_gpu * ep,
                        "latency_ms": result["total_ms"],
                        "dispatch_ms": result["dispatch_ms"],
                        "gemm_ms": result["gemm_ms"],
                        "combine_ms": result["combine_ms"],
                    }
                )
    return pd.DataFrame(records)


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    models = list(MODELS.values())
    styles = {"GlassML": {"color": "tab:blue"}, "NVLink": {"color": "tab:orange"}}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)

    for ax, model in zip(axes.flat, models):
        model_df = df[df["model"] == model.name]
        for tag, style in styles.items():
            curve = model_df[model_df["config"] == tag].sort_values(
                "system_token_batch_size"
            )
            ax.plot(
                curve["system_token_batch_size"],
                curve["latency_ms"],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=tag,
                **style,
            )
        # Leftmost point: small-batch decode. Rightmost point: prefill region.
        points = model_df.pivot(index="system_token_batch_size", columns="config", values="latency_ms").sort_index()
        for batch_size, region, offset in ((points.index[0], "Decode", (10, 12)), (points.index[-1], "Prefill", (-105, -28))):
            glassml_latency = points.loc[batch_size, "GlassML"]
            nvlink_latency = points.loc[batch_size, "NVLink"]
            speedup = nvlink_latency / glassml_latency
            speedup_ceil = math.ceil(speedup * 10.0) / 10.0
            ax.annotate(f"{region} speedup\n{speedup_ceil:.1f}×", xy=(batch_size, glassml_latency), xytext=offset, textcoords="offset points", fontsize=8, color="tab:blue", ha="left" if offset[0] > 0 else "right", va="bottom" if offset[1] > 0 else "top", bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "tab:blue", "alpha": 0.85}, arrowprops={"arrowstyle": "->", "color": "tab:blue", "lw": 0.9})
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(model.name)
        ax.set_xlabel("System token batch size (tokens across EP GPUs)")
        ax.set_ylabel("FFN layer latency (ms)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

    fig.suptitle("GlassML vs NVLink: FFN latency", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=parse_batches, default=DEFAULT_BATCHES)
    parser.add_argument("--ep", type=int, default=64)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("ffn_compare_out"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = evaluate_comparison(args.batches, args.ep, args.gpus_per_node)
    df.to_csv(args.out_dir / "glassml_vs_nvlink_latency.csv", index=False)
    plot_comparison(df, args.out_dir / "glassml_vs_nvlink_latency.png")
    print(f"Saved comparison CSV and figure to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
