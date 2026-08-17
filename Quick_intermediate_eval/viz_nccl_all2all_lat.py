#!/usr/bin/env python3
"""
Parse and visualize NCCL all-to-all benchmark results.

Expected input columns:

gpu  num_nodes  num_gpus_per_node  dtype  size(kb)  time(us)

Example:
B200-SXM-192GB  1  8  bfloat16  4096  33

Usage:
    python plot_nccl_alltoall.py results.txt

Optional:
    python plot_nccl_alltoall.py results.txt --output-dir figures
    python plot_nccl_alltoall.py results.txt --max-small-kb 65536
    cat results.txt | python plot_nccl_alltoall.py -
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from FFN_roofline import DEFAULT_BATCHES, MODELS


REQUIRED_COLUMNS = {
    "gpu",
    "num_nodes",
    "num_gpus_per_node",
    "dtype",
    "size_kb",
    "time_us",
}


def normalize_column_name(name: str) -> str:
    """Convert headers such as size(kb) and time(us) into Python-friendly names."""
    name = name.strip().lower()
    name = re.sub(r"\(kb\)", "_kb", name)
    name = re.sub(r"\(us\)", "_us", name)
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    return name.strip("_")


def parse_results(path: str) -> pd.DataFrame:
    """
    Parse a whitespace- or tab-delimited benchmark file.

    The parser accepts:
    - tab-separated text;
    - multiple-space-separated text;
    - stdin when path is "-".
    """
    if path == "-":
        text = sys.stdin.read()
    else:
        text = Path(path).read_text(encoding="utf-8")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    if not lines:
        raise ValueError("Input file is empty.")

    from io import StringIO

    df = pd.read_csv(StringIO("\n".join(lines)), skipinitialspace=True)
    df.columns = [normalize_column_name(str(col)) for col in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Parsed columns: {list(df.columns)}"
        )

    numeric_columns = [
        "num_nodes",
        "num_gpus_per_node",
        "size_kb",
        "time_us",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df["world_size"] = df["num_nodes"] * df["num_gpus_per_node"]

    # GiB/s using binary KB:
    #
    # size_bytes = size_kb * 1024
    # time_s     = time_us * 1e-6
    # GiB/s      = size_bytes / time_s / 2^30
    df["effective_bandwidth_gib_s"] = (
        df["size_kb"] * 1024.0
        / (df["time_us"] * 1e-6)
        / (1024.0**3)
    )

    # Approximate bytes sent to each rank if size_kb is the total
    # all-to-all send buffer per rank.
    df["peer_size_kb"] = df["size_kb"] / df["world_size"]

    df["configuration"] = df.apply(
        lambda row: (
            # f'{int(row["num_nodes"])} node × '
            f'{int(row["num_gpus_per_node"])} GPUs '
            f'(world={int(row["world_size"])})'
        ),
        axis=1,
    )

    return df.sort_values(
        ["world_size", "num_nodes", "num_gpus_per_node", "size_kb"]
    ).reset_index(drop=True)


def plot_latency(df: pd.DataFrame, output_path: Path) -> None:
    """Plot latency across the full message-size range."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for configuration, group in df.groupby("configuration", sort=False):
        group = group.sort_values("size_kb")

        ax.plot(
            group["size_kb"],
            group["time_us"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=configuration,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Total message size per rank (KB)")
    ax.set_ylabel("All-to-all latency (μs)")
    ax.set_title("NCCL All-to-All Latency")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_bandwidth(df: pd.DataFrame, output_path: Path) -> None:
    """Plot effective algorithmic bandwidth."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for configuration, group in df.groupby("configuration", sort=False):
        group = group.sort_values("size_kb")

        ax.plot(
            group["size_kb"],
            group["effective_bandwidth_gib_s"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=configuration,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Total message size per rank (KB)")
    ax.set_ylabel("Effective bandwidth (GiB/s)")
    ax.set_title("NCCL All-to-All Effective Bandwidth")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_small_message_zoom(
    df: pd.DataFrame,
    output_path: Path,
    max_small_kb: int,
) -> None:
    """Zoom into the startup-dominated region and preserve latency cliffs."""
    small = df[df["size_kb"] <= max_small_kb].copy()

    if small.empty:
        print(
            f"Warning: no data points at or below {max_small_kb} KB; "
            "small-message plot was skipped.",
            file=sys.stderr,
        )
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for configuration, group in small.groupby("configuration", sort=False):
        group = group.sort_values("size_kb")

        ax.plot(
            group["size_kb"],
            group["time_us"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=configuration,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Total message size per rank (KB)")
    ax.set_ylabel("All-to-all latency (μs)")
    ax.set_title(f"Small-Message NCCL All-to-All Latency (≤ {max_small_kb:,} KB)")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def format_peer_payload(value_kb: float, _position: int) -> str:
    """Format the logarithmic peer-payload axis using human-readable units."""
    value_bytes = value_kb * 1024.0
    units = ((1024.0**3, "GB"), (1024.0**2, "MB"), (1024.0, "KB"), (1.0, "B"))
    for scale, suffix in units:
        if value_bytes >= scale:
            value = value_bytes / scale
            if value >= 100 or value.is_integer():
                return f"{value:.0f}{suffix}"
            if value >= 10:
                return f"{value:.1f}{suffix}"
            return f"{value:.2f}{suffix}"
    return "0B"


def add_peer_annotations(ax, group: pd.DataFrame, color: str, label_index: int) -> None:
    """Annotate startup latency and the highest measured bandwidth on a curve."""
    group = group.sort_values("peer_size_kb")
    small = group.iloc[0]
    saturated = group.loc[group["effective_bandwidth_gib_s"].idxmax()]
    offset = 10 + (label_index % 3) * 11

    ax.annotate(
        f"small: {small["time_us"]:.0f} μs",
        xy=(small["peer_size_kb"], small["time_us"]),
        xytext=(8, offset),
        textcoords="offset points",
        color=color,
        fontsize=7,
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
    )
    ax.annotate(
        f"sat BW: {saturated["effective_bandwidth_gib_s"]:.0f} GiB/s",
        xy=(saturated["peer_size_kb"], saturated["time_us"]),
        xytext=(-8, -(offset + 5)),
        textcoords="offset points",
        ha="right",
        va="top",
        color=color,
        fontsize=7,
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
    )


def add_ffn_token_table(fig) -> None:
    return
    """Place FFN model token sweeps beneath and aligned with the x-axis."""
    token_values = ", ".join(str(value) for value in DEFAULT_BATCHES)
    rows = [[model.name, token_values] for model in MODELS.values()]
    table = fig.axes[0].table(
        cellText=rows,
        colLabels=["FFN model", "tokens / GPU"],
        colWidths=[0.30, 0.70],
        cellLoc="left",
        bbox=[0.0, -0.53, 1.0, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")


def plot_peer_size(df: pd.DataFrame, output_path: Path) -> None:
    """Plot latency against approximate message size per destination rank."""
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for label_index, (configuration, group) in enumerate(df.groupby("configuration", sort=False)):
        group = group.sort_values("peer_size_kb")
        color = colors[label_index % len(colors)]
        ax.plot(
            group["peer_size_kb"],
            group["time_us"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=configuration,
            color=color,
        )
        add_peer_annotations(ax, group, color, label_index)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(format_peer_payload))
    ax.set_xlabel("Approximate payload per destination rank")
    ax.set_ylabel("All-to-all latency (μs)")
    ax.set_title("NCCL All-to-All Latency vs. Per-Peer Payload")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=8)
    fig.subplots_adjust(bottom=0.36)
    add_ffn_token_table(fig)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of each configuration."""
    print("\nParsed configurations:")

    summary = (
        df.groupby(
            [
                "gpu",
                "num_nodes",
                "num_gpus_per_node",
                "world_size",
                "dtype",
            ],
            as_index=False,
        )
        .agg(
            min_size_kb=("size_kb", "min"),
            max_size_kb=("size_kb", "max"),
            min_latency_us=("time_us", "min"),
            max_latency_us=("time_us", "max"),
            max_bandwidth_gib_s=("effective_bandwidth_gib_s", "max"),
            samples=("size_kb", "count"),
        )
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        160,
        "display.float_format",
        lambda value: f"{value:.2f}",
    ):
        print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse and visualize NCCL all-to-all benchmark results."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="all2all.csv",
        help='Input text file, or "-" to read from stdin.',
    )
    parser.add_argument(
        "--output-dir",
        default="nccl_figures",
        help="Directory for generated figures and parsed CSV.",
    )
    parser.add_argument(
        "--max-small-kb",
        type=int,
        default=65536,
        help="Maximum message size shown in the small-message zoom plot.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = parse_results(args.input)

    parsed_csv = output_dir / "parsed_nccl_alltoall.csv"
    latency_png = output_dir / "latency_vs_size.png"
    bandwidth_png = output_dir / "bandwidth_vs_size.png"
    small_png = output_dir / "small_message_latency.png"
    peer_png = output_dir / "latency_vs_peer_payload.png"

    df.to_csv(parsed_csv, index=False)

    plot_latency(df, latency_png)
    plot_bandwidth(df, bandwidth_png)
    plot_small_message_zoom(df, small_png, args.max_small_kb)
    plot_peer_size(df, peer_png)

    print_summary(df)

    print("\nGenerated:")
    for path in [
        parsed_csv,
        latency_png,
        bandwidth_png,
        small_png,
        peer_png,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()