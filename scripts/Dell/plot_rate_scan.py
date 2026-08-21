"""Visualize the Kimi K2 arrival-rate scan results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Change this constant to read/write a different scan.
RESULTS_ROOT = Path("/mnt/bravo/tiz014/PanAgretti/results/Dell/kimiK2/GlassML-NMP1M")

RATES_PER_SECOND = (0.1, 0.5, 1, 2, 4, 8, 16, 32,64)
PHASES = (
    ("prefill_s", "Prefill"),
    ("prefill_queue_wait_s", "Prefill queue wait"),
    ("kv_ingress_s", "KV ingress"),
    ("tool_wait_s", "Tool wait"),
    ("attn_s", "Attention"),
    ("ffn_all_to_all_s", "FFN all-to-all"),
    ("ffn_s", "FFN"),
)
PHASE_COLORS = (
    "#4C78A8",
    "#9ECAE1",
    "#72B7B2",
    "#F2CF5B",
    "#E45756",
    "#B279A2",
    "#6A3D9A",
)


def _average_active_sessions(session_stats: dict[str, Any], horizon_s: float) -> float:
    if horizon_s <= 0.0:
        return 0.0
    active_seconds = 0.0
    for session in session_stats["sessions"]:
        arrival = max(0.0, float(session["arrival_time_s"]))
        finish_value = session["finish_time_s"]
        finish = horizon_s if finish_value is None else min(float(finish_value), horizon_s)
        active_seconds += max(0.0, finish - arrival)
    return active_seconds / horizon_s


def _load_rate(rate: float) -> dict[str, Any]:
    rate_directory = RESULTS_ROOT / f"rate_{rate:g}"
    result_path = rate_directory / "result.json"
    session_path = rate_directory / "tmp_session.json"

    if not result_path.exists() or not session_path.exists():
        raise FileNotFoundError(
            f"missing result files for rate {rate:g}: {rate_directory}"
        )

    with result_path.open(encoding="utf-8") as stream:
        result_text = stream.read()
    result, _ = json.JSONDecoder().raw_decode(result_text.lstrip())
    with session_path.open(encoding="utf-8") as stream:
        session_stats = json.load(stream)

    return {
        "rate_per_second": rate,
        "mean_session_latency_s": result["mean_session_latency_ms"] / 1000.0,
        "p95_session_latency_s": result["p95_session_latency_ms"] / 1000.0,
        "system_tokens_per_second": result["decode_token_throughput_per_s"],
        "emitted_tokens": result["emitted_tokens"],
        "simulation_time_s": result["simulation_time_s"],
        "average_active_sessions": _average_active_sessions(
            session_stats, result["simulation_time_s"]
        ),
        "phase_times_s": {
            **session_stats["average_phase_time_s"],
            "prefill_queue_wait_s": session_stats[
                "average_prefill_queue_wait_s"
            ],
            "tool_wait_s": session_stats["average_tool_wait_s"],
        },
    }


def _load_scan() -> list[dict[str, Any]]:
    return [_load_rate(rate) for rate in RATES_PER_SECOND]


def _write_summary_csv(records: list[dict[str, Any]]) -> Path:
    output_path = RESULTS_ROOT / "scan_summary.csv"
    fieldnames = [
        "rate_per_second",
        "mean_session_latency_s",
        "p95_session_latency_s",
        "system_tokens_per_second",
        "emitted_tokens",
        "simulation_time_s",
        "average_active_sessions",
        *(key for key, _ in PHASES),
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                key: record[key]
                for key in fieldnames
                if key in record
            }
            row.update(record["phase_times_s"])
            writer.writerow(row)
    return output_path


def _plot_latency(records: list[dict[str, Any]]) -> Path:
    rates = [item["rate_per_second"] for item in records]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        constrained_layout=True,
    )

    breakdown_axis, overall_axis = axes
    bottom = [0.0] * len(records)
    for (phase_key, phase_label), color in zip(PHASES, PHASE_COLORS):
        values = [item["phase_times_s"][phase_key] for item in records]
        breakdown_axis.bar(
            rates,
            values,
            bottom=bottom,
            width=0.08 * max(rates),
            label=phase_label,
            color=color,
        )
        bottom = [left + value for left, value in zip(bottom, values)]

    breakdown_axis.plot(
        rates,
        bottom,
        color="black",
        marker="o",
        linewidth=1.5,
        label="Component total",
    )
    breakdown_axis.set_xscale("log")
    breakdown_axis.set_xticks(rates)
    breakdown_axis.set_xticklabels([f"{rate:g}" for rate in rates])
    breakdown_axis.set_xlabel("Arrival rate (sessions/s)")
    breakdown_axis.set_ylabel("Average time per completed session (s)")
    breakdown_axis.set_title("Latency breakdown")
    breakdown_axis.grid(axis="y", alpha=0.3)

    active_axis = breakdown_axis.twinx()
    active_sessions = [item["average_active_sessions"] for item in records]
    active_axis.plot(
        rates,
        active_sessions,
        color="#1F77B4",
        marker="D",
        linestyle="--",
        linewidth=2,
        label="Average active sessions",
    )
    active_axis.set_ylabel("Average active sessions", color="#1F77B4")
    active_axis.tick_params(axis="y", labelcolor="#1F77B4")
    handles, labels = breakdown_axis.get_legend_handles_labels()
    active_handles, active_labels = active_axis.get_legend_handles_labels()
    breakdown_axis.legend(
        handles + active_handles,
        labels + active_labels,
        loc="upper left",
    )

    mean_latency_ms = [
        item["mean_session_latency_s"] * 1000.0 for item in records
    ]
    p95_latency_ms = [
        item["p95_session_latency_s"] * 1000.0 for item in records
    ]
    overall_axis.plot(
        rates,
        mean_latency_ms,
        marker="o",
        linewidth=2,
        label="Mean session latency",
    )
    overall_axis.plot(
        rates,
        p95_latency_ms,
        marker="s",
        linewidth=2,
        label="P95 session latency",
    )
    overall_axis.set_xscale("log")
    overall_axis.set_xticks(rates)
    overall_axis.set_xticklabels([f"{rate:g}" for rate in rates])
    overall_axis.set_xlabel("Arrival rate (sessions/s)")
    overall_axis.set_ylabel("Session latency (ms)")
    overall_axis.set_title("Overall latency")
    overall_axis.grid(alpha=0.3)
    overall_axis.legend()

    output_path = RESULTS_ROOT / "latency_breakdown_and_overall.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _plot_system_tokens(records: list[dict[str, Any]]) -> Path:
    rates = [item["rate_per_second"] for item in records]
    tokens_per_second = [
        item["system_tokens_per_second"] for item in records
    ]

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        rates,
        tokens_per_second,
        marker="o",
        linewidth=2,
        color="#2F6B3F",
    )
    for rate, tokens in zip(rates, tokens_per_second):
        axis.annotate(
            f"{tokens:,.0f}",
            (rate, tokens),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_xticks(rates)
    axis.set_xticklabels([f"{rate:g}" for rate in rates])
    axis.set_xlabel("Arrival rate (sessions/s)")
    axis.set_ylabel("System output (tokens/s)")
    axis.set_title("System token throughput vs. arrival rate")
    axis.grid(alpha=0.3)

    output_path = RESULTS_ROOT / "system_tokens_per_second.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    records = _load_scan()
    summary_path = _write_summary_csv(records)
    latency_path = _plot_latency(records)
    throughput_path = _plot_system_tokens(records)
    print(f"wrote {latency_path}")
    print(f"wrote {throughput_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

