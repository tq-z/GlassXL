"""Plot average decode-only time across the non-1M hardware configurations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_ROOT = Path("/mnt/bravo/tiz014/PanAgretti/results/Dell/kimiK2")
OUTPUT_FIGURE = RESULTS_ROOT / "decode_time_comparison.png"
OUTPUT_CSV = RESULTS_ROOT / "decode_time_comparison.csv"

HARDWARE_RESULTS = {
    "GPU": RESULTS_ROOT / "GPU",
    "GlassML": RESULTS_ROOT / "GlassML",
    "GlassML-NMP": RESULTS_ROOT / "GlassML-NMP",
}
RATES_PER_SECOND = (0.1, 0.5, 1, 2, 4, 8, 16, 32, 64)
DECODE_PHASES = ("attn_s", "ffn_s", "ffn_all_to_all_s")


def load_decode_time(result_dir: Path) -> float:
    """Return average per-session decode time, excluding prefill and migration."""
    stats_path = result_dir / "tmp_session.json"
    with stats_path.open() as stats_file:
        stats = json.load(stats_file)

    average_phase_time = stats["average_phase_time_s"]
    return sum(average_phase_time[phase] for phase in DECODE_PHASES)


def main() -> None:
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for hardware, hardware_dir in HARDWARE_RESULTS.items():
        for rate in RATES_PER_SECOND:
            rate_dir = hardware_dir / f"rate_{rate:g}"
            if not (rate_dir / "tmp_session.json").exists():
                raise FileNotFoundError(
                    f"Missing session statistics for {hardware} at rate {rate}: "
                    f"{rate_dir / 'tmp_session.json'}"
                )
            rows.append(
                {
                    "hardware": hardware,
                    "rate_per_second": rate,
                    "decode_time_s": load_decode_time(rate_dir),
                }
            )

    with OUTPUT_CSV.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=("hardware", "rate_per_second", "decode_time_s")
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"GPU": "tab:blue", "GlassML": "tab:orange", "GlassML-NMP": "tab:green"}
    decode_times_ms: dict[str, dict[float, float]] = {}
    for hardware in HARDWARE_RESULTS:
        hardware_rows = [row for row in rows if row["hardware"] == hardware]
        decode_times_ms[hardware] = {
            float(row["rate_per_second"]): float(row["decode_time_s"]) * 1000
            for row in hardware_rows
        }
    for hardware in HARDWARE_RESULTS:
        hardware_rows = [row for row in rows if row["hardware"] == hardware]
        ax.plot(
            [row["rate_per_second"] for row in hardware_rows],
            [row["decode_time_s"] * 1000 for row in hardware_rows],
            marker="o",
            linewidth=2,
            label=hardware,
            color=colors[hardware],
        )

    # Show speedups only at the lowest arrival rate, relative to GPU.
    low_rate = RATES_PER_SECOND[0]
    gpu_decode_times = decode_times_ms["GPU"]
    for hardware in HARDWARE_RESULTS:
        speedup = gpu_decode_times[low_rate] / decode_times_ms[hardware][low_rate]
        ax.annotate(
            f"{speedup:.2f}x",
            xy=(low_rate, decode_times_ms[hardware][low_rate]),
            xytext=(8, 8),
            textcoords="offset points",
            color=colors[hardware],
            fontsize=9,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": colors[hardware], "alpha": 0.5},
        )

    # At GPU's rate=4 point, find where GlassML-NMP reaches the same decode time.
    reference_rate = 4
    reference_time = gpu_decode_times[reference_rate]
    nmp_points = sorted(decode_times_ms["GlassML-NMP"].items())
    for (left_rate, left_time), (right_rate, right_time) in zip(nmp_points, nmp_points[1:]):
        if left_time <= reference_time <= right_time:
            fraction = (reference_time - left_time) / (right_time - left_time)
            intersection_rate = math.exp(
                math.log(left_rate)
                + fraction * (math.log(right_rate) - math.log(left_rate))
            )
            break
    else:
        raise ValueError("GPU rate=4 decode time is outside the GlassML-NMP curve")

    ax.hlines(
        reference_time,
        reference_rate,
        intersection_rate,
        colors="0.35",
        linestyles="--",
        linewidth=1.5,
    )
    ax.scatter(
        [intersection_rate],
        [reference_time],
        color=colors["GlassML-NMP"],
        edgecolors="black",
        zorder=5,
    )
    ax.annotate(
        f"{intersection_rate / reference_rate:.2f}x higher rate",
        xy=(intersection_rate, reference_time),
        xytext=(8, 12),
        textcoords="offset points",
        color=colors["GlassML-NMP"],
        fontsize=9,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": colors["GlassML-NMP"], "alpha": 0.5},
    )

    ax.set_xscale("log")
    ax.set_xlabel("Arrival rate (sessions/s)")
    ax.set_ylabel("Average decode time per session (ms)")
    ax.set_title("Decode time vs. arrival rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=180)
    plt.close(fig)

    print(f"Wrote {OUTPUT_FIGURE}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
