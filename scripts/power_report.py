#!/usr/bin/env python3
"""Figures and markdown tables for docs/power_takeaways.md from results/power runs.

  python scripts/power_report.py --out docs/figures/power_takeaways > /tmp/tables.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "power"


def load_grid(name: str) -> dict:
    return json.loads((RESULTS / name / "grid.json").read_text())


def grid_matrix(grid: dict, key, batches, contexts):
    values = np.full((len(contexts), len(batches)), np.nan)
    for cell in grid["cells"]:
        if not cell["feasible"]:
            continue
        i, j = contexts.index(cell["context"]), batches.index(cell["batch"])
        values[i, j] = key(cell)
    return values


def ctx_label(context: int) -> str:
    return f"{context // 1024}K" if context < 1024 * 1024 else f"{context // (1024 * 1024)}M"


def heatmaps(grid: dict, out: Path, tag: str) -> None:
    batches = sorted({cell["batch"] for cell in grid["cells"]})
    contexts = sorted({cell["context"] for cell in grid["cells"]})
    panels = [
        ("TPOT (ms)", lambda c: c["tpot_ms"], "viridis"),
        ("Decode throughput (tok/s)", lambda c: c["decode_tokens_per_s"], "magma"),
        ("NMP node mean power (W)", lambda c: c["decode_window_power"]["nmp"]["chip_mean_w"], "inferno"),
        ("Panel energy per token (J)", lambda c: c["decode_window_energy"]["panel_j_per_token"], "cividis"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    for ax, (title, key, cmap) in zip(axes, panels):
        values = grid_matrix(grid, key, batches, contexts)
        shown = np.log10(values) if "throughput" in title or "per token" in title else values
        image = ax.imshow(shown, origin="lower", aspect="auto", cmap=cmap)
        for i in range(len(contexts)):
            for j in range(len(batches)):
                if np.isnan(values[i, j]):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color="#dddddd", hatch="///"))
                    continue
                value = values[i, j]
                text = f"{value:.0f}" if value >= 100 else (f"{value:.1f}" if value >= 1 else f"{value:.2f}")
                ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color="white")
        ax.set_xticks(range(len(batches)), batches)
        ax.set_yticks(range(len(contexts)), [ctx_label(c) for c in contexts])
        ax.set_xlabel("batch (concurrent decoding sessions)")
        ax.set_ylabel("context length")
        ax.set_title(title + (" [log color]" if shown is not values else ""), fontsize=10)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(f"{grid['describe']} - steady decode window (grey = KV does not fit)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"grid_heatmaps_{tag}.png", dpi=140)
    plt.close(fig)


def step_breakdown(grid: dict, out: Path, tag: str) -> None:
    batches = sorted({cell["batch"] for cell in grid["cells"]})
    contexts = [c for c in (4096, 32768, 131072, 524288) if any(cell["context"] == c for cell in grid["cells"])]
    fig, axes = plt.subplots(1, len(contexts), figsize=(4.4 * len(contexts), 4), sharey=True)
    for ax, context in zip(np.atleast_1d(axes), contexts):
        cells = [cell for cell in grid["cells"] if cell["context"] == context and cell["feasible"]]
        cells.sort(key=lambda cell: cell["batch"])
        xs = [str(cell["batch"]) for cell in cells]
        attention = np.array([cell["attention_barrier_median_ms"] for cell in cells])
        a2a = np.array([cell["all_to_all_median_ms"] for cell in cells])
        ffn = np.array([cell["ffn_median_ms"] for cell in cells])
        ax.bar(xs, attention, label="attention barrier")
        ax.bar(xs, a2a, bottom=attention, label="all-to-all (2 x 94 x 20 us launch)")
        ax.bar(xs, ffn, bottom=attention + a2a, label="FFN (real-routing straggler)")
        ax.set_title(f"context {ctx_label(context)}")
        ax.set_xlabel("batch")
    np.atleast_1d(axes)[0].set_ylabel("median step time (ms)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"Decode step composition - {grid['describe']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"grid_step_breakdown_{tag}.png", dpi=140)
    plt.close(fig)


def die_power_lines(grid: dict, out: Path, tag: str) -> None:
    batches = sorted({cell["batch"] for cell in grid["cells"]})
    contexts = sorted({cell["context"] for cell in grid["cells"]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for context in contexts:
        cells = sorted(
            (cell for cell in grid["cells"] if cell["context"] == context and cell["feasible"]),
            key=lambda cell: cell["batch"],
        )
        if not cells:
            continue
        xs = [cell["batch"] for cell in cells]
        axes[0].plot(xs, [cell["decode_window_power"]["nmp"]["logic_mean_w"] for cell in cells], marker="o", label=ctx_label(context))
        axes[1].plot(xs, [cell["decode_window_power"]["nmp"]["dram_die_mean_w"] for cell in cells], marker="o", label=ctx_label(context))
    for ax, title in zip(axes, ("NMP logic die, mean over nodes (W)", "NMP DRAM die, mean over dies (W)")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(title="context", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"grid_die_power_{tag}.png", dpi=140)
    plt.close(fig)


def grid_table(grid: dict) -> str:
    batches = sorted({cell["batch"] for cell in grid["cells"]})
    contexts = sorted({cell["context"] for cell in grid["cells"]})
    lookup = {(cell["context"], cell["batch"]): cell for cell in grid["cells"]}
    lines = []
    for title, key, fmt in (
        ("TPOT (ms)", lambda c: c["tpot_ms"], "{:.1f}"),
        ("Decode throughput (tok/s)", lambda c: c["decode_tokens_per_s"], "{:.0f}"),
        ("NMP node mean power in the decode window (W)", lambda c: c["decode_window_power"]["nmp"]["chip_mean_w"], "{:.0f}"),
        ("NMP logic die, mean (W)", lambda c: c["decode_window_power"]["nmp"]["logic_mean_w"], "{:.0f}"),
        ("Panel energy per output token (J)", lambda c: c["decode_window_energy"]["panel_j_per_token"], "{:.2f}"),
        ("Mean TTFT (s)", lambda c: c["mean_ttft_ms"] / 1e3, "{:.2f}"),
    ):
        lines.append(f"\n**{title}** ({grid['describe']})\n")
        lines.append("| context \\ batch | " + " | ".join(str(b) for b in batches) + " |")
        lines.append("|---|" + "---|" * len(batches))
        for context in contexts:
            row = []
            for batch in batches:
                cell = lookup[(context, batch)]
                if not cell["feasible"]:
                    row.append("—")
                else:
                    try:
                        row.append(fmt.format(key(cell)))
                    except (TypeError, KeyError):
                        row.append("n/a")
            lines.append(f"| {ctx_label(context)} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def traces(out: Path, suffix: str = "") -> str:
    notes = []
    fixed = f"h200_perf_sim_{suffix}_traces" if suffix else "h200_perf_sim_traces"
    legacy = f"h200_perf_sim_{suffix}_traces_legacy" if suffix else "h200_perf_sim_traces_legacy"
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    for name, ax, label in (
        (legacy, axes[0], "legacy transfer booking (simulator default)"),
        (fixed, axes[1], "transfers booked at prefill completion"),
    ):
        path = RESULTS / name / "power.npz"
        if not path.exists():
            continue
        data = np.load(path)
        power = data["power_w"].astype(float)
        edges = data["t_edges_s"]
        t = edges[:-1]
        chips = data["die_chip"]
        kinds = data["die_kind"]
        nmp_chips = sorted(set(chips[kinds == "nmp"]))
        gpu_chips = sorted(set(chips[kinds == "gpu"]))
        nmp = np.array([power[:, chips == c].sum(axis=1) for c in nmp_chips])
        gpu = np.array([power[:, chips == c].sum(axis=1) for c in gpu_chips])
        hottest = int(np.argmax(nmp.mean(axis=1)))
        coolest = int(np.argmin(nmp.mean(axis=1)))
        ax.plot(t, gpu[0], lw=0.6, label=f"{gpu_chips[0]} (GPU chip)")
        ax.plot(t, nmp[hottest], lw=0.6, label=f"{nmp_chips[hottest]} (hottest NMP node)")
        ax.plot(t, nmp[coolest], lw=0.6, label=f"{nmp_chips[coolest]} (coolest NMP node)")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("W (10 ms bins)")
        ax.legend(fontsize=8, loc="upper right")
        notes.append(
            f"{name}: panel mean {power.sum(axis=1).mean():.0f} W, NMP node mean {nmp.mean():.1f} W "
            f"(hottest {nmp_chips[hottest]} {nmp[hottest].mean():.1f} W, coolest {nmp_chips[coolest]} {nmp[coolest].mean():.1f} W), "
            f"GPU chip mean {gpu.mean():.0f} W"
        )
        if name == fixed:
            # Zoom on the 3 s around the hottest node's largest 10 ms change.
            centre = t[int(np.argmax(np.abs(np.diff(nmp[hottest]))))]
            window = (t >= centre - 1.5) & (t < centre + 1.5)
            names = data["die_names"]
            for die in (f"{nmp_chips[hottest]}.logic", f"{nmp_chips[hottest]}.dram0", f"{nmp_chips[hottest]}.pic"):
                column = int(np.flatnonzero(names == die)[0])
                axes[2].plot(t[window], power[window, column], lw=0.8, label=die)
            axes[2].plot(t[window], gpu[0][window], lw=0.8, label=f"{gpu_chips[0]} (chip)")
            axes[2].set_title("3 s zoom (fixed booking): per-die power at 10 ms resolution", fontsize=10)
            axes[2].set_xlabel("time (s)")
            axes[2].set_ylabel("W")
            axes[2].legend(fontsize=8)
            diffs = np.abs(np.diff(nmp, axis=1))
            gdiffs = np.abs(np.diff(gpu, axis=1))
            notes.append(
                f"fixed run: NMP node 10 ms |dP| p99 {np.percentile(diffs, 99):.1f} W max {diffs.max():.1f} W; "
                f"GPU chip |dP| p99 {np.percentile(gdiffs, 99):.1f} W max {gdiffs.max():.1f} W; "
                f"NMP node peak {nmp.max():.1f} W; GPU chip peak {gpu.max():.1f} W"
            )
            node_means = nmp.mean(axis=1)
            fig2, ax2 = plt.subplots(figsize=(12, 3.5))
            ax2.bar(range(len(nmp_chips)), node_means - node_means.min(), bottom=node_means.min(), color="#c44e52")
            ax2.set_xticks(range(len(nmp_chips)), [c.replace("nmp0", "") for c in nmp_chips], fontsize=6, rotation=90)
            ax2.set_ylabel("mean W")
            ax2.set_title("Mean power per NMP node (fixed booking, 120 s): low-id nodes run hotter (placement tie-break)", fontsize=10)
            fig2.tight_layout()
            fig2.savefig(out / f"nmp_node_mean_power{'_' + suffix if suffix else ''}.png", dpi=140)
            plt.close(fig2)
    fig.tight_layout()
    fig.savefig(out / f"trace_timeseries{'_' + suffix if suffix else ''}.png", dpi=140)
    plt.close(fig)
    return "\n".join(notes)


def splits(out: Path, suffix: str = "", label: str = "Qwen3-235B") -> str:
    base_dir = f"splits_{suffix}" if suffix else "splits"
    path = RESULTS / base_dir / "splits.json"
    if not path.exists():
        return "(splits missing)"
    data = json.loads(path.read_text())
    runs = data["runs"]
    extra = RESULTS / f"{base_dir}_extra" / "splits.json"
    if extra.exists():
        seen = {run["name"] for run in runs}
        runs += [run for run in json.loads(extra.read_text())["runs"] if run["name"] not in seen]
    names = [run["name"] for run in runs]
    # Common set: sessions every config completed with all offered turns (capacity
    # fitting truncates the heaviest sessions on small-KV nodes, which would flatter them).
    common = None
    for run in runs:
        run["stalled"] = run["completed_sessions"] < run["sessions"]
        if run["stalled"]:
            continue  # a run that ended with sessions stuck would shrink the set arbitrarily
        full = {
            sid for sid, (latency, _, done, offered) in run.get("per_session", {}).items()
            if latency is not None and done == offered
        }
        common = full if common is None else common & full
    common = common or set()
    for run in runs:
        rows = [run["per_session"][sid] for sid in common if run["per_session"][sid][0] is not None]
        run["common_serving_s"] = float(np.mean([latency - wait for latency, wait, _, _ in rows])) if rows else float("nan")
        run["common_latency_s"] = float(np.mean([latency for latency, _, _, _ in rows])) if rows else float("nan")
    base = next(run for run in runs if run["name"] == data["baseline"])
    for run in runs:
        run["common_speedup"] = base["common_serving_s"] / run["common_serving_s"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    metrics = (
        (f"mean serving time, common {len(common)} sessions (s)", "common_serving_s"),
        ("mean TPOT (ms)", "mean_tpot_ms"),
        ("mean TTFT (s)", None),
        ("decode throughput while busy (tok/s)", "busy_decode_tokens_per_s"),
        ("energy per output token (J)", "j_per_output_token"),
        ("panel KV capacity (M tokens)", None),
    )
    for ax, (title, key) in zip(axes.ravel(), metrics):
        if title.startswith("mean TTFT"):
            values = [run["mean_ttft_ms"] / 1e3 for run in runs]
        elif title.startswith("panel KV"):
            values = [run["panel_kv_capacity_tokens"] / 1e6 for run in runs]
        else:
            values = [run[key] or 0.0 for run in runs]
        colors = ["#4c72b0" if name.startswith("1to6_16x6") else "#999999" for name in names]
        ax.barh(names, values, color=colors)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"Codex {label} workload (300 sessions, 4/s, run to completion) - blue = 1:6 default", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"splits{'_' + suffix if suffix else ''}.png", dpi=140)
    plt.close(fig)
    lines = [
        f"Common set: {len(common)} sessions completed in full by every config.\n",
        "| config | GPUs : stacks (nodes x stacks, TP) | rated kW | panel KV (M tok) | dropped / cut turns | serving s, common set (speedup) | serving s, all completed | TPOT ms | TTFT s mean (p95) | prefill queue s | busy decode tok/s | panel mean kW | J/token |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        panel = run["panel"]
        fit = run["session_fit"]
        lines.append(
            f"| {run['name']}{' (stalled)' if run['stalled'] else ''} | {panel['gpus']} : {panel['nodes'] * panel['stacks']} ({panel['nodes']}x{panel['stacks']}, TP{panel['tp']}) "
            f"| {run['panel_rated_w'] / 1e3:.1f} | {run['panel_kv_capacity_tokens'] / 1e6:.1f} "
            f"| {fit['sessions_dropped']} / {fit['turns_cut']} "
            f"| {run['common_serving_s']:.1f} ({run['common_speedup']:.2f}x) | {run['mean_serving_s']:.1f} | {run['mean_tpot_ms']:.1f} "
            f"| {run['mean_ttft_ms'] / 1e3:.1f} ({run['p95_ttft_ms'] / 1e3:.0f}) | {run['mean_prefill_queue_s']:.1f} "
            f"| {run['busy_decode_tokens_per_s']:.0f} | {run['panel_mean_w'] / 1e3:.2f} | {run['j_per_output_token']:.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suffix", default="", help="results-name infix, e.g. glm45 -> grid_glm45_*, splits_glm45")
    parser.add_argument("--label", default="Qwen3-235B", help="model label for figure titles")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    infix = f"{args.suffix}_" if args.suffix else ""
    for name in (f"grid_{infix}1to6_16x6", f"grid_{infix}1to6_12x8", f"grid_{infix}1to28_8x56x4"):
        if (RESULTS / name / "grid.json").exists():
            grid = load_grid(name)
            tag = name.replace("grid_", "")
            heatmaps(grid, args.out, tag)
            step_breakdown(grid, args.out, tag)
            die_power_lines(grid, args.out, tag)
            print(grid_table(grid))
    print("\n## traces\n" + traces(args.out, args.suffix))
    print("\n## splits\n" + splits(args.out, args.suffix, args.label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
