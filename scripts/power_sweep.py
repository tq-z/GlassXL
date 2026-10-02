#!/usr/bin/env python3
"""Power/performance sweeps on H200 prefill + perf_sim-default NMP decode panels.

Two studies, both built on power_model.simulate (the same pipeline as
scripts/power_trace.py):

  grid    batch x context: B sessions, each already at context C, prefill P new
          tokens and decode O tokens together. Reports the steady decode window
          (all B sessions active): step time, throughput, per-die power, energy
          per token; and the prefill phase: TTFT and GPU power. Cells whose KV
          cannot be resident at once are reported as infeasible.
  splits  GPU : NMP-stack ratios and NMP node sizes on the TraceLab Codex
          workload (Qwen3-235B), run to completion: TTFT, TPOT, throughput,
          per-session serving time (latency minus tool wait), power, energy/token.

A panel is P H200 chips (tensor-parallel groups of --tp) plus N NMP nodes of S
perf_sim-default stacks. Node rates are S x the per-stack effective rates that
scripts/power_crosscheck.py --geometry perf_sim_default --measure-rates measured.

  python scripts/power_sweep.py grid --gpus 16 --nodes 16 --stacks 6 --out results/power/grid_1to6 --jobs 8
  python scripts/power_sweep.py splits --out results/power/splits --jobs 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentic_sim.config import load_config  # noqa: E402
from agentic_sim.workload import SessionSpec, TurnSpec  # noqa: E402
from power_model import load_power_config  # noqa: E402
from power_model.collector import PowerCollector  # noqa: E402
from power_model.simulate import simulate  # noqa: E402
from roofline_model import PrecisionSpec, resolve_model  # noqa: E402
from roofline_model.activity import PrefillContextCoefficients, moe_layer_count  # noqa: E402
from scripts.generate_config import build_documents, build_parser, write_documents  # noqa: E402

WORKLOAD_YAML = REPOSITORY_ROOT / "configs" / "glassml" / "workload" / "workload.yaml"
SESSIONS_JSON = REPOSITORY_ROOT / "configs" / "glassml" / "workload" / "sessions.json"
POWER_YAML = REPOSITORY_ROOT / "configs" / "power" / "h200_perf_sim.yaml"
ROUTING_DIR = REPOSITORY_ROOT.parent / "moe_routing"
DEFAULT_MODEL = "qwen3-235b"

# One unpatched perf_sim stack (results/power/perf_sim_crosscheck_default.json).
STACK_ATTN_FLOPS = 2.0 * 4.5299486e13  # attention at 128K context, GQA 16: 69% of peak MACs
STACK_BYTES = 2.1816905e13  # batch-1 GEMV streaming, harmonic mean over the 8 tRCD tiers
# FFN on an S-stack node in perf_sim's own multi-stack convention (expert tensor
# parallelism = S, input all-gather included): 1 expert x 1024 tokens for FLOP/s,
# batch 1 for bytes/s, the latter scaled from the tier-0 placement by the stack's
# tier-average factor 21.82 / 26.10. Perfect S-fold scaling would be 2-3x faster.
TIER_FACTOR = 2.1816905e13 / 2.6104744e13
NODE_FFN = {
    stacks: (2.0 * macs, TIER_FACTOR * streaming)
    for stacks, macs, streaming in (
        (1, 3.8675351e13, 2.6104744e13),
        (2, 5.9926600e13, 4.7454698e13),
        (4, 7.4771634e13, 7.9482922e13),
        (6, 9.4568970e13, 1.0223114e14),
        (8, 9.5730630e13, 1.0223114e14),
    )
}
STACK_PEAK_MACS = 65536e9  # 16 ch x 16 banks x 16 x 16 MACs at 1 GHz
STACK_PEAK_BYTES = 2.42e13  # 256 banks x 4 KiB / tier-averaged tRC
STACK_CAPACITY_BYTES = 32 * 2**30
# H200 per chip: 980 TFLOPS and 4.8 TB/s, derated 0.65 / 0.75 (scripts/generate_config_nmp.py).
H200_FLOPS = 0.65 * 9.8e14
H200_BYTES = 0.75 * 4.8e12
H200_MEMORY_BYTES = 141e9
# Placeholder leakage: 10% of the perf_sim-default logic die at full load (~203 W for
# four stacks incl. 17.7 W I/O drivers), scaled with the node's stack count.
LOGIC_STATIC_PER_STACK_W = 20.3 / 4


@dataclass(frozen=True)
class Panel:
    name: str
    gpus: int
    tp: int
    nodes: int
    stacks: int
    model: str = DEFAULT_MODEL

    @property
    def total_stacks(self) -> int:
        return self.nodes * self.stacks

    def describe(self) -> str:
        return (
            f"{self.gpus} H200 ({self.gpus // self.tp} x TP{self.tp}) + {self.nodes} NMP nodes x "
            f"{self.stacks} stacks ({self.total_stacks} stacks, 1:{self.total_stacks / self.gpus:g}), "
            f"{resolve_model(self.model).display_name}"
        )


def node_weight_bytes(model_name: str, nodes: int, precision: PrecisionSpec = PrecisionSpec()) -> float:
    """Weights resident on one decode node.

    Attention projections are replicated on every node (DP attention); dense,
    shared and routed expert weights are sharded over the EP group, as the
    roofline estimator and the generator's FFN curves assume.
    """

    spec = resolve_model(model_name)
    wb = precision.weight_bytes
    attention = sum(group.count * group.attention.projection_params for group in spec.layer_groups) * wb
    nonrouted = sum(
        group.count
        * (spec.dense_active_params if group.ffn_kind == "dense"
           else spec.shared_active_params + spec.moe_common_active_params)
        for group in spec.layer_groups
    ) * wb
    routed = PrefillContextCoefficients.from_model(spec, precision).routed_weight_bytes
    return attention + (nonrouted + routed) / nodes


def required_tp(model_name: str, precision: PrecisionSpec = PrecisionSpec()) -> int:
    """Smallest power-of-two prefill group whose HBM holds the model's weights."""

    coefficients = PrefillContextCoefficients.from_model(resolve_model(model_name), precision)
    chips = math.ceil((coefficients.fixed_weight_bytes + coefficients.routed_weight_bytes) / H200_MEMORY_BYTES)
    return 1 << max(0, (chips - 1).bit_length())


def kv_capacity_bytes(panel: Panel) -> float:
    """Node DRAM minus the weights resident on the node."""

    return panel.stacks * STACK_CAPACITY_BYTES - node_weight_bytes(panel.model, panel.nodes)


def generator_document(panel: Panel, output_dir: Path) -> dict[str, Any]:
    if panel.gpus % panel.tp:
        raise ValueError(f"{panel.name}: {panel.gpus} GPUs are not a multiple of TP{panel.tp}")
    capacity = kv_capacity_bytes(panel)
    if capacity <= 0:
        raise ValueError(f"{panel.name}: node DRAM does not hold its weights")
    if panel.stacks not in NODE_FFN:
        raise ValueError(f"{panel.name}: no perf_sim FFN measurement for {panel.stacks}-stack nodes")
    ffn_flops, ffn_bytes = NODE_FFN[panel.stacks]
    return {
        "model": panel.model,
        "prefill_device_flops_per_second": H200_FLOPS,
        "prefill_device_memory_bandwidth_bytes_per_second": H200_BYTES,
        "prefill_devices": panel.gpus // panel.tp,
        "prefill_tp": panel.tp,
        "prefill_model": "roofline_context",
        "decode_attn_device_flops_per_second": panel.stacks * STACK_ATTN_FLOPS,
        "decode_attn_device_memory_bandwidth_bytes_per_second": panel.stacks * STACK_BYTES,
        "decode_ffn_device_flops_per_second": ffn_flops,
        "decode_ffn_device_memory_bandwidth_bytes_per_second": ffn_bytes,
        "dp": panel.nodes,
        "ep": panel.nodes,
        "colocate_attn_ffn": True,
        "attn_proj_reference_batch": 1,
        "prefill_tokens": 181,
        "decode_context_tokens": 131072,
        "attn_kv_capacity_bytes": capacity,
        # Every node keeps a PanAgretti PIC complement (63 x 2 Tb/s links per direction).
        "ffn_all_to_all_bandwidth_bytes_per_second": 1.64e13,
        "ffn_all_to_all_start_latency_us": 20,
        "ffn_all_to_all_min_payload_bytes": 0,
        "kv_p2p_bandwidth_bytes_per_second": 1.64e13,
        "kv_p2p_start_latency_us": 20,
        "kv_pool_bandwidth_bytes_per_second": 6.4e12,
        "kv_pool_start_latency_us": 20,
        "output_dir": str(output_dir),
        "weight_bytes": 2,
        "activation_bytes": 2,
        "kv_bytes": 2,
    }


def build_panel(panel: Panel, directory: Path, routing_dir: Path | None) -> Path:
    """Write hardware.yaml, mapping.yaml, roofline_report.yaml (skew-aware curves) for a panel."""

    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "generator.yaml"
    source.write_text(yaml.safe_dump(generator_document(panel, directory), sort_keys=False), encoding="utf-8")
    argv = ["--input-yaml", str(source), "--output-dir", str(directory), "--force"]
    if routing_dir is not None:
        argv += ["--routing-traces", str(routing_dir)]
    write_documents(directory, build_documents(build_parser().parse_args(argv)), force=True)
    return directory


def panel_power(panel: Panel, power_yaml: Path, dt_s: float):
    """The h200 / perf_sim_default_nmp coefficients with the node's stack count."""

    power = load_power_config(power_yaml)
    nmp = dict(power.nmp)
    nmp.update(
        dram_dies=float(panel.stacks),
        macs_per_second=panel.stacks * STACK_PEAK_MACS,
        dram_bytes_per_second=panel.stacks * STACK_PEAK_BYTES,
    )
    if "nmp.p_logic_static_w" in power.placeholders:
        nmp["p_logic_static_w"] = LOGIC_STATIC_PER_STACK_W * panel.stacks
    return replace(power, model=panel.model, nmp=nmp, dt_s=dt_s)


def load_panel(directory: Path):
    config = load_config(directory / "hardware.yaml", WORKLOAD_YAML, directory / "mapping.yaml")
    with (directory / "roofline_report.yaml").open(encoding="utf-8") as stream:
        report = yaml.safe_load(stream)
    return config, report


class SlimCollector(PowerCollector):
    """Drops the per-step expert-load, FFN-device and wait tuples from the step records.

    Nothing in the summary reads them, but at ~15 KB per step they take gigabytes
    on runs to completion (a million-plus decode steps).
    """

    def record_decode(self, result) -> None:
        super().record_decode(result)
        self.decode_steps[-1] = replace(
            self.decode_steps[-1], expert_loads=(), ffn_devices=(), attention_device_wait_s=()
        )


def bound_reservation_history(keep: int = 256) -> None:
    """Keep only the last ``keep`` reservations per queue in this worker process.

    Reservation histories feed only the snapshot's resource-utilization metrics,
    never the timing (``FCFSResource`` schedules from ``available_time``), but they
    grow by ~100 objects per decode step and reach 10+ GB on long runs.
    """

    from agentic_sim.core.resource import FCFSResource
    from agentic_sim.hardware.device import Device

    if getattr(FCFSResource, "_history_bounded", False):
        return
    queue_reserve, device_reserve = FCFSResource.reserve, Device._reserve

    def reserve(self, *args, **kwargs):
        result = queue_reserve(self, *args, **kwargs)
        if len(self.reservations) > 2 * keep:
            del self.reservations[:-keep], self._start_times[:-keep], self.queue_samples[:-keep]
        return result

    def reserve_on_device(self, *args, **kwargs):
        result = device_reserve(self, *args, **kwargs)
        if len(self.reservations) > 2 * keep:
            del self.reservations[:-keep]
        return result

    FCFSResource.reserve, Device._reserve = reserve, reserve_on_device
    FCFSResource._history_bounded = True


_POOL_CACHE: dict[str, Any] = {}


def routing_source(config, seed: int, routing_dir: Path | None, model_name: str = DEFAULT_MODEL):
    if routing_dir is None:
        return None
    from power_model.routing import DecodeRowPool, RoutingSource

    model = resolve_model(model_name)
    key = f"{routing_dir}:{model.key}"
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = DecodeRowPool.load(
            routing_dir, num_layers=moe_layer_count(model), num_experts=model.num_experts, top_k=model.top_k
        )
    return RoutingSource(_POOL_CACHE[key], config.mapping, seed=seed)


# ---------------------------------------------------------------------- traces
def _window_mask(trace, t0: float, t1: float) -> np.ndarray:
    lower, upper = trace.t_edges_s[:-1], trace.t_edges_s[1:]
    return (lower >= t0 - 1e-12) & (upper <= t1 + 1e-12)


def die_power_stats(trace, mask: np.ndarray) -> dict[str, Any]:
    """Mean power over the masked bins, per NMP node and GPU chip, with per-component detail."""

    widths = np.diff(trace.t_edges_s)[mask]
    power = trace.power_w[mask].astype(float)
    mean = (power * widths[:, None]).sum(axis=0) / widths.sum()
    peak = power.max(axis=0)
    kinds = np.array([die.kind for die in trace.dies])
    chips = np.array([die.chip for die in trace.dies])
    components = np.array([die.component for die in trace.dies])
    result: dict[str, Any] = {}
    for kind in ("nmp", "gpu"):
        selected = kinds == kind
        names = sorted(set(chips[selected]))
        chip_mean = np.array([mean[chips == name].sum() for name in names])
        chip_peak = np.array([power[:, chips == name].sum(axis=1).max() for name in names])
        entry = {
            "chips": len(names),
            "chip_mean_w": float(chip_mean.mean()),
            "chip_mean_max_w": float(chip_mean.max()),
            "chip_mean_min_w": float(chip_mean.min()),
            "chip_peak_w": float(chip_peak.max()),
            "sum_mean_w": float(chip_mean.sum()),
        }
        for component in sorted(set(components[selected])):
            mask_c = selected & (components == component)
            if component.startswith("dram"):
                continue
            entry[f"{component}_mean_w"] = float(mean[mask_c].mean())
            entry[f"{component}_max_w"] = float(mean[mask_c].max())
            entry[f"{component}_peak_w"] = float(peak[mask_c].max())
        dram = selected & np.char.startswith(components.astype(str), "dram")
        if dram.any():
            entry["dram_die_mean_w"] = float(mean[dram].mean())
            entry["dram_die_max_w"] = float(mean[dram].max())
            entry["dram_die_peak_w"] = float(peak[dram].max())
        result[kind] = entry
    result["panel_mean_w"] = float(mean.sum())
    result["panel_static_w"] = float(trace.static_w.sum())
    return result


# ------------------------------------------------------------------------ grid
def grid_cell(task: dict[str, Any]) -> dict[str, Any]:
    directory = Path(task["config_dir"])
    panel = Panel(**task["panel"])
    batch, context = task["batch"], task["context"]
    prefill, output = task["prefill_tokens"], task["output_tokens"]
    config, report = load_panel(directory)
    capacity = min(
        device.capabilities.decode_attn.kv_capacity_tokens
        for device in config.hardware.devices
        if device.capabilities.decode_attn is not None
    )
    footprint = context + prefill + output
    per_node = capacity // footprint
    record: dict[str, Any] = {
        "panel": panel.name,
        "batch": batch,
        "context": context,
        "node_kv_capacity_tokens": capacity,
        "sessions_per_node_max": per_node,
        "panel_sessions_max": per_node * panel.nodes,
    }
    if per_node == 0:
        return {**record, "feasible": False, "reason": "one session's KV exceeds a node"}
    if batch > per_node * panel.nodes:
        return {**record, "feasible": False, "reason": "batch KV exceeds panel capacity"}

    power = panel_power(panel, Path(task["power_yaml"]), task["dt_s"])
    specs = tuple(
        SessionSpec(index, 0.0, context, (TurnSpec(turn_index=0, prefill_tokens=prefill, output_tokens=output, tool_wait_s=0.0),))
        for index in range(batch)
    )
    routing = routing_source(
        config, task["seed"], Path(task["routing_dir"]) if task["routing_dir"] else None, panel.model
    )
    model = resolve_model(panel.model)
    run = simulate(config, power, model, PrecisionSpec(), report=report, session_specs=specs, routing=routing)
    steps = run.snapshot.decode_steps
    summary = run.snapshot.summary
    full = [step for step in steps if step.active_sessions == batch]
    achieved = batch
    if len(full) < 8:
        achieved = max(step.active_sessions for step in steps)
        full = [step for step in steps if step.active_sessions == achieved]
    t0, t1 = full[0].ready_time_s, full[-1].finish_time_s
    durations = np.array([step.finish_time_s - step.ready_time_s for step in full])
    emitted = sum(step.emitted_tokens for step in full)
    mask = _window_mask(run.trace, t0, t1)
    decode_power = die_power_stats(run.trace, mask) if mask.any() else None
    prefill_end = min(step.ready_time_s for step in full)
    prefill_mask = _window_mask(run.trace, 0.0, prefill_end)
    prefill_power = die_power_stats(run.trace, prefill_mask) if prefill_mask.any() else None
    window_energy = None
    if decode_power is not None:
        window_s = float(np.diff(run.trace.t_edges_s)[mask].sum())
        window_tokens = emitted * window_s / (t1 - t0)
        window_energy = {
            "panel_j_per_token": decode_power["panel_mean_w"] * window_s / window_tokens,
            "nmp_j_per_token": decode_power["nmp"]["sum_mean_w"] * window_s / window_tokens,
            "nmp_dynamic_j_per_token": (
                decode_power["nmp"]["sum_mean_w"]
                - run.trace.static_w[[die.kind == "nmp" for die in run.trace.dies]].sum()
            )
            * window_s
            / window_tokens,
        }
    categories = dict(zip(run.trace.meta["energy_columns"], run.trace.energy_j.sum(axis=0)))
    return {
        **record,
        "feasible": True,
        "achieved_batch": achieved,
        "window_steps": len(full),
        "window_s": t1 - t0,
        "mean_context_in_window": context + prefill + output / 2,
        "tpot_ms": 1e3 * (t1 - t0) / len(full),
        "step_median_ms": 1e3 * float(np.median(durations)),
        "attention_barrier_median_ms": 1e3 * float(np.median([s.attention_time_s for s in full])),
        "ffn_median_ms": 1e3 * float(np.median([s.ffn_time_s for s in full])),
        "all_to_all_median_ms": 1e3 * float(np.median([s.ffn_all_to_all_time_s for s in full])),
        "decode_tokens_per_s": emitted / (t1 - t0),
        "mean_ttft_ms": summary.get("mean_ttft_ms"),
        "p95_ttft_ms": summary.get("p95_ttft_ms"),
        "decode_start_spread_s": prefill_end,
        "decode_window_power": decode_power,
        "prefill_phase_power": prefill_power,
        "decode_window_energy": window_energy,
        "run_energy_j_by_category": {key: float(value) for key, value in categories.items()},
        "wall_s": run.wall_s,
    }


GRID_COLUMNS = (
    "panel", "batch", "context", "feasible", "reason", "achieved_batch", "tpot_ms", "decode_tokens_per_s",
    "attention_barrier_median_ms", "ffn_median_ms", "all_to_all_median_ms", "mean_ttft_ms",
    "nmp_chip_mean_w", "nmp_chip_mean_max_w", "nmp_logic_mean_w", "nmp_logic_max_w", "nmp_dram_die_mean_w",
    "nmp_dram_die_max_w", "nmp_chip_peak_w", "gpu_chip_mean_prefill_w", "gpu_chip_peak_prefill_w",
    "panel_mean_decode_w", "panel_j_per_token", "nmp_dynamic_j_per_token", "window_steps", "sessions_per_node_max",
)


def flatten_grid(record: dict[str, Any]) -> dict[str, Any]:
    row = {key: record.get(key) for key in GRID_COLUMNS}
    decode = record.get("decode_window_power")
    if decode:
        nmp = decode["nmp"]
        row.update(
            nmp_chip_mean_w=nmp["chip_mean_w"],
            nmp_chip_mean_max_w=nmp["chip_mean_max_w"],
            nmp_logic_mean_w=nmp.get("logic_mean_w"),
            nmp_logic_max_w=nmp.get("logic_max_w"),
            nmp_dram_die_mean_w=nmp.get("dram_die_mean_w"),
            nmp_dram_die_max_w=nmp.get("dram_die_max_w"),
            nmp_chip_peak_w=nmp["chip_peak_w"],
            panel_mean_decode_w=decode["panel_mean_w"],
        )
    prefill = record.get("prefill_phase_power")
    if prefill:
        row.update(gpu_chip_mean_prefill_w=prefill["gpu"]["chip_mean_w"], gpu_chip_peak_prefill_w=prefill["gpu"]["chip_peak_w"])
    energy = record.get("decode_window_energy")
    if energy:
        row.update(panel_j_per_token=energy["panel_j_per_token"], nmp_dynamic_j_per_token=energy["nmp_dynamic_j_per_token"])
    return row


def run_grid(args) -> int:
    tp = args.tp or required_tp(args.model)
    panel = Panel(
        args.name or f"g{args.gpus}_n{args.nodes}x{args.stacks}", args.gpus, tp, args.nodes, args.stacks, args.model
    )
    out = args.out
    config_dir = build_panel(panel, out / "config", None if args.no_routing else args.routing_dir)
    tasks = [
        {
            "config_dir": str(config_dir),
            "panel": panel.__dict__,
            "batch": batch,
            "context": context,
            "prefill_tokens": args.prefill_tokens,
            "output_tokens": args.output_tokens,
            "power_yaml": str(args.power),
            "dt_s": args.dt,
            "seed": args.seed,
            "routing_dir": None if args.no_routing else str(args.routing_dir),
        }
        for context in args.contexts
        for batch in args.batches
    ]
    # Longest cells first so the pool drains evenly.
    tasks.sort(key=lambda task: -task["batch"] * math.log2(task["context"]))
    started = time.time()
    with Pool(args.jobs) as pool:
        records = pool.map(grid_cell, tasks, chunksize=1)
    records.sort(key=lambda record: (record["context"], record["batch"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.json").write_text(
        json.dumps({"panel": panel.__dict__, "describe": panel.describe(), "args": {
            "prefill_tokens": args.prefill_tokens, "output_tokens": args.output_tokens, "dt_s": args.dt,
            "power": str(args.power)}, "cells": records}, indent=2, default=float),
        encoding="utf-8",
    )
    with (out / "grid.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=GRID_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(flatten_grid(record))
    feasible = sum(record["feasible"] for record in records)
    print(f"{panel.describe()}: {feasible}/{len(records)} feasible cells in {time.time() - started:.0f} s -> {out}")
    return 0


# ---------------------------------------------------------------------- splits
DEFAULT_SPLITS = (
    # name, gpus, tp, nodes, stacks  (1:6 is the default ratio)
    ("1to6_16x6", 16, 4, 16, 6),
    ("1to6_24x4", 16, 4, 24, 4),
    ("1to6_12x8", 16, 4, 12, 8),
    ("1to6_16x6_tp8", 16, 8, 16, 6),
    ("1to8_16x8", 16, 4, 16, 8),
    ("1to4_16x4", 16, 4, 16, 4),
    # Iso-power (~23.5 kW at 600 W per H200 + 144.5 W per stack, perf_sim's Stratum-L budget).
    ("1to16_8x16", 8, 4, 16, 8),
    ("1to16_8x32x4", 8, 4, 32, 4),
    ("1to9.3_12x14x8", 12, 4, 14, 8),
    ("1to4_20x20x4", 20, 4, 20, 4),
    ("1to2.7_24x16x4", 24, 4, 16, 4),
    # GPU-heavy but with 8-stack nodes, so the longest sessions still fit on one node.
    ("1to2.7_24x8x8", 24, 4, 8, 8),
    ("1to4_20x10x8", 20, 4, 10, 8),
    ("1to1.4_28x5x8", 28, 4, 5, 8),
    # PanAgretti paper split with perf_sim-default nodes (not iso-power: ~37 kW).
    ("1to28_8x56x4", 8, 4, 56, 4),
)


def split_run(task: dict[str, Any]) -> dict[str, Any]:
    import power_trace as trace_script

    bound_reservation_history()
    panel = Panel(**task["panel"])
    directory = Path(task["config_dir"])
    config, report = load_panel(directory)
    power = panel_power(panel, Path(task["power_yaml"]), task["dt_s"])
    with SESSIONS_JSON.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    from run_sweeps import build_sessions, steady_state

    specs = build_sessions(
        payload, count=task["num_sessions"], arrival_rate=task["arrival_rate"], seed=task["seed"],
        context_scale=1.0, output_scale=1.0,
    )
    offered_turns = {spec.session_id: len(spec.turns) for spec in specs}
    specs, fitting = trace_script.fit_sessions_to_capacity(specs, trace_script.attention_capacity_tokens(config))
    routing = routing_source(
        config, task["seed"], Path(task["routing_dir"]) if task["routing_dir"] else None, panel.model
    )
    model = resolve_model(panel.model)
    horizon = task.get("horizon_s")
    if horizon is not None:
        config = replace(config, workload=replace(config.workload, simulation_end_s=horizon))
    run = simulate(
        config, power, model, PrecisionSpec(), report=report, session_specs=specs, routing=routing,
        horizon_s=horizon, collector_class=SlimCollector,
    )
    snapshot = run.snapshot
    summary = snapshot.summary
    sessions = [item for item in snapshot.sessions if item.completion_latency_s is not None]
    # A run that drains its calendar with sessions still waiting for KV capacity is a
    # stall (the FCFS admission mode left open in docs/power_audit.md): its averages
    # cover only the sessions that finished before the panel locked up.
    waiting = len(run.runtime.attention.waiters)
    stall = {
        "stalled": waiting > 0 or len(sessions) < len(specs),
        "sessions_waiting_for_capacity": waiting,
        "admission_deadlock_releases": run.runtime.attention.deadlock_releases,
        "longest_decode_step_s": max(
            (step.finish_time_s - step.ready_time_s for step in snapshot.decode_steps), default=0.0
        ),
    }
    serving = [item.completion_latency_s - item.tool_wait_s for item in sessions]
    trace = run.trace
    widths = np.diff(trace.t_edges_s)
    mask = np.ones(len(widths), dtype=bool)
    stats = die_power_stats(trace, mask)
    total_j = float((trace.power_w * widths[:, None]).sum())
    tokens = summary["emitted_tokens"]
    kinds = np.array([die.kind for die in trace.dies])
    nmp_static = float(trace.static_w[kinds == "nmp"].sum())
    gpu_static = float(trace.static_w[kinds == "gpu"].sum())
    categories = dict(zip(trace.meta["energy_columns"], trace.energy_j.sum(axis=0)))
    panel_peak_w = float(trace.power_w.sum(axis=1).max())
    return {
        "name": panel.name,
        "describe": panel.describe(),
        "panel": panel.__dict__,
        "node_kv_capacity_tokens": trace_script.attention_capacity_tokens(config),
        "panel_kv_capacity_tokens": trace_script.attention_capacity_tokens(config) * panel.nodes,
        "session_fit": fitting,
        "stall": stall,
        "sessions": len(specs),
        "completed_sessions": len(sessions),
        "makespan_s": summary["simulation_time_s"],
        "emitted_tokens": tokens,
        "mean_ttft_ms": summary["mean_ttft_ms"],
        "p95_ttft_ms": summary["p95_ttft_ms"],
        "mean_tpot_ms": summary["mean_inter_token_ms"],
        "steady_decode_tokens_per_s": steady_state(snapshot.decode_steps),
        # Tokens per second of time with any decode step running (excludes idle gaps).
        "busy_decode_tokens_per_s": tokens / max(
            1e-12, sum(step.finish_time_s - step.ready_time_s for step in snapshot.decode_steps)
        ),
        "mean_active_sessions_per_step": float(np.mean([step.active_sessions for step in snapshot.decode_steps])),
        "mean_serving_s": float(np.mean(serving)),
        "p95_serving_s": float(np.percentile(serving, 95)),
        "mean_session_latency_s": float(np.mean([item.completion_latency_s for item in sessions])),
        "mean_capacity_wait_s": float(np.mean([turn.attention_capacity_wait_s for turn in snapshot.turns])),
        "mean_prefill_queue_s": float(np.mean([turn.prefill_queue_s for turn in snapshot.turns])),
        "attention_retries": summary.get("attention_retries"),
        "panel_mean_w": total_j / trace.horizon_s,
        "panel_peak_w_bin": panel_peak_w,
        "panel_static_w": float(trace.static_w.sum()),
        "nmp_static_w": nmp_static,
        "gpu_static_w": gpu_static,
        "panel_rated_w": panel.gpus * 600.0 + panel.total_stacks * 144.53,
        "energy_j": total_j,
        "j_per_output_token": total_j / tokens,
        "energy_j_by_category": {key: float(value) for key, value in categories.items()},
        "power": stats,
        "wall_s": run.wall_s,
        # session id -> [latency s, tool wait s, completed turns, offered turns before capacity fit]
        "per_session": {
            str(item.session_id): [item.completion_latency_s, item.tool_wait_s, item.completed_turns,
                                   offered_turns[item.session_id]]
            for item in snapshot.sessions
        },
    }


def run_splits(args) -> int:
    minimum_tp = required_tp(args.model)
    panels: list[Panel] = []
    seen: set[tuple[int, int, int, int]] = set()
    for name, gpus, tp, nodes, stacks in DEFAULT_SPLITS:
        if args.only and name not in set(args.only):
            continue
        tp = max(tp, minimum_tp)
        if gpus % tp:
            print(f"skip {name}: {gpus} H200 is not a multiple of the TP{tp} that {args.model} needs")
            continue
        if (gpus, tp, nodes, stacks) in seen:
            print(f"skip {name}: duplicates an earlier split at TP{tp}")
            continue
        seen.add((gpus, tp, nodes, stacks))
        panels.append(Panel(name, gpus, tp, nodes, stacks, args.model))
    out = args.out
    tasks = []
    for panel in panels:
        directory = build_panel(panel, out / "configs" / panel.name, None if args.no_routing else args.routing_dir)
        tasks.append(
            {
                "config_dir": str(directory),
                "panel": panel.__dict__,
                "power_yaml": str(args.power),
                "dt_s": args.dt,
                "seed": args.seed,
                "num_sessions": args.num_sessions,
                "arrival_rate": args.arrival_rate,
                "routing_dir": None if args.no_routing else str(args.routing_dir),
                "horizon_s": args.horizon,
            }
        )
    started = time.time()
    with Pool(args.jobs) as pool:
        records = pool.map(split_run, tasks, chunksize=1)
    baseline = next((record for record in records if record["name"] == args.baseline), records[0])
    for record in records:
        record["speedup_serving_vs_baseline"] = baseline["mean_serving_s"] / record["mean_serving_s"]
        record["speedup_tpot_vs_baseline"] = baseline["mean_tpot_ms"] / record["mean_tpot_ms"]
        record["speedup_ttft_vs_baseline"] = baseline["mean_ttft_ms"] / record["mean_ttft_ms"]
        record["speedup_makespan_vs_baseline"] = baseline["makespan_s"] / record["makespan_s"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(
        json.dumps({"baseline": baseline["name"], "workload": {
            "sessions_json": str(SESSIONS_JSON), "num_sessions": args.num_sessions,
            "arrival_rate": args.arrival_rate, "seed": args.seed}, "runs": records}, indent=2, default=float),
        encoding="utf-8",
    )
    columns = (
        "name", "describe", "panel_rated_w", "panel_kv_capacity_tokens", "node_kv_capacity_tokens", "completed_sessions",
        "mean_serving_s", "speedup_serving_vs_baseline", "mean_tpot_ms", "speedup_tpot_vs_baseline", "mean_ttft_ms",
        "p95_ttft_ms", "speedup_ttft_vs_baseline", "makespan_s", "busy_decode_tokens_per_s",
        "mean_active_sessions_per_step", "mean_capacity_wait_s", "mean_prefill_queue_s", "panel_mean_w",
        "panel_static_w", "panel_peak_w_bin", "j_per_output_token",
    )
    with (out / "splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    print(f"{len(records)} splits in {time.time() - started:.0f} s -> {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("grid", "splits"):
        command = sub.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--power", type=Path, default=POWER_YAML)
        command.add_argument("--routing-dir", type=Path, default=ROUTING_DIR)
        command.add_argument("--no-routing", action="store_true", help="equal routing (timing is identical)")
        command.add_argument("--seed", type=int, default=11)
        command.add_argument("--jobs", type=int, default=8)
        command.add_argument("--model", default=DEFAULT_MODEL, help="roofline_model alias (default qwen3-235b)")
    grid = sub.choices["grid"]
    grid.add_argument("--name")
    grid.add_argument("--gpus", type=int, default=16)
    grid.add_argument(
        "--tp", type=int, help="prefill TP (default: smallest power of two whose HBM holds the FP16 weights)"
    )
    grid.add_argument("--nodes", type=int, default=16)
    grid.add_argument("--stacks", type=int, default=6)
    grid.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128, 256])
    grid.add_argument("--contexts", type=int, nargs="+", default=[1024 * 2**k for k in range(11)])
    grid.add_argument("--prefill-tokens", type=int, default=128)
    grid.add_argument("--output-tokens", type=int, default=256)
    grid.add_argument("--dt", type=float, default=0.001, help="power bin width (s)")
    splits = sub.choices["splits"]
    splits.add_argument("--num-sessions", type=int, default=300)
    splits.add_argument("--arrival-rate", type=float, default=4.0)
    splits.add_argument("--dt", type=float, default=0.1, help="power bin width (s)")
    splits.add_argument("--baseline", default="1to6_16x6")
    splits.add_argument("--horizon", type=float, help="simulated seconds (default: run to completion)")
    splits.add_argument("--only", nargs="+", help="run only these split names")
    args = parser.parse_args(argv)
    return run_grid(args) if args.command == "grid" else run_splits(args)


if __name__ == "__main__":
    raise SystemExit(main())
