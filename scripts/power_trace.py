#!/usr/bin/env python3
"""Run a simulation with the power collector and write per-die power traces.

  python scripts/power_trace.py --config-dir configs/power/smoke \
      --power configs/power/panagretti_b200.yaml \
      --sessions-json configs/glassml/workload/sessions.json --num-sessions 50 \
      --arrival-rate 2 --seed 11 --horizon 90 --out results/power/smoke_equal

Re-price an earlier run with edited coefficients, without re-simulating:

  python scripts/power_trace.py --from-activity results/power/smoke_equal/activity.npz \
      --config-dir configs/power/smoke --power edited.yaml --out results/power/repriced

Outputs in --out: activity.npz (binned counters), power.npz (power_w[T, die]),
dies.csv and chips.csv (mean/p95/peak power and energy by category), and
run_summary.json (simulator summary, energy totals, checks, timing diagnostics).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentic_sim.config import load_config  # noqa: E402
from agentic_sim.config.models import SimulationConfig  # noqa: E402
from agentic_sim.metrics import MetricsCollector  # noqa: E402
from agentic_sim.runtime.runtime import AgenticRuntime  # noqa: E402
from power_model import (  # noqa: E402
    BinnedActivity,
    DeviceLayout,
    load_power_config,
    price,
    write_outputs,
)
from power_model.checks import anchors, completeness, conservation, io_balance  # noqa: E402
from power_model.simulate import simulate  # noqa: E402
from roofline_model import PrecisionSpec, resolve_model  # noqa: E402
from roofline_model.activity import moe_layer_count  # noqa: E402
from run_sweeps import build_sessions  # noqa: E402

DEFAULT_WORKLOAD = REPOSITORY_ROOT / "configs" / "glassml" / "workload" / "workload.yaml"


def load_report(config_dir: Path) -> dict[str, Any] | None:
    path = config_dir / "roofline_report.yaml"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_simulation(config_dir: Path, workload_yaml: Path, horizon_s: float | None) -> SimulationConfig:
    config = load_config(config_dir / "hardware.yaml", workload_yaml, config_dir / "mapping.yaml")
    if horizon_s is not None:
        config = SimulationConfig(
            hardware=config.hardware,
            workload=replace(config.workload, simulation_end_s=horizon_s),
            mapping=config.mapping,
        )
    return config


def resolve_precision(power, report: dict[str, Any] | None) -> PrecisionSpec:
    if power.precision is not None:
        return power.precision
    document = ((report or {}).get("inputs") or {}).get("precision")
    return PrecisionSpec(**document) if document else PrecisionSpec()


def session_specs(args) -> tuple | None:
    if args.sessions_json is None:
        return None
    with Path(args.sessions_json).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return build_sessions(
        payload,
        count=args.num_sessions,
        arrival_rate=args.arrival_rate,
        seed=args.seed,
        context_scale=args.context_scale,
        output_scale=args.output_scale,
    )


def fit_sessions_to_capacity(specs, capacity_tokens: int) -> tuple[tuple, dict[str, int]]:
    """Keep each session's turns while its KV footprint fits one attention device.

    Attention admission is strict FCFS, so one session that can never fit on a
    node (e.g. a 945K-token context against 553K tokens of NMP KV capacity)
    blocks every later session and drains the event calendar. Sessions whose
    first turn cannot fit are dropped.
    """

    kept = []
    dropped = turns_cut = 0
    for spec in specs:
        context = spec.initial_context_tokens
        turns = []
        for turn in spec.turns:
            footprint = context + turn.prefill_tokens + turn.output_tokens
            if footprint > capacity_tokens:
                break
            context = footprint
            turns.append(turn)
        turns_cut += len(spec.turns) - len(turns)
        if not turns:
            dropped += 1
            continue
        kept.append(replace(spec, turns=tuple(turns)))
    return tuple(kept), {
        "capacity_tokens": capacity_tokens,
        "sessions_offered": len(specs),
        "sessions_dropped": dropped,
        "turns_cut": turns_cut,
    }


def attention_capacity_tokens(sim_config: SimulationConfig) -> int:
    capacities = [
        device.capabilities.decode_attn.kv_capacity_tokens
        for device in sim_config.hardware.devices
        if device.capabilities.decode_attn is not None
    ]
    return min(capacities)


def build_routing(args, sim_config, model):
    if args.routing == "equal":
        return None
    from power_model.routing import DecodeRowPool, RoutingSource

    pool = DecodeRowPool.load(
        Path(args.routing_dir),
        num_layers=moe_layer_count(model),
        num_experts=model.num_experts,
        top_k=model.top_k,
        file_glob=args.routing_glob,
    )
    return RoutingSource(pool, sim_config.mapping, seed=args.seed)


def run_power_trace(args) -> dict[str, Any]:
    config_dir = Path(args.config_dir)
    power = load_power_config(args.power)
    report = load_report(config_dir)
    model = resolve_model(power.model)
    precision = resolve_precision(power, report)
    sim_config = load_simulation(config_dir, Path(args.workload_yaml), args.horizon)
    specs = session_specs(args)
    fitting = None
    if specs is not None and not args.no_fit_capacity:
        specs, fitting = fit_sessions_to_capacity(specs, attention_capacity_tokens(sim_config))
    routing = build_routing(args, sim_config, model)

    run = simulate(
        sim_config,
        power,
        model,
        precision,
        report=report,
        session_specs=specs,
        routing=routing,
        horizon_s=args.horizon,
        track_reservations=args.check_completeness,
        prefill_transfers_at_completion=not args.legacy_prefill_transfers,
    )
    runtime = run.runtime
    waiting = len(runtime.attention.waiters)
    stalled = waiting > 0 and (args.horizon is None or runtime.engine.now < args.horizon)
    steps = run.snapshot.decode_steps
    longest_step = max((step.finish_time_s - step.ready_time_s for step in steps), default=0.0)
    checks: dict[str, Any] = {
        "conservation": conservation(run.binned, run.trace),
        "io_balance": io_balance(run.binned),
        "anchors": anchors(power, run.layout, report),
        "stall": {
            "ok": not stalled,
            "sessions_waiting_for_capacity": waiting,
            "simulation_ended_s": runtime.engine.now,
            # FCFS deadlocks the admission controller resolved by releasing protected
            # cached copies; each one is a moment where the old simulator stopped.
            "admission_deadlock_releases": runtime.attention.deadlock_releases,
            "longest_decode_step_s": longest_step,
        },
    }
    if fitting is not None:
        checks["session_fit"] = {"ok": True, **fitting}
    if args.check_completeness:
        checks["completeness"] = completeness(run.collector, runtime)
    if args.check_parity:
        checks["parity"] = parity(sim_config, specs, run.snapshot, not args.legacy_prefill_transfers)
    if routing is not None:
        checks["routing"] = routing.summary()
    return {
        "sim_config": sim_config,
        "runtime": runtime,
        "snapshot": run.snapshot,
        "collector": run.collector,
        "binned": run.binned,
        "trace": run.trace,
        "power": power,
        "layout": run.layout,
        "checks": checks,
        "wall_s": run.wall_s,
    }


def parity(sim_config: SimulationConfig, specs, snapshot, transfers_at_completion: bool) -> dict[str, Any]:
    """Re-run with the plain collector; the power collector must not change the simulation."""

    runtime = AgenticRuntime(
        sim_config, session_specs=specs, prefill_transfers_at_completion=transfers_at_completion
    )
    runtime.metrics = MetricsCollector()
    runtime.metrics.capture_attention(0.0, runtime.attention)
    reference = runtime.run()
    return {
        "ok": reference.summary == snapshot.summary and reference.decode_steps == snapshot.decode_steps,
        "summary_equal": reference.summary == snapshot.summary,
        "decode_steps_equal": reference.decode_steps == snapshot.decode_steps,
    }


def mean_local_batch(snapshot) -> float:
    active = [step.active_sessions for step in snapshot.decode_steps]
    return 0.0 if not active else sum(active) / len(active)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--power", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workload-yaml", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--sessions-json", type=Path)
    parser.add_argument("--num-sessions", type=int, default=50)
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--context-scale", type=float, default=1.0)
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--horizon", type=float, help="simulated seconds (default: run to completion)")
    parser.add_argument("--routing", choices=("equal", "traces"), default="equal")
    parser.add_argument("--routing-dir", type=Path, default=REPOSITORY_ROOT.parent / "moe_routing")
    parser.add_argument("--routing-glob", default="235b_*.npy")
    parser.add_argument(
        "--no-fit-capacity",
        action="store_true",
        help="keep turns whose KV footprint exceeds one attention device (FCFS admission then stalls)",
    )
    parser.add_argument(
        "--legacy-prefill-transfers",
        action="store_true",
        help="book prefill KV transfers at launch (simulator default; stalls decode behind future transfers)",
    )
    parser.add_argument("--check-parity", action="store_true", help="re-run with the plain collector")
    parser.add_argument("--check-completeness", action="store_true", help="track every reservation")
    parser.add_argument("--from-activity", type=Path, help="re-price an activity.npz without simulating")
    args = parser.parse_args(argv)

    if args.from_activity is not None:
        power = load_power_config(args.power)
        report = load_report(args.config_dir)
        sim_config = load_simulation(args.config_dir, args.workload_yaml, None)
        layout = DeviceLayout.from_config(sim_config, power, report)
        binned, meta_json = BinnedActivity.load(args.from_activity)
        trace = price(binned, layout, power, report)
        summary = write_outputs(args.out, trace, warmup_s=power.warmup_s)
        summary["repriced_from"] = str(args.from_activity)
        summary["original_run"] = json.loads(meta_json)
        summary["checks"] = {"conservation": conservation(binned, trace), "anchors": anchors(power, layout, report)}
        (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ("energy_j_total", "placeholders")}, indent=2))
        return 0

    result = run_power_trace(args)
    run_meta = {
        "config_dir": str(args.config_dir),
        "power": str(args.power),
        "sessions_json": None if args.sessions_json is None else str(args.sessions_json),
        "num_sessions": args.num_sessions,
        "arrival_rate": args.arrival_rate,
        "seed": args.seed,
        "horizon": args.horizon,
        "routing": args.routing,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    result["binned"].save(args.out / "activity.npz", json.dumps(run_meta))
    summary = write_outputs(args.out, result["trace"], warmup_s=result["power"].warmup_s, extra_meta={"run": run_meta})
    summary.update(
        {
            "run": run_meta,
            "wall_s": result["wall_s"],
            "simulator": result["snapshot"].summary,
            "mean_active_sessions_per_step": mean_local_batch(result["snapshot"]),
            "timing_ratio_simulated_over_activity": result["collector"].timing_summary(),
            "checks": result["checks"],
        }
    )
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    brief = {
        "wall_s": round(result["wall_s"], 2),
        "decode_steps": result["snapshot"].summary.get("decode_steps"),
        "energy_j_total": summary["energy_j_total"],
        "energy_j_by_category": summary["energy_j_by_category"],
        "chips_by_kind": summary["chips_by_kind"],
        "timing": summary["timing_ratio_simulated_over_activity"],
        "checks_ok": {name: check.get("ok") for name, check in result["checks"].items() if isinstance(check, dict)},
    }
    print(json.dumps(brief, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
