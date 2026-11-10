"""
Sweep driver: run the Stratum-calibrated configs over the TraceLab workload.

Batch size is an OUTPUT here, not an input. AttentionAdmissionController gates
on context + oracle output against kv_capacity_tokens per device, with strict
FCFS head-of-line blocking, so concurrency is whatever the panel can sustain.
We sweep arrival rate and read achieved concurrency off DecodeStepMetric.

Sessions are replayed from real traces (preserving within-session correlation)
and held byte-identical across hardware arms, so speedup ratios are clean.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace

from agentic_sim.config import load_config
from agentic_sim.config.models import SimulationConfig
from agentic_sim.metrics import MetricsCollector
from agentic_sim.metrics.records import DecodeStepMetric
from agentic_sim.runtime.runtime import AgenticRuntime
from agentic_sim.workload import SessionSpec, TurnSpec

CONFIG_ROOT = Path("configs/glassml")
ARMS_DEFAULT = ("nvl64_baseline", "glassml_arm_a", "glassml_arm_b")
ARMS = ("nvl64_baseline", "glassml_arm_a", "glassml_arm_b")


class TrimmedCollector(MetricsCollector):
    """
    Same numbers, far less garbage.

    A 64-device run appends one AttentionDeviceMetric per device on every decode
    step and a 64-entry tuple per step for expert loads, FFN devices, and
    attention waits. None of that reaches `summary`, and expert loads are
    deterministic (ExpertMapping.equal_loads) so they carry no per-step
    information. Dropping them is verified summary-identical by
    --verify-collector.
    """

    def __init__(self, attention_sample_interval: int = 32) -> None:
        super().__init__()
        self._interval = attention_sample_interval
        self._captures = 0

    def record_decode(self, result) -> None:
        loads = tuple(batch.activation_tokens for batch in result.expert_batches)
        self.decode_steps.append(DecodeStepMetric(
            decode_step=result.decode_step,
            ready_time_s=result.batch.ready_time_s,
            finish_time_s=result.token_completion_time_s,
            active_sessions=result.batch.size,
            attention_barrier_s=result.attention_barrier_s,
            attention_device_wait_s=(),
            routed_activation_tokens=result.activation_tokens,
            send_tokens=result.send_tokens,
            dispatch_latency_s=result.dispatch.service_time,
            combine_latency_s=result.combine.service_time,
            expert_loads=(),
            ffn_devices=(),
            ffn_load_imbalance_tokens=(max(loads) - min(loads)) if loads else 0,
            emitted_tokens=result.batch.size,
        ))
        for pending in result.pool_writes:
            self.record_pool_write(pending)

    def capture_attention(self, time_s, attention) -> None:
        self._captures += 1
        if self._captures % self._interval and self._captures != 1:
            return
        super().capture_attention(time_s, attention)


def build_sessions(payload, *, count, arrival_rate, seed, context_scale, output_scale):
    segments = payload["segments"]
    rng = random.Random(seed)
    chosen = [segments[rng.randrange(len(segments))] for _ in range(count)]
    sessions, clock = [], 0.0
    for index, segment in enumerate(chosen):
        clock += rng.expovariate(arrival_rate)
        turns = tuple(
            TurnSpec(turn_index=position,
                     prefill_tokens=max(0, int(turn["prefill_tokens"] * context_scale)),
                     output_tokens=max(1, int(turn["output_tokens"] * output_scale)),
                     tool_wait_s=turn["tool_wait_ms"] / 1000.0)
            for position, turn in enumerate(segment["turns"])
        )
        sessions.append(SessionSpec(
            session_id=index,
            arrival_time_s=clock,
            initial_context_tokens=max(1, int(segment["initial_context_tokens"] * context_scale)),
            turns=turns,
        ))
    return tuple(sessions)


def steady_state(steps, low=0.2, high=0.8):
    """
    Throughput over the middle of the run.

    summary['decode_token_throughput_per_s'] divides by the whole horizon, which
    includes ramp-up and drain, so it understates a saturated panel badly.
    """
    if len(steps) < 10:
        return None
    lo, hi = int(low * len(steps)), int(high * len(steps))
    window = steps[lo:hi]
    span = window[-1].finish_time_s - window[0].ready_time_s
    if span <= 0:
        return None
    return sum(item.emitted_tokens for item in window) / span


def run_one(arm, sessions, *, workload_yaml, attention_sample_interval, horizon=None,
            config_root=None):
    root = Path(config_root) if config_root else CONFIG_ROOT
    config = load_config(str(root / arm / "hardware.yaml"), str(workload_yaml),
                         str(root / arm / "mapping.yaml"))
    if horizon is not None:
        # A real agentic session spends most of its life waiting on tools (mean
        # 32 s), so run-to-completion spends almost all its steps on an idle
        # panel. Bound the simulated window instead and measure steady state
        # inside it; throughput comes from decode steps, not from completions.
        config = SimulationConfig(
            hardware=config.hardware,
            workload=replace(config.workload, simulation_end_s=horizon),
            mapping=config.mapping,
        )
    runtime = AgenticRuntime(config, session_specs=sessions)
    runtime.metrics = TrimmedCollector(attention_sample_interval)
    runtime.metrics.capture_attention(0.0, runtime.attention)
    started = time.time()
    snapshot = runtime.run()
    wall = time.time() - started

    steps = list(snapshot.decode_steps)
    summary = dict(snapshot.summary)
    concurrency = [item.active_sessions for item in steps] or [0]
    step_latency = [item.finish_time_s - item.ready_time_s for item in steps] or [0.0]
    dispatch = [item.dispatch_latency_s for item in steps] or [0.0]
    combine = [item.combine_latency_s for item in steps] or [0.0]

    # Device-side endpoint reservations and the logical fabric queues both tag
    # their busy time with the same operation string (e.g. every all-to-all
    # reserves ~65 endpoint queues AND the fabric queue). Summing them into one
    # key produced nonsense like 10,659 busy-seconds inside a 300 s run. Keep
    # device-seconds and fabric-seconds under separate keys.
    utilization = {}
    for record in snapshot.resources:
        prefix = "" if record.resource_type == "physical_device" else "fabric:"
        for operation, seconds in record.operation_busy_time_s:
            key = prefix + operation
            utilization[key] = utilization.get(key, 0.0) + seconds

    return dict(
        arm=arm,
        wall_seconds=round(wall, 1),
        decode_steps=summary["decode_steps"],
        emitted_tokens=summary["emitted_tokens"],
        completed_sessions=summary["completed_sessions"],
        offered_sessions=summary["offered_sessions"],
        simulation_time_s=summary["simulation_time_s"],
        steady_state_tokens_per_s=steady_state(steps),
        mean_concurrency=statistics.mean(concurrency),
        p95_concurrency=sorted(concurrency)[int(0.95 * (len(concurrency) - 1))],
        max_concurrency=max(concurrency),
        mean_ttft_ms=summary["mean_ttft_ms"],
        p95_ttft_ms=summary["p95_ttft_ms"],
        mean_inter_token_ms=summary["mean_inter_token_ms"],
        mean_step_latency_ms=statistics.mean(step_latency) * 1000.0,
        mean_dispatch_ms=statistics.mean(dispatch) * 1000.0,
        mean_combine_ms=statistics.mean(combine) * 1000.0,
        attention_retries=summary["attention_retries"],
        lru_evictions=summary["lru_evictions"],
        attention_cache_hits=summary["attention_cache_hits"],
        attention_cache_misses=summary["attention_cache_misses"],
        operation_busy_s=utilization,
    )


def verify_collector(sessions, workload_yaml):
    """Prove the trimmed collector changes no reported number."""
    arm = ARMS[1]
    config = load_config(str(CONFIG_ROOT / arm / "hardware.yaml"), str(workload_yaml),
                         str(CONFIG_ROOT / arm / "mapping.yaml"))
    baseline = AgenticRuntime(config, session_specs=sessions).run().summary
    runtime = AgenticRuntime(config, session_specs=sessions)
    runtime.metrics = TrimmedCollector(32)
    runtime.metrics.capture_attention(0.0, runtime.attention)
    trimmed = runtime.run().summary
    differing = {k: (baseline[k], trimmed[k]) for k in baseline if baseline[k] != trimmed[k]}
    return differing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-json", type=Path,
                        default=CONFIG_ROOT / "workload" / "sessions.json")
    parser.add_argument("--workload-yaml", type=Path,
                        default=CONFIG_ROOT / "workload" / "workload.yaml")
    parser.add_argument("--out", type=Path, default=Path("configs/glassml/results.json"))
    parser.add_argument("--num-sessions", type=int, default=120)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--arrival-rates", type=float, nargs="*", default=[2.0])
    parser.add_argument("--context-scales", type=float, nargs="*", default=[1.0])
    parser.add_argument("--output-scales", type=float, nargs="*", default=[1.0])
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--attention-sample-interval", type=int, default=32)
    parser.add_argument("--config-root", type=Path, default=None,
                        help="directory of <name>/{hardware,mapping}.yaml configs")
    parser.add_argument("--horizon", type=float, default=None,
                        help="bound each run to this many simulated seconds")
    parser.add_argument("--verify-collector", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(args.sessions_json.read_text())

    if args.verify_collector:
        sessions = build_sessions(payload, count=8, arrival_rate=1.0, seed=1,
                                  context_scale=0.02, output_scale=0.02)
        differing = verify_collector(sessions, args.workload_yaml)
        print("collector equivalence:",
              "IDENTICAL across all summary keys" if not differing else differing)
        return 0 if not differing else 1

    rows = []
    grid = [(rate, cs, os_)
            for rate in args.arrival_rates
            for cs in args.context_scales
            for os_ in args.output_scales]
    for rate, context_scale, output_scale in grid:
        sessions = build_sessions(payload, count=args.num_sessions, arrival_rate=rate,
                                  seed=args.seed, context_scale=context_scale,
                                  output_scale=output_scale)
        for arm in args.arms:
            try:
                row = run_one(arm, sessions, workload_yaml=args.workload_yaml,
                              attention_sample_interval=args.attention_sample_interval,
                              horizon=args.horizon, config_root=args.config_root)
            except Exception as exc:  # noqa: BLE001 - record, do not abort the grid
                row = dict(arm=arm, failed=f"{type(exc).__name__}: {exc}")
                print(f"  {arm:16s} rate={rate:<5g} ctx={context_scale:<5g} "
                      f"out={output_scale:<5g} FAILED: {type(exc).__name__}: {exc}")
            else:
                print(f"  {arm:16s} rate={rate:<5g} ctx={context_scale:<5g} out={output_scale:<5g} "
                      f"steps={row['decode_steps']:>7d} conc={row['mean_concurrency']:6.1f} "
                      f"ITL={row['mean_inter_token_ms']:8.1f}ms wall={row['wall_seconds']}s")
            row.update(arrival_rate=rate, context_scale=context_scale,
                       output_scale=output_scale, num_sessions=args.num_sessions,
                       horizon=args.horizon)
            rows.append(row)
            args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
