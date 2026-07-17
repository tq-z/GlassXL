# Agentic Sim

Agentic Sim is the PR1--PR3 implementation of the top-down agentic inference
simulator described in [the implementation plan](docs/IMPLEMENTATION_PLAN.md).
It is a deterministic, typed Python discrete-event simulator with synthetic
multi-turn agents and a coarse prefill/decode-disaggregated hardware model.

## Scope

This version contains:

- a deterministic event engine and generic task DAGs;
- Poisson arrivals and configurable workload distributions;
- multi-turn agents, context growth, and tool-wait suspend/resume behavior;
- distinct prefill and decode devices with phase-specific token rates;
- serialized FCFS compute, memory, and directed latency-plus-bandwidth links;
- configurable `memory`, `far_memory`, or `discard` KV retention;
- static task placement and in-memory request/resource/system metrics.

It intentionally does not contain xPU/NMP equations, MoE, optimized scheduling,
passive optical details, traces, sweeps/plots, persistent metrics export, or
ASTRA-Sim integration.

## Coarse P-D model

Each compute device declares `prefill_tokens_per_second` and
`decode_tokens_per_second`. The static policy maps the two phases to different
devices with independent FCFS queues.

```text
First turn:
prefill new input -> transfer prompt KV P->D -> decode

Later retained turn:
read prior KV -> restore to P -> prefill new input -> transfer prompt KV P->D -> decode

Later discarded turn:
prefill prior context + new input -> transfer prompt KV P->D -> decode
```

KV bytes equal token count times `kv.bytes_per_token`. For retained KV, a
non-final decode forks into tool wait and background KV storage. The next turn
waits for both; if the tool returns first, the agent enters `KV_WAIT`. `memory`
and `far_memory` use the same generic resource abstractions and differ only in
their configured memory/link performance.

Invocation completion is decode completion. Background storage contributes to
agent and resource accounting but does not extend invocation completion time.
Memory capacity is descriptive; allocation and eviction are not modeled.

## Architecture

```text
YAML configuration
  -> seeded WorkloadGenerator
  -> AgenticRuntime / hardware-independent TaskBuilder / ExecutionPolicy
  -> EventEngine
  -> SimpleHardwareBackend
  -> MetricsCollector
```

The event engine owns time but has no workload or hardware knowledge. Placement
maps logical task phases, transfer directions, and retention operations onto
generic devices, memories, and links. Later hardware backends can replace these
models without changing the event engine.

## Run

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
.venv/bin/python examples/run_simulation.py
```

The example prints a JSON system summary followed by per-resource metrics.
Metrics remain in memory in this PR3 scope; file export and sweep tooling belong
to PR4.

## Configuration notes

Compute devices require both phase token rates even when statically assigned one
role. A retained mode requires the selected generic memory resource plus D-to-
memory and memory-to-P links. P-to-D is always required. A
`default_all_to_all_link` can provide missing direct links lazily; there is no
routing or shared-fabric model.
