# Phase 1 Implementation Summary

## Status

Phase 1, corresponding to Pull Requests 1–3 in the implementation plan, is
complete. The repository contains a deterministic, end-to-end discrete-event
simulator for synthetic multi-turn agent workloads running on generic
prefill/decode compute, memory, and latency-plus-bandwidth link resources.

The implementation deliberately stops before Phase 1 experiment automation and
all architecture-specific later phases. It provides the framework and reference
execution path on which those phases can build.

## Implemented scope

The simulator includes:

- a deterministic priority-queue event engine with sequence-ID tie breaking;
- generic compute, memory, transfer, and wait task abstractions;
- serial FCFS resource reservation and natural queue buildup;
- seeded Poisson arrivals and reusable workload distributions;
- multi-turn agents with context growth and tool waits;
- coarse prefill/decode disaggregation across separate compute devices;
- configurable memory, far-memory, or discard KV retention;
- static task placement onto compute devices, memory resources, and links;
- request, invocation, resource, queue, throughput, and utilization metrics;
- configuration validation, an example configuration, and pytest coverage.

## Architecture

```text
YAML configuration
  -> seeded WorkloadGenerator
  -> AgenticRuntime
       -> hardware-independent TaskBuilder
       -> static ExecutionPolicy
  -> EventEngine
  -> SimpleHardwareBackend
  -> MetricsCollector
```

The event engine owns simulation time but has no agent, model, or hardware
semantics. The task builder creates logical work and dependencies without
physical resource IDs. The execution policy maps that work onto the topology,
and the backend calculates service times and reserves FCFS resources.

This separation is intentional: later hardware models and scheduling policies
can replace the simple backend or reference policy without rewriting the event
engine or synthetic workload generator.

## Workload and agent model

The workload generator uses one `random.Random(seed)` stream for every random
draw. It supports Poisson arrivals plus constant, uniform, exponential,
lognormal, discrete categorical, and bounded-integer distributions.

Each agent contains sampled per-turn input tokens, output tokens, and tool-wait
durations. Context grows after decode by:

```text
new_context = prior_context + input_tokens + output_tokens
```

The agent state machine is:

```text
NOT_ARRIVED
  -> READY
  -> RUNNING
  -> TOOL_WAIT
  -> optional KV_WAIT
  -> READY for the next turn
  -> FINISHED after the final decode
```

`KV_WAIT` occurs when the tool returns before background KV storage finishes.

## Coarse prefill/decode model

Each generic compute device declares both:

- `prefill_tokens_per_second`;
- `decode_tokens_per_second`.

The static reference policy maps prefill and decode to distinct devices with
independent FCFS queues. Compute service times are:

```text
T_prefill = fixed_latency + prefill_tokens / prefill_tokens_per_second
T_decode  = fixed_latency + output_tokens / decode_tokens_per_second
```

The implementation uses one lumped task per phase. It does not simulate
individual decode tokens, Transformer layers, xPU/NMP internals, batching, or
compute/memory overlap inside a device.

## KV retention model

KV size is controlled by `invocation_model.kv.bytes_per_token`. Let
`prior_context` be all tokens completed before a turn:

```text
P-to-D bytes = (prior_context + input_tokens) * kv_bytes_per_token
stored bytes = (prior_context + input_tokens + output_tokens) * kv_bytes_per_token
restore bytes = prior_context * kv_bytes_per_token
```

The global retention mode selects one of three behaviors.

### Memory or far-memory retention

The first turn runs:

```text
prefill new input -> transfer prompt KV P-to-D -> decode
```

A later turn runs:

```text
memory read
  -> restore prior KV to prefill device
  -> prefill new input only
  -> transfer the updated prompt KV P-to-D
  -> decode
```

After a non-final decode, the runtime forks two concurrent branches:

```text
branch 1: tool wait
branch 2: transfer final KV D-to-tier -> memory write
```

The next invocation is released only after both branches complete. Invocation
completion time remains decode completion; background storage is included in
agent and resource accounting but excluded from invocation service and queueing
metrics.

`memory` and `far_memory` use identical generic task semantics. Their behavior
differs only through the selected resource's bandwidth/latency and its directed
links.

### Discard retention

No memory read, write, store, or restore tasks are created. A later turn
recomputes the accumulated prompt:

```text
prefill prior context + new input -> transfer prompt KV P-to-D -> decode
```

## Hardware and placement assumptions

- Prefill and decode device IDs must be distinct.
- Every run requires a direct P-to-D link or a configured default link.
- A retained mode additionally requires D-to-selected-memory and
  selected-memory-to-P links.
- Compute, memory, and links are independently serialized FCFS resources.
- Link transfers use fixed latency plus bytes divided by bandwidth.
- Memory operations use fixed latency plus bytes divided by bandwidth.
- Memory capacity is descriptive; aggregate allocation, eviction, and
  oversubscription are not modeled.
- Stored KV is treated as residing exclusively in the selected tier until it is
  restored.
- Final-turn KV is not stored because no later turn consumes it.

## Metrics

Completed-agent metrics include arrival/start/finish time, latency, summed
hardware service time, tool-wait time, summed queueing delay, turns, and final
context size. Invocation completion is measured at decode completion.

Resource metrics include busy time, utilization, maximum FCFS queue length,
and processed bytes. System metrics include throughput, latency percentiles,
offered/completed load, aggregate utilization, average active agents, average
tool-wait agents, and average KV-wait agents.

Utilization and bytes are clipped to the simulation horizon. If all agents
finish before `simulation_end_s`, the final agent event determines the measured
makespan.

## Example result

The checked-in memory-retention example uses seed 7, a 1000-second horizon,
6000 possible agents, and a 0.1-agent/s Poisson rate. At the horizon, 96 agents
had arrived and completed; remaining generated arrivals were scheduled later.

The verified run produced:

| Metric | Value |
|---|---:|
| Simulation horizon | 1000 s |
| Offered / completed agents | 96 / 96 |
| Agent throughput | 0.096 agents/s |
| Invocation throughput | 0.221 invocations/s |
| Mean latency | 165.04 ms |
| p50 / p95 / p99 latency | 144.87 / 259.42 / 290.70 ms |
| Prefill-device utilization | 0.3444% |
| Decode-device utilization | 0.5292% |
| Selected-memory utilization | 0.0091% |
| P-to-D KV bytes | 951,803,904 |
| Memory read/write bytes | 888,193,024 |
| Maximum compute queue | 1 |

Storage finishes within the example's sampled tool waits, so its average
KV-wait count is zero. A dedicated slower-storage test exercises `KV_WAIT` and
verifies that storage overlaps, rather than serializes with, tool wait.

## Verification

The Phase 1 suite contains 30 passing tests covering:

- event ordering, deterministic ties, horizons, and dependency enforcement;
- FCFS reservations, queueing, and utilization;
- deterministic arrivals and supported distributions;
- multi-turn state, tool waits, and context growth;
- exact prefill/decode and memory/link service equations;
- first-turn, retained-turn, and discard task graphs and byte formulas;
- memory/far-memory placement and required-resource/link validation;
- tool-wait/KV-store fork-and-join behavior and `KV_WAIT`;
- deterministic end-to-end results and queue buildup under load.

Run verification with:

```bash
.venv/bin/pytest
.venv/bin/python examples/run_simulation.py
```

## Deferred work

Phase 1 does not include experiment sweeps or persistent result export from PR4,
xPU/NMP-specific analytical models, MoE, optimized scheduling, detailed KV
capacity/eviction/migration, passive optical fabric modeling, trace input, or
ASTRA-Sim integration.
