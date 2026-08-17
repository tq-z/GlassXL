# Output Metrics Reference

This document describes the metrics returned by the current simulator, their
units, formulas, populations, and interpretation. It also calls out places
where aggregate values and detailed records intentionally measure different
things.

## Result structure

`Simulation.run()` returns a `SimulationResult` with seven metric collections:

| Field | Contents |
|---|---|
| `summary` | Aggregate system metrics as a dictionary |
| `agents` | One `AgentMetrics` record per completed agent |
| `invocations` | One `InvocationMetrics` record per invocation that became ready, including incomplete invocations |
| `resources` | One `ResourceMetrics` record per generic compute device, memory resource, and directed link |
| `prefill_batches` | One `PrefillBatchMetric` record per started prefill batch |
| `ingress_flows` | One `IngressFlowMetric` record per completed P-to-D ingress flow |
| `decode_iterations` | One `DecodeIterationMetric` record per started decode iteration |

The example runner prints only `summary` and `resources`. Detailed records are
available programmatically:

```python
from agentic_sim import Simulation, load_config

result = Simulation(load_config("configs/example.yaml")).run()

print(result.summary)
print(result.agents[0].to_dict())
print(result.invocations[0].to_dict())
print(result.prefill_batches[0])
print(result.ingress_flows[0])
print(result.decode_iterations[0])
```

## Units and conventions

| Suffix or name | Unit |
|---|---|
| `_s` | Seconds |
| `_ms` | Milliseconds |
| `_bytes` or `byte_count` | Bytes |
| `_bytes_per_s` | Bytes per second, not bits per second |
| `_per_s` | Events, agents, invocations, or tokens per second |
| `utilization` | Fraction from 0 to 1; multiply by 100 for percent |
| queue/agent/flow occupancy averages | Time-weighted average count |

Times are simulation timestamps, not wall-clock runtime. Turn indices are
zero-based. IDs are deterministic under a fixed random seed and configuration.

Percentiles use linear interpolation between adjacent sorted samples. When the
sample population is empty, the aggregate mean or percentile is reported as
zero.

## Measurement horizon

`simulation_time_s` is the event engine's final simulated time and is the
denominator for throughput, utilization, and time-weighted occupancy metrics.

- If the configured horizon stops a run while future events remain,
  `simulation_time_s` equals `simulation_end_s`.
- If the event queue empties first, it equals the timestamp of the last event,
  even when `simulation_end_s` is larger.
- Resource busy time is clipped at the measured horizon.
- Resource byte counts include only generic tasks that finish by the horizon.
- Completed-agent summaries exclude unfinished agents.
- Completed-invocation throughput excludes invocations without a decode finish.
- Invocation detail records still include invocations that became ready but did
  not finish.

Prefill batches and decode iterations are recorded when they start. Therefore,
their counts and mean planned durations can include an operation whose finish
event lies beyond a truncated horizon. `decode_tokens_emitted`, in contrast,
counts only iterations whose finish is within the horizon. Ingress flow records
are created on completion, so their count is a completed-flow count.

## Aggregate summary metrics

### Simulation population and throughput

| Metric | Definition |
|---|---|
| `simulation_time_s` | Final simulated timestamp used as the measurement horizon |
| `offered_agents` | Generated agents whose sampled arrival time is at or before the horizon |
| `completed_agents` | Agents that reached `FINISHED` by the horizon |
| `offered_load_per_s` | `offered_agents / simulation_time_s` |
| `completed_load_per_s` | `completed_agents / simulation_time_s` |
| `agent_throughput_per_s` | Same current formula as `completed_load_per_s` |
| `invocation_throughput_per_s` | Completed invocations divided by the horizon |

An agent may contain several invocations, so invocation throughput can be much
higher than agent throughput. Under a finite horizon, a large gap between
offered and completed load indicates backlog or work that arrived too late to
finish. It is not, by itself, a dropped-request count.

### Agent latency and accumulated time

| Metric | Definition and population |
|---|---|
| `mean_latency_ms` | Mean completed-agent latency, `(finish - arrival) × 1000` |
| `p50_latency_ms` | Median completed-agent latency |
| `p95_latency_ms` | 95th percentile completed-agent latency |
| `p99_latency_ms` | 99th percentile completed-agent latency |
| `mean_queueing_ms` | Mean accumulated queue delay per completed agent |
| `mean_tool_wait_ms` | Mean accumulated tool-wait duration per completed agent |
| `mean_agent_prefill_time_ms` | Mean completed-agent sum of per-turn prefill service durations |
| `mean_agent_decode_time_ms` | Mean completed-agent sum of per-turn decode durations |

Agent queueing time includes controller queues and generic resource queues. It
can include queue delays on background KV storage tasks. Agent active execution
time and queueing time are attribution totals, not a disjoint decomposition of
wall-clock latency: shared batch service is attributed to each participating
session, and background KV work overlaps tool wait. Consequently, adding
queueing, active execution, and tool time need not exactly reproduce latency.

### Time-weighted agent-state occupancy

| Metric | Definition |
|---|---|
| `average_active_agents` | Time average of agents in `READY` or `RUNNING` |
| `average_tool_wait_agents` | Time average of agents in `TOOL_WAIT` |
| `average_kv_wait_agents` | Time average of agents in `KV_WAIT` |

For a count `C(t)`, each average is:

```text
average_count = integral(C(t) dt) / simulation_time_s
```

`KV_WAIT` occurs only when a tool returns before tier-backed KV storage
finishes. It should be zero for `same_device` and `discard`; it can also be zero
for `memory` or `far_memory` when storage always finishes before tool return.

### Aggregate resource metrics

| Metric | Definition |
|---|---|
| `compute_utilization` | Arithmetic mean utilization across all configured compute devices |
| `memory_utilization` | Arithmetic mean utilization across all configured memory resources |
| `link_utilization` | Arithmetic mean utilization across all instantiated directed links |
| `max_compute_queue` | Maximum of generic compute FCFS queues, prefill waiting queue, and decode waiting queue |
| `max_memory_queue` | Maximum FCFS queue length among memory resources |
| `max_link_queue` | Maximum FCFS queue length among directed links |

Aggregate utilization includes idle configured resources. For example, an
unused far-memory tier lowers mean memory utilization. Use `result.resources`
when the utilization of a specific P device, D device, memory tier, or link is
needed.

The shared `hardware.kv_ingress` pool is not emitted as a generic
`ResourceMetrics` entry. Its activity is represented by the ingress summary and
flow records described below.

## Prefill metrics

| Metric | Definition |
|---|---|
| `prefill_batches` | Number of prefill batches started |
| `mean_prefill_batch_requests` | Mean number of invocations selected per batch |
| `mean_prefill_batch_tokens` | Mean sum of effective prefill tokens per batch |
| `mean_prefill_batch_service_ms` | Mean `(finish_time_s - start_time_s) × 1000` across batches |
| `mean_prefill_queueing_ms` | Mean invocation prefill queue delay across all invocation records |
| `max_prefill_queue` | Largest number waiting for prefill, excluding the running batch |
| `average_prefill_queue` | Time-weighted average number waiting for prefill |

For batch `b`, service time is:

```text
T_prefill(b) = fixed_latency
             + max(
                 longest_tokens / single_request_tokens_per_second,
                 total_tokens / max_system_tokens_per_second
               )
```

Effective prefill tokens depend on workload and KV retention:

```text
first turn = context_tokens, when configured
retained later turn = new input_tokens
discarded later turn = prior_context + new input_tokens
```

When `context_tokens` is omitted, legacy turn 1 uses `input_tokens`.

`mean_prefill_queueing_ms` is averaged over every invocation that became ready.
An invocation still blocked in restore at a truncated horizon retains its
default zero prefill-queue value, so this mean can be diluted by invocations
that never entered the prefill queue.

### Detailed `PrefillBatchMetric`

| Field | Meaning |
|---|---|
| `batch_id` | Deterministic batch sequence ID |
| `start_time_s` | Batch service start |
| `finish_time_s` | Scheduled common completion time |
| `request_count` | Number of invocations in the batch |
| `total_tokens` | Sum of effective prefill tokens in packed layout |
| `longest_tokens` | Maximum effective prefill length in the batch |
| `queue_length_at_start` | Number still waiting after selected members are removed |

All members of one batch share the same prefill completion timestamp.

## P-to-D KV-ingress metrics

| Metric | Definition |
|---|---|
| `ingress_flows` | Number of completed per-session P-to-D flows |
| `total_ingress_bytes` | Bytes across completed ingress flows |
| `mean_ingress_completion_ms` | Mean time from flow submission through completion, including fixed latency |
| `mean_ingress_wait_ms` | Mean time from submission until the flow joins active data transfer |
| `mean_ingress_achieved_bandwidth_bytes_per_s` | Mean per-flow `bytes / active_data_duration` |
| `average_aggregate_ingress_bandwidth_bytes_per_s` | Time average of allocated aggregate shared-pool bandwidth over the full horizon |
| `average_active_ingress_flows` | Time average of flows actively transferring data |
| `peak_concurrent_ingress_flows` | Maximum flows actively sharing bandwidth |

Each flow moves:

```text
byte_count = (prior_context + current_turn_input) * kv_bytes_per_token
```

In shared-pool mode with `N` active flows:

```text
per_flow_bandwidth = min(
    per_session_max_bandwidth_bytes_per_sec,
    max_system_bandwidth_bytes_per_sec / N
)
```

`average_aggregate_ingress_bandwidth_bytes_per_s` includes idle time in its
denominator. It can therefore be far below the bandwidth observed by active
flows. `mean_ingress_achieved_bandwidth_bytes_per_s` excludes fixed latency in
shared-pool mode.

Legacy direct-link mode has different instrumentation: `data_start_time_s` is
set equal to submission time, achieved bandwidth includes link queue and
service time, `mean_ingress_wait_ms` is zero, and aggregate shared-pool
bandwidth is reported as zero. Link utilization and bytes remain available in
`result.resources`. In this mode, active/peak ingress-flow counts represent
outstanding submitted flows from submission through completion, including link
queueing, rather than only flows actively transferring data.

### Detailed `IngressFlowMetric`

| Field | Meaning |
|---|---|
| `flow_id` | Deterministic flow sequence ID within the selected ingress mode |
| `session_id` | Invocation/session ID owning the flow |
| `submitted_time_s` | Time prefill completed and ingress was submitted |
| `data_start_time_s` | Time the flow entered active shared transfer |
| `finish_time_s` | Independent flow completion time |
| `byte_count` | Prompt-KV bytes transferred |
| `achieved_bandwidth_bytes_per_s` | `byte_count / (finish - data_start)` when active duration is positive |

## Decode metrics

| Metric | Definition |
|---|---|
| `decode_iterations` | Number of decode iterations started |
| `decode_tokens_emitted` | Tokens emitted by iterations completed within the horizon |
| `mean_decode_batch_size` | Mean active sessions per started iteration |
| `mean_decode_step_ms` | Mean decode-iteration duration |
| `decode_token_throughput_per_s` | `decode_tokens_emitted / simulation_time_s` |
| `mean_decode_session_completion_ms` | Mean time from first decode admission to invocation decode completion |
| `mean_decode_queueing_ms` | Mean decode waiting time across all invocation records |
| `max_decode_queue` | Maximum sessions waiting for an iteration-boundary admission |
| `average_decode_queue` | Time-weighted average waiting sessions |
| `average_decode_active_sessions` | Time-weighted average sessions in the active decode batch |

For active batch size `B`, one iteration emits one token per active session:

```text
T_decode_step(B) = fixed_latency
                 + max(
                     1 / single_session_tokens_per_second,
                     B / max_system_tokens_per_second
                   )
```

New sessions enter and completed sessions leave only at iteration boundaries.
There is one decode-finish event per iteration, not one per token/session.

Like prefill queueing, `mean_decode_queueing_ms` uses all invocation records.
Invocations that have not reached decode at a truncated horizon retain a zero
decode-queue value.

### TTFT and inter-token metrics

| Metric | Definition and population |
|---|---|
| `mean_ttft_ms` | Mean invocation TTFT |
| `p50_ttft_ms` | Median invocation TTFT |
| `p95_ttft_ms` | 95th percentile invocation TTFT |
| `p99_ttft_ms` | 99th percentile invocation TTFT |
| `mean_inter_token_ms` | Mean of all adjacent-token intervals pooled across completed invocations |
| `p95_inter_token_ms` | 95th percentile of pooled adjacent-token intervals |
| `p99_inter_token_ms` | 99th percentile of pooled adjacent-token intervals |

TTFT is measured from invocation readiness, not from prefill start:

```text
TTFT = first_token_time_s - ready_time_s
```

It therefore includes any restore, prefill queue/service, ingress, decode
queue, and first decode-step time. Tool wait before a later invocation is not
included because that invocation becomes ready only after its turn barrier is
released.

Inter-token intervals are:

```text
token_times_s[i] - token_times_s[i - 1]
```

They are pooled by token interval rather than averaged once per invocation, so
longer outputs contribute more samples. Completed zero-token or one-token
invocations contribute no inter-token samples. A completed zero-token
invocation also has no TTFT sample.

### Detailed `DecodeIterationMetric`

| Field | Meaning |
|---|---|
| `iteration_id` | Deterministic iteration sequence ID |
| `start_time_s` | Iteration service start |
| `finish_time_s` | Scheduled iteration finish |
| `active_sessions` | Active batch size after boundary admissions |
| `waiting_sessions` | Sessions still waiting after admissions |
| `emitted_tokens` | One token per active session; currently equal to `active_sessions` |

## Per-agent metrics

Only agents completed by the horizon appear in `result.agents`.

| Field | Meaning |
|---|---|
| `agent_id` | Deterministic agent ID |
| `arrival_time_s` | Sampled arrival timestamp |
| `start_time_s` | First hardware or prefill service start |
| `finish_time_s` | Final turn's decode completion boundary |
| `total_latency_s` | `finish_time_s - arrival_time_s` |
| `active_execution_time_s` | Accumulated attributed hardware/controller service time |
| `tool_wait_time_s` | Sum of actual non-final tool waits |
| `queueing_time_s` | Accumulated controller and generic-resource queue delay |
| `prefill_time_s` | Sum of all invocation prefill service durations for this agent |
| `decode_time_s` | Sum of all invocation decode durations for this agent |
| `num_turns` | Sampled turns for the agent |
| `final_context_tokens` | Initial prompt + later inputs + all generated outputs |

`start_time_s - arrival_time_s` can be nonzero because the first prefill or
restore may queue. Final-turn KV is not stored, so agent finish is final decode
completion.

`prefill_time_s` excludes time waiting in the prefill queue, KV restore, and
P-to-D ingress. Each invocation receives the entire wall-clock duration of its
prefill batch, so the sum across concurrently batched agents can exceed the
prefill device's elapsed busy time.

`decode_time_s` excludes time waiting for decode admission. For each invocation,
it spans from admission at an iteration boundary through final-token completion.

## Per-invocation metrics

Every invocation that became ready appears in `result.invocations`, even if it
did not finish before the horizon. Optional timestamps remain `None` when their
phase was not reached.

| Field | Meaning |
|---|---|
| `invocation_id` | Global deterministic invocation/session ID |
| `agent_id` | Owning agent |
| `turn_index` | Zero-based turn index |
| `ready_time_s` | Turn barrier released and invocation created |
| `finish_time_s` | Decode completion; `None` if incomplete |
| `queueing_time_s` | Accumulated non-background queue delay |
| `service_time_s` | Accumulated non-background service attribution |
| `prefill_queued_time_s` | Entry into prefill waiting queue |
| `prefill_start_time_s` | Start of selected prefill batch |
| `prefill_finish_time_s` | Common batch completion timestamp |
| `prefill_time_s` | Derived `prefill_finish - prefill_start`, or zero if prefill did not finish |
| `ingress_submit_time_s` | P-to-D ingress submission |
| `ingress_finish_time_s` | This session's independent ingress completion |
| `decode_queued_time_s` | Entry into decode waiting queue |
| `decode_start_time_s` | First iteration admission |
| `decode_time_s` | Derived `finish_time_s - decode_start_time_s`, or zero if decode did not finish |
| `first_token_time_s` | First decode-step completion for the session |
| `token_times_s` | Completion timestamp of every generated token |
| `prefill_queueing_time_s` | `prefill_start - prefill_queued` |
| `decode_queueing_time_s` | `decode_start - decode_queued` |
| `ttft_s` | Derived `first_token - ready`, or `None` |
| `inter_token_times_s` | Derived adjacent differences in `token_times_s` |

Invocation service time includes restore tasks, prefill batch service attributed
to the invocation, ingress service, and all decode steps in which it
participates. Background KV store/write is excluded from invocation queue and
service metrics, although it remains in agent and resource accounting.

## Generic resource metrics

`result.resources` uses these IDs:

```text
compute:<device_id>
memory:<memory_id>
link:<src_id>-><dst_id>
```

| Field | Meaning |
|---|---|
| `resource_id` | String identifier shown above |
| `resource_type` | `compute`, `memory`, or `link` |
| `busy_time_s` | Sum of reserved service intersecting the horizon |
| `utilization` | `busy_time_s / simulation_time_s` |
| `max_queue_length` | Maximum queue observed by that generic FCFS resource |
| `bytes_processed` | Completed bytes for memory/link tasks; zero for compute |

Prefill and decode controllers reserve their selected compute resources, so
their busy time appears here. Their waiting queues live in controllers, which
is why a device's own `max_queue_length` can be zero while
`max_prefill_queue`, `max_decode_queue`, or `max_compute_queue` is nonzero.

Memory and directed links are serial FCFS resources. Restore currently executes
as a memory read followed by a memory-to-P link transfer, so its two service
times add. Store similarly executes D-to-tier transfer followed by memory
write. These bytes are counted separately on both resources.

Retention-mode expectations:

| Mode | Memory/link restore-store bytes | Prefill work | `KV_WAIT` possible |
|---|---:|---|---|
| `same_device` | None | New input only | No |
| `memory` | Selected memory tier and links | New input only | Yes |
| `far_memory` | Selected far-memory tier and links | New input only | Yes |
| `discard` | None | Prior context + new input | No |

All modes still perform P-to-D prompt-KV ingress. For `same_device`, any
coherence of decode-produced KV back to P is assumed zero-cost and device KV
capacity is not measured.

## Reading bottlenecks from the output

### Prefill-bound workload

Typical signals:

- P-device utilization approaches 1;
- `average_prefill_queue` and `max_prefill_queue` grow;
- TTFT rises with prefill queueing;
- completed load falls below offered load;
- larger mean batch size may improve throughput without eliminating backlog.

### Decode-bound workload

Typical signals:

- D-device utilization approaches 1;
- decode queue and active occupancy rise;
- TTFT includes a growing decode admission delay;
- inter-token time rises when the aggregate system-rate term dominates;
- mean decode batch size approaches `max_active_sessions` under heavy load.

### Ingress-bound workload

Typical signals:

- peak and average active flows rise;
- achieved per-flow bandwidth falls as the system cap is shared;
- ingress completion time grows;
- P and D utilization can remain below saturation while sessions accumulate
  between them.

### KV-tier-bound workload

Typical signals for `memory` or `far_memory`:

- selected memory and restore/store link utilization increase;
- memory or link FCFS queues grow;
- `average_kv_wait_agents` becomes nonzero when storage outlasts tool wait;
- later-turn TTFT includes larger restore delay.

`same_device` removes these modeled tier costs; it should not be interpreted as
a capacity-feasibility result because device KV capacity is currently
descriptive/absent.

## Important interpretation limits

- Metrics are deterministic model outputs, not hardware measurements.
- Compute rates are coarse token-rate envelopes, not per-layer execution.
- Shared ingress is a fluid equal-share pool, not packet or fabric simulation.
- Restore/store resources are serial and not pipelined.
- Aggregate utilization is an arithmetic mean across configured resources.
- Queue averages are time-weighted; queue-delay means are record-weighted.
- Detailed arrays can be large because decode has one record per iteration and
  invocations retain every token completion timestamp.
- No capacity admission, KV eviction, migration optimization, MoE, xPU/NMP,
  detailed fabric, trace input, or ASTRA-Sim behavior is represented yet.
