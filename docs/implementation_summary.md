# PD/AFD simulator reconstruction summary

## Status

Milestones 1–7 of the breaking V1 reconstruction are implemented. The
production entry point accepts only the typed `SimulationConfig` loaded from
`hardware.yaml`, `workload.yaml`, and `mapping.yaml`. The Phase 1/1.5 task
graph, static P/D policy, byte ingress metrics, and monolithic decode model have
been removed; no compatibility fallback is reachable.

## Implemented architecture

The simulator composes one deterministic event calendar with typed controllers
for prefill, attention admission/residency, and whole-model decode steps.
Hardware owns analytical service models and queues; workload owns seeded session
and turn records; mapping owns logical model shape, role placement, and expert
ownership.

Every physical device has exactly one FCFS queue. Compute and endpoint work for
all capabilities declared on that device serialize through that queue.
All-to-all, KV P2P, and the KV pool also have independent global queues while
occupying their physical endpoints.

The generic FCFS queue maintains the same deterministic queue-length contract
with logarithmic historical lookup, which keeps long decode-step runs practical.

## Configuration and workload

V1 uses strict, independently parsed schema-version-1 files. Unknown fields,
duplicate YAML keys, unsupported discriminator values, legacy aliases, missing
capabilities, invalid expert ownership, and the legacy one-file call shape are
errors.

A generated session starts with warm context `C` already recoverable in the KV
pool. Every turn adds incremental prefill `P` and exact oracle output `O`:

```text
context_after_prefill = context_before_prefill + P
context_after_decode = context_after_prefill + O
```

The checked-in example uses eight two-turn sessions so it serves as a fast
end-to-end smoke run rather than a stress benchmark.

## Prefill and KV movement

Stateful prefill uses sticky round-robin placement and retains a per-session
prefill mirror; later turns load only the decode-produced delta since that
mirror. Stateless prefill uses round robin, loads full current context, and
keeps its transient copy until attention transfer finishes.

Each deterministic greedy batch executes sequentially:

```text
KV-pool load -> sum(P_i) / rate compute -> one-startup P2P batch
```

Warm context is never prefill compute. Attention transfer is full current
context on a miss/`active_only` restore and only the new delta on an LRU hit.

## Attention admission and residency

Admission reserves `context_at_decode_start + O_oracle` for the full episode.
Round robin scans feasible devices; least-active-context filters by feasibility
and breaks ties by device ID. FCFS capacity wait contributes to TTFT. Deferred
prefill transfers remain decode-ineligible across both immediate admission and
capacity retry until transfer completion.

Active and `DRAINING_DIRTY` KV are pinned. `lru_cache` retains clean inactive KV
with deterministic `(last_decode_time, session_id)` eviction; cached affinity
does not migrate. `active_only` removes the attention copy after the pool write.

## Decode and AFD

One synchronized V1 step is:

```text
local attention on every DP device
  -> global attention barrier
  -> dispatch all-to-all
  -> one atomic expert job per EP device
  -> global FFN barrier
  -> identical combine all-to-all
  -> one token per active session
```

Whole-model attention and FFN rates run once per step regardless of
`num_layers`. For active batch `B`:

```text
A = B * top_k
N_EP = len(decode_ffn.device_ids)
send_tokens = max(1, ceil(A / N_EP))
```

The communication floor never fabricates compute activations. Expert-ID-ordered
quotient/remainder loads sum exactly to `A`, then group by the configured
non-replicated ownership. Full logical dispatch and combine remain charged for
co-located mappings.

Each final token transitions attention KV to `DRAINING_DIRTY` and schedules one
modeled pool write containing unsynchronized `P + O`. Pool completion makes the
copy recoverable before retention/pruning. Tool wait overlaps that write, while
the next turn waits for both. Zero-output turns skip AFD but still perform their
P-only pool write.

## Metrics

The result exposes Section 18 token-unit records for:

- session and turn readiness, queueing, TTFT, inter-token, tool, and completion;
- attention admission, capacity, placement, cache, retry, eviction, and
  dirty/clean state;
- prefill batches, full/delta loads, compute and transfer tokens;
- dispatch/combine, expert loads, FFN device work, barriers, and imbalance;
- physical-device operation utilization and logical fabric/pool resources.

No byte metric or legacy invocation/task record is reported.

## Verified example

On 2026-08-17, the documented command completed with the checked-in example and
reported:

- 8 offered and 8 completed sessions;
- 16 completed turns;
- 1,359 decode steps and 2,615 emitted tokens;
- 4,819 prefill compute tokens;
- 43,708 KV-pool load, 7,434 KV-pool offload, and 47,164 KV-P2P tokens;
- 5,230 routed activations and 5,230 dispatch-plus-combine send tokens;
- 16 attention admissions: 8 cache misses and 8 cache hits;
- final simulation time 9.144063075685896 seconds.

These are deterministic outputs of the illustrative analytical configuration,
not calibrated hardware performance claims.

## Verification and deferred work

The Milestone 7 suite includes deterministic multi-turn execution, token
reconciliation, capacity retry and clean LRU eviction, Section 18 records,
legacy removal, every prior milestone unit test, and the documented example.

On 2026-08-17, the full suite completed with 125 passed tests.

V1 intentionally defers layer-wise execution, trace-derived routing, expert
replication, asynchronous AFD overlap, speculative decoding, active attention
migration, packet-level networks, cold-start prefill, and byte conversion.
