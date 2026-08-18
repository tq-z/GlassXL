# V1 output metrics reference

`Simulation.run()` returns a `SimulationResult` populated only from the typed
PD/AFD runtime. All compute, capacity, payload, and transfer quantities are
token-equivalent tokens. Times are simulated seconds unless a summary key ends
in `_ms`. No byte metrics are reported.

## Result structure

| Field | Population |
|---|---|
| `summary` | Aggregate metrics computed through the measured horizon. |
| `sessions` | One `SessionMetric` per generated session, including unfinished sessions. |
| `turns` | One `TurnMetric` per turn that started. |
| `attention_admissions` | One record per successful initial or retried capacity admission. |
| `attention_devices` | Device-state snapshots taken at admission/transfer, decode, and cleanup boundaries. |
| `prefill_batches` | One record per started deterministic prefill batch. |
| `prefill_sessions` | One record per session selected into a prefill batch. |
| `decode_steps` | One record per token-complete whole-model decode step within the horizon. |
| `resources` | One record per physical device, all-to-all fabric, P2P fabric, and KV pool. |
| `session_states` | Final mutable runtime states for reconciliation and advanced inspection. |

IDs and ordering are deterministic for a fixed configuration and seed. Turn and
decode-step indices are zero-based.

## Measurement horizon

When `simulation_end_s` is null, the horizon is the last processed event. When
a future event lies past a configured horizon, the event engine stops at that
horizon and leaves the event pending. Aggregate token counts and step records
include only operations completed by the horizon. Resource busy time is clipped
to the horizon; utilization is `busy_time_s / simulation_time_s`.

Empty populations produce zero aggregate means and percentiles. Percentiles use
linear interpolation between adjacent sorted samples.

## Summary

| Key | Definition |
|---|---|
| `simulation_time_s` | Measurement horizon. |
| `offered_sessions` | Sessions whose arrival is at or before the horizon. |
| `completed_sessions` | Sessions whose final episode pool write completed. |
| `completed_turns` | Turns whose pool write completed. |
| `decode_steps` | Whole-model steps whose token completion is within the horizon. |
| `emitted_tokens` | Sum of active sessions over completed decode steps. |
| `decode_token_throughput_per_s` | `emitted_tokens / simulation_time_s`. |
| `mean_session_latency_ms` | Mean `final pool completion - arrival` for completed sessions. |
| `p95_session_latency_ms` | Interpolated p95 of completed-session latency. |
| `mean_ttft_ms`, `p95_ttft_ms` | Turn-ready through first emitted-token completion; zero-output turns are excluded. |
| `mean_inter_token_ms` | Mean adjacent-token interval pooled across completed turns. |
| `prefill_compute_tokens` | Sum of incremental `P` passed to prefill compute. |
| `kv_pool_load_tokens` | Completed pool-load payloads. |
| `kv_pool_offload_tokens` | Completed episode `P + O` writes. |
| `kv_p2p_tokens` | Completed prefill-to-attention KV payloads. |
| `routed_activation_tokens` | Sum of `B * top_k` over completed steps. |
| `all_to_all_send_tokens` | Dispatch plus combine `send_tokens`; both collectives are counted. |
| `attention_admissions` | Successful capacity admissions. |
| `attention_retries` | Admissions that spent nonzero time in the FCFS capacity wait queue. |
| `attention_cache_hits`, `attention_cache_misses` | Admission decisions by attention residency. |
| `lru_evictions` | Clean inactive sessions evicted by successful admission plans. |

## Session and turn records

`SessionMetric` contains arrival/final-pool timestamps, completion latency,
completed and configured turn counts, initial/final context, generated tokens,
and the configured inter-turn tool wait actually used. `finish_time_s` and
`completion_latency_s` are null for unfinished sessions.

`TurnMetric` contains:

- arrival, turn-ready, and pool-finish timestamps;
- attention-capacity wait and ready-to-admission latency;
- prefill queue/service time;
- KV-pool load, P2P, and episode offload service time;
- TTFT and every adjacent inter-token latency;
- attention-ready through final-token decode latency;
- configured inter-turn tool wait and total ready-to-pool completion latency;
- warm context, incremental `P`, oracle/generated `O`, and pool-write tokens;
- attention device and cache-hit decision.

TTFT is exactly:

```text
first_token_completion - turn_ready_time
```

It includes prefill queue/compute, KV movement, capacity wait, device queueing,
attention, dispatch, FFN, and combine. A zero-output turn has no fabricated TTFT
or decode token.

## Attention metrics

`AttentionAdmissionMetric` records the selected device, oracle final-KV
commitment, transfer payload, capacity wait, cache result, retry flag, and clean
LRU victims. A retry is identified by positive capacity wait.

Each `AttentionDeviceMetric` snapshot reports active/resident session counts,
active context, oracle commitment, resident used tokens, configured capacity,
capacity utilization, and dirty/clean-inactive counts. Active and
`DRAINING_DIRTY` copies are pinned. A valid run never reports utilization above
one.

## Prefill metrics

`PrefillBatchMetric` reports device/state mode, batch timestamps, request count,
sum of contexts, incremental compute tokens, pool-load tokens, P2P transfer
tokens, queue latency, and compute service latency.

`PrefillSessionMetric` additionally distinguishes `full_context` from `delta`
loads and reports stateful-affinity hits. `compute_tokens` is always the turn's
incremental `P`; it is never warm context or decode output.

The analytical compute equation is:

```text
T_prefill = sum(P_i) / prefill_tokens_per_second
```

Pool load, compute, and P2P execute sequentially for a batch.

## Decode and FFN metrics

Each `DecodeStepMetric` reports:

- launch and token-completion timestamps and global active batch size;
- attention barrier timestamp and per-DP-device barrier wait;
- routed activations `A`, collective `send_tokens`, and dispatch/combine service;
- every `(expert_id, activation_tokens, device_id)` load;
- per-EP-device expert IDs, activation total, service time, and FFN barrier wait;
- device activation-load imbalance and emitted tokens.

The frozen equations are:

```text
A = B * top_k
send_tokens = max(1, ceil(A / N_EP))
sum(expert activation tokens) = A
```

Dispatch and combine use identical payload and service equations. Expert loads
use deterministic quotient/remainder by expert ID, then group by configured
ownership. `num_layers` does not repeat these records or costs in V1.

## Resource metrics

Every `ResourceMetric` contains resource ID/type, clipped busy time, accumulated
queue delay, utilization, maximum queue length, operation-specific busy time,
and transferred tokens.

Physical-device records break busy time down by `prefill`, `decode_attn`,
`decode_ffn`, `ffn_all_to_all`, `kv_p2p`, `kv_pool_load`, and
`kv_pool_offload` as applicable. This makes co-location contention visible.
Logical records cover the global all-to-all fabric, global P2P fabric, and shared
KV-pool queue. Their transferred-token totals use the corresponding logical
payload; physical endpoint records do not duplicate those token totals.
