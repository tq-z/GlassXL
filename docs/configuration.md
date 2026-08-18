# V1 configuration reference

V1 replaces the legacy single-file configuration with three required,
independently parsed YAML files:

- `hardware.yaml` describes physical resources and calibrated service models.
- `workload.yaml` describes run controls and session generation.
- `mapping.yaml` describes logical model shape, placement, and policies.

All files are strict: unknown fields, legacy aliases, duplicate YAML keys, and
unsupported discriminator values are errors. Internal time is seconds. YAML
field names identify external time units. Compute, transfer, payload, KV
capacity, and traffic values use token-equivalent units.

Load all three files with:

```python
from agentic_sim.config import load_config

config = load_config(
    "configs/example/hardware.yaml",
    "configs/example/workload.yaml",
    "configs/example/mapping.yaml",
)
```

The confirmed three-path command-line shape is:

```bash
python examples/run_simulation.py \
  --hardware configs/example/hardware.yaml \
  --workload configs/example/workload.yaml \
  --mapping configs/example/mapping.yaml
```

The result is one recursively immutable `SimulationConfig`: records are frozen
dataclasses, and repeated values are tuples. The independent
`load_hardware_config`, `load_workload_config`, and `load_mapping_config`
functions parse one file without resolving cross-file references.
`load_config` calls all three and then performs cross-file validation. Passing
one legacy combined file is rejected; there is no fallback or normalization.

## hardware.yaml

### Root and devices

| YAML path | Type | Req. | Default | Unit | Allowed / validation | Equation or runtime effect and relationships |
|---|---|---:|---|---|---|---|
| `schema_version` | integer | yes | — | — | exactly `1` | Selects this breaking hardware schema. |
| `devices` | list | yes | — | — | non-empty; device IDs unique | Defines physical resources referenced by all mapping role groups. |
| `devices[].id` | integer | yes | — | — | >= 0; unique | Physical identity. One device will own one queue shared by all of its capabilities. |
| `devices[].capabilities` | mapping | yes | — | — | non-empty; keys only `prefill`, `decode_attn`, `decode_ffn` | Declares which roles the physical device can perform. A device may declare multiple capabilities. |

A declared capability need not be selected by `mapping.yaml`. A mapped role
must, however, reference a device declaring that capability. V1 permits
attention and FFN on the same physical device. Per the frozen V1 prefill
capacity decision, mapped prefill and mapped decode-attention device sets must
not overlap.

### Prefill capability

| YAML path | Type | Req. | Default | Unit | Allowed / validation | Equation or runtime effect and relationships |
|---|---|---:|---|---|---|---|
| `devices[].capabilities.prefill` | mapping | conditional | — | — | required to map the device for prefill | Selects the prefill service model. |
| `...prefill.model` | string | yes | — | — | `aggregate_throughput` | V1 uses the aggregate model interface. |
| `...prefill.tokens_per_second` | number | yes | — | tokens/s | finite and > 0 | For batch B, `T = sum(P_i) / rate`. Context and batch metadata remain model inputs. |

### Decode-attention capability

| YAML path | Type | Req. | Default | Unit | Allowed / validation | Equation or runtime effect and relationships |
|---|---|---:|---|---|---|---|
| `devices[].capabilities.decode_attn` | mapping | conditional | — | — | required to map the device for attention | Selects the attention service model and resident-KV budget. |
| `...decode_attn.model` | string | yes | — | — | `linear_context` | V1 uses the linear context interface. |
| `...decode_attn.context_tokens_per_second` | number | yes | — | context tokens/s | finite and > 0 | For local batch B, `T = sum(C_i / rate)`. |
| `...decode_attn.kv_capacity_tokens` | integer | yes | — | tokens | >= 0 | Bounds active, draining, and cached attention KV under later admission logic. |

### Decode-FFN capability

| YAML path | Type | Req. | Default | Unit | Allowed / validation | Equation or runtime effect and relationships |
|---|---|---:|---|---|---|---|
| `devices[].capabilities.decode_ffn` | mapping | conditional | — | — | required to map the device for FFN | Selects the expert-compute service model. |
| `...decode_ffn.model` | string | yes | — | — | `linear_expert_batch`, `batch_latency_curve` | Selects a scalar rate or batch-sensitive per-device expert-work model. |
| `...decode_ffn.tokens_per_second` | number | conditional | — | routed activation tokens/s | required and finite > 0 only for `linear_expert_batch` | For resident expert batches, `T = sum(B_e) / rate`. |
| `...decode_ffn.points` | list | conditional | — | — | required only for `batch_latency_curve`; at least two points | Piecewise-linear latency as a function of total routed activations on this EP device. |
| `...decode_ffn.points[].activation_tokens` | integer | yes | — | routed activation tokens | first value 0; strictly increasing and >= 0 | `top_k` has already been applied by the runtime. |
| `...decode_ffn.points[].latency_us` | number | yes | — | microseconds | first value 0; finite, non-negative, and nondecreasing; final segment has positive slope | Exact at breakpoints, linearly interpolated between points, and extrapolated using the final segment. |

The two FFN variants are mutually strict: a scalar model rejects `points`, and
a curve model rejects `tokens_per_second`. Each FFN device still submits one
atomic whole-model job and participates in the same global FFN barrier.

### KV pool

| YAML path | Type | Req. | Default | Unit | Allowed / validation | Equation or runtime effect and relationships |
|---|---|---:|---|---|---|---|
| `kv_pool` | mapping | yes | — | — | strict fields below | Declares the recoverable KV backing resource. |
| `kv_pool.capacity_tokens` | null | no | `null` | tokens | only `null` in V1; finite values rejected | Makes the V1 pool unbounded. |
| `kv_pool.load_tokens_per_second` | number | yes | — | tokens/s | finite and > 0; equal to offload rate | Pool read service component is `tokens / rate`. |
| `kv_pool.offload_tokens_per_second` | number | yes | — | tokens/s | finite and > 0; equal to load rate | Pool write service component is `tokens / rate`. V1 writes unsynchronized `P + O`. |
| `kv_pool.start_latency_us` | number | yes | — | microseconds | finite and >= 0 | Each pool batch operation adds `start_latency_us / 1e6` seconds. |

The two direction-named fields remain explicit in the schema, but V1 validates
that they are equal because the frozen model uses one symmetric pool rate.
Later milestones give pool operations their confirmed resource and
physical-device queue behavior.

### Interconnects

| YAML path | Type | Req. | Default | Unit | Allowed / validation | Equation or runtime effect and relationships |
|---|---|---:|---|---|---|---|
| `interconnects` | mapping | yes | — | — | exactly both blocks below | Both global logical fabrics are required even when role groups overlap. |
| `interconnects.ffn_all_to_all` | mapping | yes | — | — | strict fields below | One globally modeled all-to-all resource used once for dispatch and once for combine per decode step. |
| `...ffn_all_to_all.model` | string | yes | — | — | `startup_min_package` | Selects the V1 collective equation. |
| `...ffn_all_to_all.start_latency_us` | number | yes | — | microseconds | finite and >= 0 | `START_LAT = value / 1e6` seconds. |
| `...ffn_all_to_all.min_package_tokens` | integer | yes | — | tokens | >= 0 | Payload floor in `max(send_tokens, floor)`; it is not a cap. |
| `...ffn_all_to_all.throughput_tokens_per_second` | number | yes | — | tokens/s | finite and > 0 | One collective costs `START_LAT + max(send_tokens, floor) / rate`. Dispatch and combine each pay it. |
| `interconnects.kv_p2p` | mapping | yes | — | — | strict fields below | One global logical KV-migration fabric. |
| `...kv_p2p.model` | string | yes | — | — | `startup_throughput` | Selects the V1 P2P equation. |
| `...kv_p2p.start_latency_us` | number | yes | — | microseconds | finite and >= 0 | Charged once per KV transfer batch. |
| `...kv_p2p.throughput_tokens_per_second` | number | yes | — | tokens/s | finite and > 0 | `T = start_latency_us / 1e6 + kv_tokens / rate`; there is no minimum-package floor. |

Byte-rate, byte-capacity, flat phase-rate, memory-tier, and legacy
`kv_ingress` fields are not accepted.

## workload.yaml

### Root and source

| YAML path | Type | Req. | Default | Unit | Allowed / validation | Equation or runtime effect and relationships |
|---|---|---:|---|---|---|---|
| `schema_version` | integer | yes | — | — | exactly `1` | Selects this breaking workload schema. |
| `seed` | integer | yes | — | — | any integer; booleans rejected | Sole seed for deterministic synthetic workload draws. |
| `simulation_end_s` | number or null | no | `null` | seconds | finite and >= 0 when present | Stops event processing at the horizon; null runs until no events remain. |
| `source` | mapping | yes | — | — | strict fields below | Workload data only; it cannot contain device IDs or hardware rates. |
| `source.type` | string | yes | — | — | `synthetic` | Selects the seeded synthetic source. |
| `source.num_sessions` | integer | yes | — | sessions | >= 0 | Number of generated long-lived sessions. |
| `source.arrival` | mapping | yes | — | — | strict fields below | Defines initial session arrival timestamps. |
| `source.arrival.type` | string | yes | — | — | `poisson` | Cumulative inter-arrivals are sampled from an exponential process. |
| `source.arrival.rate_per_second` | number | yes | — | sessions/s | finite and > 0 | Mean inter-arrival time is `1 / rate`. |

### Session distributions

| YAML path | Type | Req. | Default | Unit | Allowed / validation | Runtime semantics |
|---|---|---:|---|---|---|---|
| `source.turns` | distribution | yes | — | turns | integer distribution; every possible draw >= 1 | Number of decode episodes for a session. |
| `source.context_tokens` | distribution | yes | — | tokens | integer distribution; draws >= 0 | Sampled once. This is warm context already in the KV pool at arrival and incurs no initial prefill compute. |
| `source.prefill_tokens` | distribution | yes | — | tokens | integer distribution; draws >= 0 | Incremental P for each modeled turn, including the first observed turn. |
| `source.output_tokens` | distribution | yes | — | tokens | integer distribution; draws >= 0 | Exact oracle O for attention capacity admission and decode completion. |
| `source.tool_wait_ms` | distribution | yes | — | milliseconds | numeric distribution; draws >= 0 | Delay between episodes. The next turn joins tool completion with the episode pool write. |

Integer quantities implement `constant`, `bounded_integer`, and `discrete`.
Tool wait additionally implements `uniform`, `exponential`, and
`lognormal`. Aliases such as `bounded_int` and `categorical` are rejected.

### Distribution variants

| YAML path relative to a distribution | Type | Req. | Default | Allowed / validation | Sampling effect |
|---|---|---:|---|---|---|
| `type` | string | yes | — | one implemented value listed below | Selects exactly one strict variant. |
| `type: constant` + `value` | integer or number | yes | — | finite and at least the field-specific minimum; integer for turns/tokens | Always returns `value`. |
| `type: bounded_integer` + `min`, `max` | integers | yes | — | field minimum <= `min` <= `max` | Uniform inclusive integer draw. |
| `type: discrete` + `values` | list | yes | — | non-empty; finite; field minimum; integer for turns/tokens | Selects one listed value. |
| `type: discrete` + `probabilities` | list of numbers | yes | — | same length as values; each finite and >= 0; sum equals 1 within numeric tolerance | Selection weights in list order. |
| `type: uniform` + `min`, `max` | numbers | yes | — | finite; 0 <= `min` <= `max`; tool wait only | Continuous uniform draw. |
| `type: exponential` + `rate` | number | yes | — | finite and > 0; tool wait only | Exponential draw with mean `1 / rate`. |
| `type: lognormal` + `mean` | number | yes | — | finite; tool wait only | Mean parameter of the underlying normal distribution. |
| `type: lognormal` + `sigma` | number | yes | — | finite and >= 0 | Standard deviation parameter of the underlying normal distribution. |
| `type: lognormal` + `min` | number | no | no lower clipping | finite and >= 0; <= max if both present | Clips the sampled tool wait from below. |
| `type: lognormal` + `max` | number | no | no upper clipping | finite and >= 0; >= min if both present | Clips the sampled tool wait from above. |

Each variant rejects fields belonging to another variant. All validation occurs
while loading, before any random generator is constructed.

## mapping.yaml

### Root and model

| YAML path | Type | Req. | Default | Unit | Allowed / validation | Equation or runtime effect and relationships |
|---|---|---:|---|---|---|---|
| `schema_version` | integer | yes | — | — | exactly `1` | Selects this breaking mapping schema. |
| `granularity` | string | yes | — | — | `decode_step` | V1 invokes each whole-model attention, communication, and FFN stage once per decode step. |
| `model` | mapping | yes | — | — | strict fields below | Owns logical model shape rather than hardware or workload policy. |
| `model.num_layers` | integer | yes | — | layers | >= 1 | Model identity and future layer-wise seam; does not multiply V1 latency or traffic. |
| `model.num_experts` | integer | yes | — | experts | >= 1 | Defines valid expert IDs `[0, num_experts)`. |
| `model.top_k` | integer | yes | — | experts/token | >= 1 and <= `num_experts` | Routed activation count is `A = active_sessions * top_k`. |

### Prefill mapping

| YAML path | Type | Req. | Default | Allowed / validation | Runtime effect and relationships |
|---|---|---:|---|---|---|
| `prefill.device_ids` | list of integers | yes | — | non-empty, unique, IDs >= 0; each exists and has `prefill` | Eligible prefill devices. Must not overlap mapped attention devices in V1. |
| `prefill.state` | string | yes | — | `stateful`, `stateless` | Stateful retains a prefill mirror; stateless reloads full existing context. |
| `prefill.placement` | string | yes | — | `sticky_round_robin`, `round_robin` | Stateful requires `sticky_round_robin`; stateless requires `round_robin`. |

### Attention mapping

| YAML path | Type | Req. | Default | Allowed / validation | Runtime effect and relationships |
|---|---|---:|---|---|---|
| `decode_attn.device_ids` | list of integers | yes | — | non-empty, unique, IDs >= 0; each exists and has `decode_attn` | Defines attention DP width `N_DP`. A session stays on one selected device for an episode. |
| `decode_attn.placement` | string | yes | — | `round_robin`, `least_active_context` | Selects among devices feasible under oracle commitment; least-load ties use device ID. |
| `decode_attn.residency` | string | yes | — | `active_only`, `lru_cache` | Controls cross-turn attention KV retention. Both values compose with both placement policies. |
| `decode_attn.capacity_reservation` | mapping | yes | — | strict field below | Configures admission commitment. |
| `decode_attn.capacity_reservation.model` | string | yes | — | `oracle_output` | Reserves `C_decode_start + O_oracle` for the episode; context growth offsets remaining output. |

### FFN mapping

| YAML path | Type | Req. | Default | Allowed / validation | Runtime effect and relationships |
|---|---|---:|---|---|---|
| `decode_ffn.device_ids` | list of integers | yes | — | non-empty, unique, IDs >= 0; each exists and has `decode_ffn` | Defines FFN EP width `N_EP`; it is not `num_experts`. |
| `decode_ffn.experts_by_device` | mapping of integer device ID to list | yes | — | exactly one entry for every listed FFN device; no extra devices | One placement shared by every layer in V1. |
| `decode_ffn.experts_by_device.<device>` | list of integers | yes | — | non-empty; IDs in `[0, num_experts)`; every expert covered exactly once | Assigns several experts to one EP rank. Replication and uncovered experts are rejected. |

## Decode-step execution semantics

The mapping fields above drive one synchronized whole-model step;
`model.num_layers` does not repeat any stage. At launch time, every mapped
attention DP device submits one local batch, including an empty batch when it
has no ready sessions. Local attention uses `sum(C_i) / rate`, and all mapped DP
devices meet at a global attention barrier. For global active batch `B`:

```text
A = B * model.top_k
N_EP = len(decode_ffn.device_ids)
send_tokens = max(1, ceil(A / N_EP))
```

Dispatch and combine each submit the same full logical all-to-all over the union
of mapped attention and FFN device IDs. This cost remains present when the two
roles are co-located. The communication floor does not create compute work:
`A` is divided across `model.num_experts` by expert-ID-ordered quotient and
remainder, so expert activation counts sum exactly to `A`. Loads are grouped by
`decode_ffn.experts_by_device`, and every FFN device submits one atomic job
before the global FFN barrier. Combine completes before each ready session emits
exactly one token and increments its context.

A session keeps its selected attention-device affinity for the episode. After
the oracle output completes, the runtime schedules one KV-pool offload containing
the unsynchronized `P + O` tokens. The copy remains `DRAINING_DIRTY` and consumes
attention capacity until that modeled reservation completes. Completion makes
the pool snapshot recoverable, then `lru_cache` retains a clean inactive copy or
`active_only` removes it. A zero-output episode skips decode/AFD work and still
performs its required P-only pool write.

The same physical ID may be mapped for attention and FFN when the hardware
device declares both capabilities. All compute, collective endpoint, KV-transfer
endpoint, and pool endpoint work for that physical ID shares one FCFS queue.

## Error behavior

`ConfigurationError` messages identify the failing YAML path, for example:

```text
mapping.decode_attn.device_ids[0]: hardware device 7 does not provide decode_attn
mapping.decode_ffn.experts_by_device[4]: expert 2 is already owned by device 3;
replication is not supported in V1
hardware.interconnects: missing required field(s): kv_p2p
```

Independent loaders report file-local structural and numerical errors.
Cross-file reference, capability, placement, and expert-coverage errors are
reported only by `load_config` or an explicit
`validate_simulation_config(config)` call.
