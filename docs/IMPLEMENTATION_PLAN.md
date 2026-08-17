# Implementation Plan: Breaking PD/AFD Simulator Reconstruction

> Status: design target; implementation has not started.
>
> This document replaces the historical Phase 1/1.5 and PR-by-PR plan. The
> existing source still implements that legacy design. A later implementation
> must treat this document as the new target and may delete incompatible code.
>
> Labels used below:
>
> - CONFIRMED: explicitly requested by the user.
> - PROPOSED: a concrete default needed for an implementable design, but not yet
>   explicitly confirmed.
> - OPEN: the implementation must not guess; resolve the numbered question in
>   Section 17 first.

## 1. Objective

Reconstruct the simulator around two composable forms of inference
disaggregation:

- prefill/decode disaggregation (PD);
- decode attention/FFN disaggregation (AFD).

The first implementation uses coarse analytical models expressed in token
units. Its purpose is to study scheduling, queueing, placement, KV residency,
and stage interaction. More detailed compute curves, trace-derived expert
routing, and layer-wise execution will replace individual model or orchestration
interfaces later.

The three logical compute capabilities are:

- prefill;
- decode_attn;
- decode_ffn.

A physical device may provide any subset of these capabilities. A single
device may therefore execute prefill and decode, attention and FFN, or all three
functions.

## 2. Breaking-design boundary

CONFIRMED:

- No legacy YAML or runtime compatibility is required.
- Delete compatibility aliases, normalizers, and tests whose only purpose is to
  preserve the Phase 1/1.5 behavior.
- The legacy monolithic phase fields and lumped decode equation are not part
  of the target. Simple per-capability token/s latency models are the intended
  V1 implementation.
- The old single-file configuration is not part of the target.
- The old byte-based KV ingress, memory/far-memory/discard retention choices,
  and mandatory distinct prefill/decode device IDs are not authoritative.
- Existing test results and example metrics validate only the legacy code.

Reusable infrastructure may be retained when its behavior still matches the
new design, especially deterministic event ordering and the generic FCFS
resource primitive.

## 3. Terminology

Physical device
: A schedulable hardware resource identified by device_id.

Capability
: One of prefill, decode_attn, or decode_ffn implemented by a physical
  device.

Global event calendar
: The deterministic timestamp/sequence-ID priority heap that advances
  simulation time.

Device queue
: The single service queue owned by a physical device. All capabilities on that
  device contend for it.

Session
: A long-lived request/agent state carrying context across one or more decode
  episodes.

Decode episode
: Consecutive decode steps beginning when a session is admitted to an attention
  batch and ending when its requested output for that turn is complete.

Active session
: A session currently admitted to decode and eligible to emit a token.

Inactive session
: A session between decode episodes, for example during tool wait.

Resident session
: A session whose KV is currently stored on an attention device. A resident
  session may be active or inactive.

Context tokens, C
: The complete context length already established at the current timestamp.

Prefill tokens, P
: Newly arrived tokens that require prefill computation.

Output tokens, O
: Tokens generated during a decode episode.

Expert batch, B_e
: The number of routed token activations assigned to one expert for a modeled
  FFN stage. Its decode GEMM shape is conceptually 1 x B_e.

DP
: Data parallel attention replicas. A session belongs to one attention replica
  during a decode episode.

EP
: Expert parallel FFN ranks/devices. N_EP is the number of FFN EP
  ranks participating in the global FFN step. It is distinct from the total
  number of experts; one EP device may serve multiple experts.

KV pool
: The backing resource that stores recoverable session KV outside attention
  device residency.

## 4. Non-negotiable invariants

1. Each physical device owns exactly one execution queue.
2. Co-located capabilities serialize through that shared device queue.
3. Workload data does not contain device IDs or hardware rates.
4. Hardware models do not contain workload-generation policy.
5. Placement and expert ownership live in the mapping configuration.
6. All coarse compute capacities, transfer throughputs, payloads, and attention
   KV capacities use token-equivalent units.
7. Internal time uses seconds. YAML field names state their external unit.
8. Initial context is warm state at timestamp t and incurs no initial prefill
   compute.
9. A session stays on one attention device across all decode steps in a decode
   episode.
10. Active attention KV is pinned and cannot be evicted.
11. V1 executes decode step by step, while its interfaces carry enough context
    to support later layer-wise execution.

CONFIRMED:

- Keep one deterministic global event calendar.
- Interpret one queue per device as one physical service queue shared by every
  capability on that device, not as an independent event calendar.

PROPOSED:

- Use FCFS discipline for every physical device service queue.
- Give the KV pool and each logical interconnect their own independently
  contended resource queue.
- Infer PD versus AFD structurally from mapping and co-location rather than
  adding a separate mode flag.

## 5. Target architecture

~~~text
hardware.yaml + workload.yaml + mapping.yaml
                    |
                    v
          typed configuration loader
                    |
                    v
         cross-file mapping validator
                    |
                    v
        seeded workload/session source
                    |
                    v
              AgenticRuntime
       +------------+-------------+
       |            |             |
       v            v             v
 PrefillController  AttentionAdmission  DecodeStepCoordinator
       |            |             |
       +------------+-------------+
                    |
                    v
       device / interconnect / KV-pool resources
                    |
                    v
        deterministic global event calendar
                    |
                    v
               MetricsCollector
~~~

The runtime owns session lifecycle and stage dependencies. Mapping components
select eligible resources. Hardware components calculate service time and
reserve resources. The event engine remains unaware of PD, AFD, KV, experts,
and agent turns.

## 6. Configuration contract

PROPOSED CLI shape for the confirmed three-file split:

~~~text
--hardware configs/example/hardware.yaml
--workload configs/example/workload.yaml
--mapping configs/example/mapping.yaml
~~~

The loader returns one typed aggregate SimulationConfig. There is no
single-file fallback and no legacy field normalization.

### 6.1 hardware.yaml

Owns only physical resources and calibrated service models:

~~~yaml
schema_version: 1

devices:
  - id: 0
    capabilities:
      prefill:
        model: aggregate_throughput
        tokens_per_second: 40000
  - id: 1
    capabilities:
      decode_attn:
        model: linear_context
        context_tokens_per_second: 2000000
        kv_capacity_tokens: 262144
  - id: 2
    capabilities:
      decode_attn:
        model: linear_context
        context_tokens_per_second: 2000000
        kv_capacity_tokens: 262144
  - id: 3
    capabilities:
      decode_ffn:
        model: linear_expert_batch
        tokens_per_second: 100000
  - id: 4
    capabilities:
      decode_ffn:
        model: linear_expert_batch
        tokens_per_second: 100000

kv_pool:
  capacity_tokens: null
  load_tokens_per_second: 1000000
  offload_tokens_per_second: 1000000
  start_latency_us: 5

interconnects:
  ffn_all_to_all:
    model: startup_min_package
    start_latency_us: 10
    min_package_tokens: 1
    throughput_tokens_per_second: 1000000
  kv_p2p:
    model: startup_throughput
    start_latency_us: 5
    throughput_tokens_per_second: 1000000
~~~

The three snippets in this section form one cross-file-valid proposed example;
their values are illustrative, not calibrated defaults. PROPOSED: V1 treats a
null kv_pool.capacity_tokens value as an unbounded pool and rejects a finite
value until pool-full behavior is designed. decode_attn.kv_capacity_tokens
accounts for attention residency only; whether stateful-prefill KV shares a
physical memory budget is OPEN.

PROPOSED: V1 defines one global queued ffn_all_to_all resource and one global
queued kv_p2p resource. Named fabrics, routes, and endpoint-specific links are
deferred rather than referenced implicitly from mapping.yaml.

Hardware validation must reject:

- duplicate device IDs;
- an empty capability map;
- unknown capability or model names;
- nonpositive rates;
- negative latencies or capacities;
- attention placement on a device without decode_attn;
- expert placement on a device without decode_ffn;
- prefill placement on a device without prefill;
- legacy flat phase-rate fields;
- byte-unit transfer fields in the new token-unit schema.

OPEN:

- Whether KV-pool synchronization and direct PD P2P migration are distinct
  additive resources or two names for the same transfer path.
- Whether pool load and offload use separate rates, as shown, or one shared
  throughput parameter.

### 6.2 workload.yaml

Owns only run controls and workload/session data:

~~~yaml
schema_version: 1
seed: 7
simulation_end_s: 10000

source:
  type: synthetic
  num_sessions: 1000
  arrival:
    type: poisson
    rate_per_second: 1.0
  turns:
    type: constant
    value: 2
  context_tokens:
    type: bounded_integer
    min: 4096
    max: 8192
  prefill_tokens:
    type: bounded_integer
    min: 128
    max: 512
  output_tokens:
    type: bounded_integer
    min: 64
    max: 256
  tool_wait_ms:
    type: constant
    value: 100
~~~

Semantics:

- context_tokens is sampled once and is already prefilled at the session's
  initial timestamp;
- prefill_tokens is incremental input for later modeled prefills;
- output_tokens is the exact oracle generated length for a decode episode and
  is used by capacity admission;
- the workload file does not select prefill, attention, or FFN devices;
- a future trace source must produce the same typed session/turn records.

The distribution loader must validate all variants strictly, including finite
values, nonnegative token draws, probability sums, bounds, and deterministic
seed use.

### 6.3 mapping.yaml

Owns logical model shape, placement, and runtime policies:

~~~yaml
schema_version: 1
granularity: decode_step

model:
  num_layers: 32
  num_experts: 8
  top_k: 2

prefill:
  device_ids: [0]
  state: stateful
  placement: sticky_round_robin

decode_attn:
  device_ids: [1, 2]
  placement: least_active_context
  residency: lru_cache
  capacity_reservation:
    model: oracle_output

decode_ffn:
  device_ids: [3, 4]
  experts_by_device:
    3: [0, 1, 2, 3]
    4: [4, 5, 6, 7]
~~~

Required mapping enums:

- granularity: decode_step in V1; layer is reserved for a later implementation;
- prefill.state: stateful or stateless;
- decode_attn.placement: round_robin or least_active_context;
- decode_attn.residency: active_only or lru_cache;
- decode_attn.capacity_reservation.model: oracle_output.

In V1, N_DP is the number of decode_attn.device_ids and N_EP is the number of
decode_ffn.device_ids. model.num_experts is independent of N_EP, and
experts_by_device may assign several experts to one EP device. model.num_layers
is retained for model identity and the future layer-wise scheduler; it does not
scale V1 latency or traffic.

PROPOSED prefill placement contract:

- stateful uses sticky_round_robin so a retained mirror has stable ownership;
- stateless uses round_robin because it retains no local session state;
- selection, cursor advancement, state loss, and remapping still require
  confirmation before these values become part of the strict schema.

Cross-file validation must guarantee:

- every referenced device exists and provides the assigned capability;
- the same physical ID may appear in multiple role groups;
- at least one prefill and one attention device exist;
- every expert ID is within range;
- all experts have an owner;
- V1 uses one expert placement for every layer;
- duplicate expert ownership follows the expert-replication choice frozen in
  Milestone 0;
- old policy.prefill_device and policy.decode_device fields are rejected.

OPEN:

- num_layers, num_experts, and top_k are required model concepts; confirm that
  mapping.yaml is their public configuration owner.
- Expert replication is proposed out of scope, but this has not been confirmed.

### 6.4 Examples and reference documentation

Implementation must create:

~~~text
configs/example/hardware.yaml
configs/example/workload.yaml
configs/example/mapping.yaml
docs/configuration.md
~~~

Every optional field must have a nearby comment explaining its default. Every
enum/model discriminator must have a comment listing all currently implemented
values. Do not advertise future values as usable.

docs/configuration.md must include, for every field:

- file and YAML path;
- type;
- required or optional;
- default;
- unit;
- allowed values;
- validation constraints;
- equation or runtime effect;
- cross-file relationships.

## 7. Replaceable analytical interfaces

The first models are simple, but controller code must never embed their
equations.

Suggested immutable inputs:

~~~python
@dataclass(frozen=True)
class PrefillWork:
    session_id: int
    context_tokens: int
    prefill_tokens: int
    unsynced_decode_tokens: int

@dataclass(frozen=True)
class AttentionWork:
    session_id: int
    context_tokens: int
    device_id: int

@dataclass(frozen=True)
class ExpertWork:
    expert_id: int
    activation_tokens: int
    device_id: int

@dataclass(frozen=True)
class StageContext:
    decode_step: int
    layer_id: int | None
~~~

Suggested protocols:

~~~python
class PrefillLatencyModel(Protocol):
    def latency_s(
        self,
        batch: Sequence[PrefillWork],
        context: StageContext,
    ) -> float: ...

class AttentionLatencyModel(Protocol):
    def latency_s(
        self,
        batch: Sequence[AttentionWork],
        context: StageContext,
    ) -> float: ...

class ExpertComputeLatencyModel(Protocol):
    def latency_s(
        self,
        expert_batches: Sequence[ExpertWork],
        context: StageContext,
    ) -> float: ...

class CollectiveLatencyModel(Protocol):
    def latency_s(self, send_tokens: int, participants: int) -> float: ...

class P2PLatencyModel(Protocol):
    def latency_s(self, tokens: int, src: int, dst: int) -> float: ...
~~~

These interfaces intentionally expose batch/session context even when the V1
equation ignores some fields.

## 8. Coarse compute models

CONFIRMED rate scope:

- decode_attn and decode_ffn rates summarize all layers for one whole-model
  decode step;
- V1 invokes each summarized attention, dispatch, FFN, and combine model once
  per decode step;
- V1 does not multiply latency or communication by model.num_layers;
- a later layer-wise model uses per-layer calibration and repeats the same
  interfaces for each layer.

### 8.1 Prefill compute

CONFIRMED:

For a prefill batch B on device d:

~~~text
T_prefill_compute(d, B) =
    sum(P_i for i in B) / prefill_tokens_per_second(d)
~~~

The interface must also receive batch size, every C_i, and every P_i for later
models.

### 8.2 Decode attention

CONFIRMED:

Batching does not reduce the modeled attention work because sessions have
different contexts.

For session i on attention device d:

~~~text
T_attn_task(i, d) =
    C_i / attention_context_tokens_per_second(d)
~~~

For the local attention batch B_d:

~~~text
T_attn(d) =
    sum(T_attn_task(i, d) for i in B_d)
~~~

CONFIRMED:

All attention DP replicas synchronize before the global FFN step. Each attention
device reserves and executes its local batch independently, and the stage
completes at the latest participating finish:

~~~text
T_attn_stage_finish =
    max(actual_finish_time(d) over participating attention devices)
~~~

### 8.3 Decode FFN expert compute

CONFIRMED:

- Expert compute depends on the number of routed activations for that expert.
- A decode expert GEMM is conceptually 1 x B_e.
- Device-to-expert ownership comes from mapping.yaml.

The simple rate model is:

~~~text
T_expert(e, d) =
    B_e / expert_tokens_per_second(d)
~~~

PROPOSED:

For each FFN stage, the runtime submits one atomic per-device job containing
the Sequence[ExpertWork] for all experts resident on that device:

~~~text
T_ffn_service(d) =
    sum(T_expert(e, d) for active expert e on d)

T_ffn_stage_finish =
    max(actual_finish_time(d) over participating FFN devices)
~~~

No other capability job may interleave inside that atomic FFN-device job. Its
actual finish includes any delay in the shared physical queue. This is
PROPOSED and must be confirmed; a refined grouped-GEMM model can later replace
the sum without changing the per-device expert-batch contract.

## 9. AFD activation exchange

The logical AFD boundary is:

~~~text
attention DP group
  -> dispatch routed hidden activations and routing metadata
  -> FFN EP group executes experts
  -> combine/return expert outputs
  -> attention side resumes
~~~

CONFIRMED V1 core stage sequence:

~~~text
local attention jobs on every DP device
  -> global attention barrier
  -> dispatch all-to-all
  -> per-device expert compute
  -> global FFN barrier
  -> reverse combine all-to-all
  -> emit one token for every active session
  -> increment every emitted session's context by one
~~~

All attention DP replicas synchronize and feed one global FFN step. Dispatch and
combine use the same payload and the same all-to-all model, so their combined
communication cost is twice the one-collective cost. P2P is reserved for KV
movement; no extra activation P2P is added.

PROPOSED ownership boundary: the attention side owns session scheduling,
context/KV state, token order, and sampling, while FFN workers are stateless
expert services.

OPEN:

- For fully co-located attention/FFN work, confirm whether local activation
  exchange is zero-cost.
- For partially overlapping DP/EP groups, define whether the collective charges
  only remote payload or the full logical payload.
- Confirm the proposed attention/FFN ownership boundary.

## 10. FFN routing approximation

CONFIRMED:

- Trace-derived active-session and expert routing will be added later.
- Attention is data-parallel and all DP replicas contribute to one global
  active decode batch B.
- Every active token produces top_k routed expert-activation copies.
- The total routed activation count is A = B * top_k.
- N_EP is FFN expert-parallel width, represented in V1 by the number of
  decode_ffn.device_ids. It is not the number of experts.
- model.num_experts is a separate value.
- Several experts may reside on the same EP device.
- V1 distributes communication equally across EP ranks with ceiling behavior.
- When B * top_k is smaller than N_EP, modeled communication to every EP rank is
  at least one token.

The equal-routing communication payload for each identical collective is:

~~~text
B = total active sessions across all attention DP replicas
A = B * top_k
N_EP = len(decode_ffn.device_ids)
send_tokens = max(1, ceil(A / N_EP))
~~~

The one-token floor applies to communication only; it must not fabricate expert
GEMM activations. The runtime separately constructs a conservation-preserving
expert-load map and groups it by experts_by_device:

~~~text
expert_id -> activation_tokens
sum(activation_tokens over experts) = A
~~~

OPEN:

- How should A be divided across num_experts in the equal-routing compute
  approximation: deterministic quotient/remainder, per-expert ceiling, or
  another rule?
- Confirm one atomic per-device FFN job that sums or groups all resident-expert
  GEMMs, versus individually interleavable expert jobs.
- Is expert replication forbidden in V1?

V1 otherwise uses one listed decode_ffn device as one EP rank; N_EP is exactly
len(decode_ffn.device_ids).

## 11. Interconnect and KV-transfer models

### 11.1 FFN all-to-all

CONFIRMED one-collective equation:

~~~text
T_one_all_to_all =
    START_LAT
  + max(send_tokens, MIN_PACKAGE_TOKENS)
      / THROUGHPUT_TOKENS_PER_SECOND
~~~

Dispatch and combine have the same payload and cost:

~~~text
T_dispatch = T_combine = T_one_all_to_all
T_ffn_communication = 2 * T_one_all_to_all
~~~

The minimum-package value is therefore a payload floor, not a cap.

### 11.2 P2P KV migration

CONFIRMED:

- P2P represents KV migration.
- Payload and throughput use token units.
- KV load/offload throughput is separate from compute throughput.

PROPOSED simple equation:

~~~text
T_p2p =
    START_LAT + kv_tokens / THROUGHPUT_TOKENS_PER_SECOND
~~~

OPEN:

- whether P2P has a minimum-package floor;
- whether pool load/offload also pays P2P startup;
- whether the single V1 resource contends globally or by endpoint;
- whether a transfer also occupies either endpoint's physical device queue.

## 12. Warm-start workload and context evolution

CONFIRMED:

- C at the initial timestamp is an already-established context.
- Initial cold-start prefill is excluded.
- The simulator starts after the session's initial prefill.

For a later turn with incremental prefill P and generated output O:

~~~text
context_before_prefill = C
context_after_prefill = C + P
context_after_decode = C + P + O
~~~

Each emitted decode token increments current context by one before the next
attention calculation.

OPEN:

- At initial arrival, is warm KV already on the selected attention device, only
  in the KV pool, or both?
- Under stateful prefill, is the warm initial snapshot also present on its
  sticky prefill device?
- Whether an initial session may also carry nonzero P at timestamp t is not yet
  specified.

## 13. Stateful and stateless prefill

Prefill has three logical stages:

~~~text
required KV load
  -> prefill compute for new P
  -> transfer the required updated KV to decode attention
~~~

CONFIRMED modes:

Stateful
: The prefill side retains session state and later loads only incremental state
  missing from that retained copy.

Stateless
: The prefill side retains no session KV between turns and loads the full
  existing context before each prefill.

CONFIRMED destination rule:

After prefill, the updated KV required for decode is sent from the prefill side
to the selected decode-attention device. V1 does not add a separate
prefill-to-pool write; pool synchronization occurs after the decode episode.

The attention-transfer payload depends on residency:

- active_only has no inactive attention copy, so it transfers the full updated
  context C + P from the prefill device to the selected attention device;
- an lru_cache hit remains on its cached attention device and needs only the
  newly produced P;
- PROPOSED: an lru_cache miss transfers the full updated context C + P.

PROPOSED stateful interpretation:

- Pin the session to a prefill device while its prefill state is retained.
- Let D_i be decode-output KV created since that prefill mirror last
  synchronized.
- Load D_i before the next prefill.
- Compute P_i.
- Retain the updated prefill-side copy.
- Keep that copy available until the required attention transfer completes.

PROPOSED stateless interpretation:

- Load C_i before every prefill.
- Compute P_i.
- Keep the transient C_i + P_i copy until the attention transfer completes.
- Discard the prefill-side copy after the transfer.

The compute component remains:

~~~text
T_prefill_compute =
    sum(P_i) / R_prefill
~~~

KV load and P-to-D transfer latency are charged by their own models. Their
payloads must not be folded into prefill compute throughput.

OPEN:

- Confirm that the stateful loaded delta is exactly decode-generated KV created
  since the prefill mirror last synchronized, and identify its source.
- Confirm whether pool load and P-to-D use direction-specific rates.
- Serialize load, compute, and attention transfer in V1, or allow overlap?
- Define capacity/eviction behavior for retained stateful-prefill KV, including
  co-location with an attention capability.
- Define prefill batching and stateful prefill-device placement.
- Confirm that attention admission occurs after prefill and that a transient
  stateless copy remains held while capacity admission waits.

## 14. Attention placement, affinity, and capacity

### 14.1 Policy dimensions

CONFIRMED placement options:

- round_robin;
- least_active_context.

CONFIRMED residency options:

- active_only;
- lru_cache.

PROPOSED: placement and residency are orthogonal configuration fields,
producing four valid combinations. Confirm this composition before freezing the
schema.

### 14.2 Active-load score

For attention device d:

~~~text
active_context_load(d) =
    sum(current_context_tokens(s)
        for active session s assigned to d)
~~~

CONFIRMED least-active-context placement:

1. Determine devices that can fit the new session's oracle commitment.
2. Among those feasible devices, select the one with minimum
   active_context_load.
3. If no device is feasible, wait.

PROPOSED: break equal active-load scores deterministically by device ID.

PROPOSED round-robin placement:

1. Start from the round-robin cursor.
2. Scan each attention device at most once.
3. Choose the first feasible device.
4. Advance the cursor only after a successful placement.
5. If none is feasible, wait without advancing the cursor.

### 14.3 Sticky affinity

CONFIRMED:

- A session uses the same attention device for every decode step in its current
  decode episode.
- Under lru_cache, an inactive cached session retains the same attention-device
  affinity across turns until that cache entry is evicted.
- An LRU cache hit reactivates on that same device.
- LRU eviction removes the cached affinity and requires a full-context transfer
  before later decode.
- active_only retains no inactive attention KV and therefore performs a full
  current-context P-to-D transfer before every later decode episode.

OPEN:

- Does active_only retain a logical device assignment after pruning, or rerun
  the configured placement policy for every episode?
- If an LRU-affined device cannot fit the next oracle output commitment after
  evicting other inactive entries, must the session wait there or may it
  migrate?

### 14.4 Oracle capacity commitment

CONFIRMED:

Capacity admission uses the workload's oracle decode length, not the earlier
max(context, 64K) estimate. The 64K heuristic is superseded.

Let C_decode_start include all context that attention will hold when the episode
starts, including the current turn's prefill P. Let O_oracle be the exact number
of tokens the episode will generate:

~~~text
required_final_kv =
    C_decode_start + O_oracle
~~~

A device may admit the session only if it can hold that final footprint. There
is no output-overrun case when the workload obeys its oracle length.

For capacity accounting, distinguish:

~~~text
active_commitment(d) =
    sum(current_context(s) + remaining_oracle_output(s)
        for active session s on d)

resident_used(d) =
    sum(actual current context
        for active and inactive resident sessions on d)
~~~

PROPOSED:

- Hold active_commitment constant through the episode: actual context grows by
  one as remaining oracle output falls by one.
- Treat draining dirty context as pinned until its pool write completes.
- A new candidate is feasible when active and draining commitments plus the
  candidate fit after evicting enough clean inactive entries.
- For an LRU cache hit, protect the candidate's cached C from eviction, replace
  that resident entry with its C + P + O_oracle commitment, and charge only the
  incremental P + O_oracle capacity. Do not double-count or self-evict it.

### 14.5 Capacity waiting and TTFT

CONFIRMED:

- If no eligible attention device can fit a new decode session, the session
  waits in a capacity-blocked state.
- Capacity wait contributes to TTFT.
- Active sessions cannot be pruned.

Use an explicit WAITING_ATTN_CAPACITY state and an FCFS waiting queue
(PROPOSED). Retry waiters whenever feasibility may change, including episode
completion, KV-pool write completion, cache eviction, or capacity release.

PROPOSED TTFT boundary:

~~~text
TTFT = first emitted-token completion time - turn_ready_time
~~~

For the initial episode, turn_ready_time is session arrival. For later turns it
is the timestamp at which new turn input is ready, before any prefill, capacity,
or KV-transfer wait.

### 14.6 active_only residency

CONFIRMED:

- active_only retains no clean inactive attention cache.
- The next decode episode transfers the full current KV context from the
  prefill side to its selected attention device.

PROPOSED draining exception: after decode finishes, the device may temporarily
hold a capacity-pinned DRAINING_DIRTY copy while the overlapping episode-end
pool write runs. This is not reusable inactive cache. Once the write is clean,
remove the attention copy and release its capacity.

### 14.7 lru_cache residency

CONFIRMED:

- Inactive session KV may remain on the attention device across turns.
- A cache hit retains that device and requires only new prefill KV.
- When admitting a new session, prune inactive sessions on the selected device
  in LRU order until the new oracle commitment fits.
- If all evictable inactive sessions are removed and pinned commitments still
  prevent admission, the new session waits.
- Eviction clears cached affinity; a later activation requires a full-context
  transfer.

LRU order must be deterministic. PROPOSED key:

~~~text
(last_access_time, session_id)
~~~

The exact event that refreshes last_access_time remains OPEN.

## 15. Decode-output KV pool synchronization

CONFIRMED:

- KV is sent to the KV pool once after the complete decode episode, not after
  each token step.
- The episode-end pool transfer has modeled cost.
- The transfer may overlap the session's tool wait.
- LRU pruning has modeled cost.

Suggested residency state machine:

~~~text
ACTIVE
  -> DRAINING_DIRTY
  -> INACTIVE_CLEAN
  -> EVICTED
~~~

ACTIVE
: The session is decoding; device KV is pinned.

DRAINING_DIRTY
: Decode ended and the episode-end pool transfer is still running.

INACTIVE_CLEAN
: The pool contains the required recoverable current copy; the attention-device
  copy may remain cached.

EVICTED
: The attention-device copy is gone; a later episode requires a full-context
  transfer.

The generic episode-end write equation is:

~~~text
T_pool_write =
    START_LAT + pool_write_tokens / R_pool_offload
~~~

OPEN payload definition:

- P + O, all unsynchronized additions since the prior pool snapshot; this
  reduces to O when P = 0; or
- the complete final context C + P + O.

The choice must leave the pool with a recoverable current version. A general
O-only write is invalid because prefill P is not written to the pool by another
path.

Other OPEN behavior:

- Must the next turn wait for both tool return and pool-write completion, or
  only when it needs the pool copy?
- Is DRAINING_DIRTY pinned and ineligible for LRU, or may selection force its
  write to finish?
- Is freeing an INACTIVE_CLEAN entry instantaneous because the episode write
  already paid the pruning cost, or does eviction pay another transfer?
- Confirm the proposed active_only rule that holds capacity until the episode
  write completes and then prunes immediately.
- Is KV-pool capacity unbounded in V1?

## 16. Event and scheduling model

CONFIRMED:

- Retain one deterministic global timestamp/sequence-ID event heap.
- Treat one queue per device as one physical service queue, not as an
  independent simulation clock.
- Every capability on a physical device submits through that same queue.
- All attention DP replicas synchronize before one global FFN step.

A controller must not report a stage as started before the shared device
queue's actual reservation start.

PROPOSED:

- Use FCFS discipline for device service queues and capacity waiters.
- Give all-to-all, KV P2P, and the KV pool independently contended queues.
- Transfers use those communication queues without also occupying endpoint
  compute-device queues.
- Admit and remove sessions at decode-step boundaries, retry newly feasible
  waiters, and then start the next global step.

Suggested event vocabulary:

~~~text
SESSION_ARRIVAL
TURN_READY
PREFILL_KV_LOAD_FINISH
PREFILL_COMPUTE_FINISH
PREFILL_KV_TRANSFER_FINISH
ATTN_ADMISSION_RETRY
DECODE_STEP_START
ATTN_STAGE_FINISH
FFN_DISPATCH_FINISH
FFN_COMPUTE_FINISH
FFN_COMBINE_FINISH
DECODE_STEP_FINISH
KV_POOL_WRITE_FINISH
TOOL_RETURN
SESSION_FINISH
~~~

Exact event count is an implementation choice, but every asynchronous resource
completion and barrier must be explicit and deterministic.

CONFIRMED synchronous V1 core behavior:

- reserve each participating attention device independently, then wait for all;
- charge one dispatch collective;
- reserve each participating FFN device independently, then wait for all;
- charge one identical combine collective;
- emit one token per active session.

Later asynchronous microbatching and compute/communication overlap are additive
extensions, not assumptions hidden in V1.

OPEN: confirm FCFS and the proposed communication/pool queue ownership,
including whether transfers also occupy endpoint device queues.

## 17. Remaining open decisions

The following choices are now confirmed and no longer block implementation:
the all-to-all max floor; two identical dispatch/combine collectives;
B * top_k routed traffic; N_EP as EP parallel width distinct from num_experts;
all-layer summary rates; one globally synchronized attention-DP/FFN step;
post-prefill KV transfer to attention; one episode-end pool transfer that may
overlap tool wait; oracle-output capacity admission; cached cross-turn LRU
affinity; and one global event calendar plus one service queue per physical
device.

The remaining V1 questions are below. The implementation must not infer them
silently.

### 17.1 FFN and AFD

1. Co-location: is attention-to-FFN traffic free when both capabilities are on
   the same physical device? For partially overlapping DP/EP groups, charge
   only remote traffic or the full logical collective?
   [A] No partially overlaped. assume full logical collective for simpler implementation
2. Expert loads: divide A = B * top_k across num_experts by deterministic
   quotient/remainder, per-expert ceiling, or another approximation?
   [A] CONSERVATIVE here. We can model the traffic a little more than precise.  
3. Expert execution: submit one atomic per-device job containing all resident
   expert batches, or allow per-expert interleaving/parallelism?
   [A] one atomic job
4. Mapping: forbid expert replication in V1, or support replicated expert
   ownership? One listed FFN device is one EP rank in V1.
   [A] assume no replication for now 
5. Ownership: confirm that attention owns session scheduling, KV, token order,
   and sampling while FFN devices are stateless expert services.
   [A] Yes

### 17.2 KV transfer, pool, and pruning

6. P2P equation: use START_LAT + tokens / throughput, and does it have a
   minimum-package floor?
   [A] No
7. Transfer aggregation: charge startup once per batch, once per session, once
   per source/destination pair, or once per direction?
   [A] KV transfer only charge once per batch. All to all time is modeled as a whole operator. It will be charge once per all-to-all operator. (assume all src-dest have been prepared)
8. Resource queues: do P2P, all-to-all, and pool operations use independent
   queues only, or also occupy endpoint physical-device queues? Do pool load and
   offload share a queue while using direction-specific rates?
   [A] Ues physical-device queues. not direction-specific rate (they are symmetric)
9. Pool payload: after an episode, write all unsynchronized P + O, or the
   complete final context? P + O reduces to O when P = 0.
   [A] Do the unsynchronized P+O as default.
10. Pool/tool join: must the next turn wait for both tool return and pool-write
    completion, or only when it needs the pool copy?
   [A] both completion
11. Dirty pruning: is DRAINING_DIRTY ineligible for LRU until clean? Is clean
    deletion free, or does eviction pay another cost beyond the episode write?
    [A] free
12. Capacity: use an unbounded KV pool in V1? Can workload input pre-populate
    inactive attention cache entries, or do they arise only during simulation?
    [A] KV pool in unbounded

### 17.3 Prefill and warm-copy ownership

13. Stateful delta: is the stateful load exactly decode-produced KV since the
    retained prefill copy last synchronized, and is its source the KV pool?
    [A] yes
14. Stateful capacity: is retained prefill KV capacity modeled? If prefill and
    attention are co-located, do their KV copies share one memory budget?
    [A] Prefill will not be allowed to map to same node as decode. Prefill KV capcity should be large enough for prefill.
15. Timing: serialize KV load, prefill compute, and P-to-D transfer, or allow
    overlap?
    [A] sequential 
16. Batching: use deterministic FCFS greedy prefill batching with no formation
    wait, or another policy?
    [A] yes
17. Placement: use sticky_round_robin for stateful prefill and round_robin for
    stateless prefill? Define cursor, state-loss, and remapping behavior.
    [A] yes. We assume the prefill nodes' memory is shared by all prefill compute node. (in the current coarse implementation, prefill node is abstracted as a single device. So there is no remapping issue. )
18. Warm start: where does initial KV exist -- prefill, attention, pool, or
    multiple locations -- and which first transfer is free?
    [A] Assume the initl KV context has been prefilled and in the pool. 
19. Initial input: may the first observed session also have nonzero P even
    though its initial C is already warm?
    [A] Yes
20. Ordering: perform prefill before attention admission as proposed, and keep a
    transient stateless copy while admission is capacity-blocked?
    [A]y Yes

### 17.4 Attention placement and residency

21. active_only affinity: after pruning, retain a logical D assignment or rerun
    placement for the next episode?
    [A] no rerun placement for next epsiode. 
22. LRU capacity conflict: if the affined D cannot fit the next oracle
    commitment, wait for that D or permit migration?
    [A] no migration. Just wait
23. Oracle commitment: hold C_decode_start + O_oracle for the full episode, with
    actual context growth offset by declining remaining output?
    [A] Yes
24. Round-robin: scan for the first feasible device, or test only the cursor
    device?
    [A] scan for the first
25. Tie handling: use device ID for equal least-active-context scores?
    [A] yes
26. Composition: are both placement policies valid with both residency modes?
    [A] yes
27. LRU ordering: use (last_access_time, session_id), and which event refreshes
    last_access_time?
    [A] the KV contex last decode step time 
28. active_only completion: keep the copy capacity-pinned until the overlapping
    episode pool write is clean, then prune?
    [A] yes

### 17.5 Public schema and metrics

29. Configuration ownership: accept hardware.yaml, workload.yaml, and
    mapping.yaml with seed/horizon in workload and model shape in mapping?
    [A] yes
30. PD/AFD selection: infer behavior structurally from co-located/separate
    mappings, or add an explicit mode?
    [A] infer
31. Field names and CLI: accept context_tokens/prefill_tokens and the proposed
    three-path CLI?
    [A] yes.
32. TTFT: measure turn_ready_time through first emitted-token completion, with
    initial turn_ready_time equal to arrival?
    [A] yes. 
33. Queue discipline: use FCFS for every device queue and capacity waiter?
    [A] yes. 

## 18. Metrics

Session/turn metrics:

- arrival and ready timestamps;
- attention-capacity wait;
- prefill queue and service time;
- KV load/offload and pool synchronization time;
- decode queue/admission time;
- TTFT;
- inter-token latency;
- decode-episode latency;
- tool wait;
- completion latency;
- context and output tokens.

Attention metrics:

- active and resident sessions per device over time;
- active_context_load;
- active_commitment and resident_used tokens;
- capacity utilization;
- admission wait/retry counts;
- placement decisions;
- cache hits and misses;
- LRU evictions;
- dirty/clean inactive counts.

Prefill metrics:

- stateful/stateless mode;
- batch requests and tokens;
- full-context versus delta load tokens;
- compute tokens;
- full-context versus incremental P-to-D transfer tokens;
- service and queue latency;
- stateful-affinity hit/miss.

FFN metrics:

- dispatch/combine latency and token payload;
- active experts;
- B_e distribution;
- per-device expert assignments;
- FFN compute latency and barrier wait;
- load imbalance.

Resource metrics:

- one utilization and queue trace per physical device;
- utilization by capability and total physical utilization;
- all-to-all, P2P, and pool queue/busy time;
- transferred token totals;
- blocked time at each barrier.

Do not report byte totals in the new coarse token-unit model unless a later
conversion model explicitly introduces them.

## 19. Source-file reconstruction

The exact filenames may be adjusted during implementation, but these ownership
boundaries are required.

### 19.1 Keep or minimally adapt

~~~text
src/agentic_sim/core/event_engine.py
src/agentic_sim/core/resource.py
src/agentic_sim/workload/arrival.py
src/agentic_sim/workload/distributions.py
~~~

Retain only behavior compatible with the new typed configuration and state
model.

### 19.2 Replace the monolithic configuration loader

~~~text
delete: src/agentic_sim/config.py

add:
src/agentic_sim/config/__init__.py
src/agentic_sim/config/models.py
src/agentic_sim/config/loader.py
src/agentic_sim/config/validation.py
~~~

Requirements:

- independently load the three files;
- reject unknown keys;
- return typed immutable configuration;
- provide path-specific validation errors;
- perform cross-file capability, device, interconnect, and expert checks;
- reject all legacy aliases.

### 19.3 Reconstruct hardware

~~~text
rewrite:
src/agentic_sim/hardware/device.py
src/agentic_sim/hardware/topology.py

add:
src/agentic_sim/hardware/compute_models.py
src/agentic_sim/hardware/interconnect.py
src/agentic_sim/hardware/kv_pool.py

retire after migration:
src/agentic_sim/hardware/link.py
src/agentic_sim/hardware/bandwidth_pool.py
src/agentic_sim/hardware/simple_backend.py
~~~

Device must combine capability models with exactly one physical queue.

### 19.4 Add explicit mapping components

~~~text
add:
src/agentic_sim/mapping/__init__.py
src/agentic_sim/mapping/model.py
src/agentic_sim/mapping/prefill.py
src/agentic_sim/mapping/attention.py
src/agentic_sim/mapping/experts.py
~~~

Responsibilities:

- immutable role groups and expert ownership;
- stateful-prefill affinity;
- round-robin and least-active-context placement;
- feasibility checks;
- active/inactive residency;
- deterministic LRU;
- expert-load grouping by device.

### 19.5 Reconstruct runtime

~~~text
replace:
src/agentic_sim/runtime/phase_state.py
src/agentic_sim/runtime/prefill_batcher.py
src/agentic_sim/runtime/decode_batcher.py
src/agentic_sim/runtime/kv_ingress.py
src/agentic_sim/runtime/runtime.py

add:
src/agentic_sim/runtime/session.py
src/agentic_sim/runtime/prefill_controller.py
src/agentic_sim/runtime/attention_admission.py
src/agentic_sim/runtime/decode_controller.py
src/agentic_sim/runtime/kv_controller.py

retire:
src/agentic_sim/runtime/reference_policy.py
src/agentic_sim/runtime/task_builder.py
~~~

The new runtime must use typed stage work, not metadata dictionaries whose
meaning depends on legacy task types.

### 19.6 Adapt workload, metrics, assembly, and examples

~~~text
rewrite as needed:
src/agentic_sim/workload/agent.py
src/agentic_sim/workload/invocation.py
src/agentic_sim/workload/workload_generator.py
src/agentic_sim/metrics/*
src/agentic_sim/core/event.py
src/agentic_sim/core/simulation.py
examples/run_simulation.py
README.md
docs/metrics_reference.md
~~~

README and metrics_reference updates belong with implementation, not this
docs-only design turn.

## 20. Implementation sequence

### Milestone 0: resolve the open contract

- Answer every item in Section 17.
- Freeze formulas, stage ordering, state ownership, and YAML names.
- Update this plan and the proposed examples before coding.

Exit criterion: no OPEN item affects V1 behavior.

### Milestone 1: typed three-file configuration

- Add typed models and strict independent loaders.
- Add cross-file validation.
- Add fully commented examples and docs/configuration.md.
- Reject the old one-file schema.

Exit criterion: the three cross-file-consistent examples load into one immutable
SimulationConfig and invalid references fail with precise messages.

### Milestone 2: hardware and analytical primitives

- Implement capability-bearing Device with one shared queue.
- Implement simple prefill, attention, and per-device expert models.
- Implement the confirmed all-to-all equation and the P2P equation frozen in Milestone 0.
- Implement the KV-pool capacity policy frozen in Milestone 0.
- Remove legacy monolithic phase fields and byte-only ingress.

Exit criterion: exact unit tests cover every equation, atomic FFN-device work,
and shared-device serialization.

### Milestone 3: workload, session, and KV-state primitives

- Separate warm initial C, later P, and generated O.
- Add decode-episode state and deterministic context growth.
- Add ACTIVE, DRAINING_DIRTY, INACTIVE_CLEAN, and EVICTED residency state.
- Add pool-copy version/token accounting and LRU timestamps.
- Preserve seeded synthetic workload behavior under the new schema.

Exit criterion: initial C incurs no prefill; context and every logical KV copy
evolve exactly once per P/O token in state-machine unit tests.

### Milestone 4: attention admission and residency

- Implement both confirmed placement policies and their confirmed details.
- Implement capacity reservation and WAITING_ATTN_CAPACITY.
- Implement active_only and lru_cache against the KV-state primitives.
- Implement sticky decode-episode placement, retry, and LRU.
- Implement cache-hit and evicted-session restore paths.

Exit criterion: no device exceeds attention KV capacity, no prune loses the only
recoverable copy, and capacity wait is reflected in TTFT.

### Milestone 5: stateful/stateless prefill

- Implement the confirmed prefill batching policy.
- Implement stateful and stateless KV load rules.
- Implement prefill compute through the replaceable interface.
- Implement every confirmed P copy/synchronization destination.
- Implement stateful-prefill affinity and any confirmed capacity behavior.

Exit criterion: stateful/stateless token movement and latency match the frozen
contract, including warm-start and remapping cases.

### Milestone 6: step-wise PD/AFD decode and output flush

- Build local attention batches.
- Build equal-route expert loads.
- Execute attention, dispatch, atomic per-device FFN compute, combine, and token
  completion.
- Apply confirmed barriers and shared-queue contention.
- Generate confirmed episode-end pool writes and dirty-to-clean transitions.
- Support co-located and disaggregated mappings with the same runtime.

Exit criterion: one step emits one token per active session, attention affinity
is stable for the episode, output KV becomes recoverable, and co-located traffic
matches the frozen rule.

### Milestone 7: metrics, cleanup, and documentation

- Replace legacy metrics with the Section 18 contract.
- Delete retired compatibility paths and tests.
- Update README, metrics_reference, and this summary with verified behavior.
- Run the full new suite and one documented example.

Exit criterion: no production path accepts legacy configuration or invokes the
legacy monolithic decode model, and all reported results come from the new
implementation.

## 21. Required tests

Configuration:

- all three example YAML files load;
- unknown fields and legacy aliases fail;
- missing capability, device, required global interconnect block, and expert
  references fail;
- uncovered experts fail, while duplicate ownership follows the frozen
  replication rule;
- one physical device may legally provide multiple roles;
- example comments enumerate every option.

Analytical models:

- exact prefill sum(P)/R;
- exact attention sum(C_i/R);
- exact expert batch rate;
- confirmed all-to-all boundary below, at, and above MIN_PACKAGE_TOKENS;
- exact P2P and pool equations;
- zero and invalid inputs are handled explicitly.

Device/resource behavior:

- prefill and attention on one device serialize;
- attention and FFN on one device serialize;
- different devices may run concurrently;
- interconnect/pool contention matches the confirmed queue contract;
- deterministic event ties remain stable.

Attention placement:

- sticky device across decode steps;
- deterministic round-robin;
- least-active-context chooses among feasible devices only;
- the deterministic tie policy frozen in Milestone 0;
- capacity wait and retry;
- TTFT includes capacity wait;
- oracle commitment prevents mid-episode capacity overflow;
- active sessions cannot be evicted.

Residency:

- active_only removes inactive residency safely;
- lru_cache hits retain resident KV;
- deterministic inactive-only LRU;
- dirty/clean transitions;
- confirmed episode-end pool-write payload and cost;
- eviction and full-context reload;
- LRU eviction clears cached affinity;
- active_only performs a full-context P-to-D transfer before reactivation.

Prefill:

- warm initial context skips initial prefill;
- stateful loads only the confirmed incremental delta;
- stateless loads full C;
- both compute only P;
- both synchronize the required new P copies;
- stateful affinity is deterministic;
- serialized or overlapped timing matches the confirmed contract.

Decode/AFD:

- exact stage order;
- attention DP assignment;
- A = B * top_k routing and equal-traffic ceiling including A < N_EP;
- token/expert assignment conservation under the confirmed approximation;
- configured device-to-expert ownership;
- per-device FFN aggregation and barrier;
- all attention DP devices synchronize into one FFN step;
- dispatch and combine each charge the same confirmed collective;
- one token per active session;
- C increments before the next attention step;
- whole-model summary rates and communication are applied once per decode step;
- co-located PD and split AFD mappings both execute;
- shared physical queues affect latency.

End to end:

- fixed seed produces identical output;
- multi-turn sessions, tool wait, pool writes, and later prefill join correctly;
- cache pressure produces capacity wait and LRU eviction;
- no device exceeds capacity;
- all generated and migrated token counts reconcile;
- the summary contains no unverified numerical claims.

Future-seam test:

- A fake two-layer execution plan reuses the same analytical interfaces and
  expert mapping without modifying Device or configuration types;
- exercising that seam does not accidentally apply the V1 all-layer summary
  rate twice.

## 22. Acceptance criteria

V1 is complete only when:

- the three-file strict schema is the sole supported configuration;
- every physical device has one shared queue across its capabilities;
- prefill, decode_attn, and decode_ffn can be co-located or separated;
- warm C, incremental P, and generated O have distinct semantics;
- stateful/stateless prefill behavior is implemented;
- attention placement supports round_robin and least_active_context;
- attention residency supports active_only and lru_cache;
- sticky placement, KV capacity, capacity waiting, and TTFT impact are correct;
- output KV synchronization and KV-pool state are modeled;
- attention uses sum of per-session context latency;
- FFN uses explicit expert batches and device-to-expert mapping;
- AFD dispatch and combine each charge the same max-floor collective;
- A = B * top_k routing uses N_EP separately from num_experts;
- all attention DP devices synchronize into one global FFN step;
- whole-model summary models run once per decode step regardless of num_layers;
- oracle-output capacity replaces the earlier 64K estimate;
- one episode-end pool transfer may overlap tool wait;
- LRU affinity persists across turns until eviction, while active_only
  reactivation transfers the full current context;
- AFD dispatch, expert compute, and combine run in the confirmed order;
- all transfer and compute quantities use token-equivalent units;
- metrics expose stage, queue, capacity, cache, and expert behavior;
- the new tests pass and their result is recorded only after execution;
- no legacy compatibility path remains reachable.

## 23. Explicit V1 non-goals

- detailed GPU/xPU/NMP microarchitecture;
- trace-derived expert routing;
- layer-wise execution;
- per-layer expert placement;
- expert replication or migration unless explicitly added before implementation;
- asynchronous AFD microbatch overlap;
- speculative decoding;
- attention-session migration while active;
- advanced admission optimization;
- packet-level or route-level network simulation;
- cold-start initial prefill;
- backward compatibility;
- ASTRA-Sim integration.

The interfaces must leave room for these features without pretending they are
implemented.
