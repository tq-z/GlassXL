# Codex Plan v2: Top-Down Agentic Inference Simulator

## 0. Main Objective

Build a lightweight, modular, discrete-event simulator for agentic inference on a heterogeneous multi-node platform.

Implementation priority:

1. Get a complete end-to-end simulator running.
2. Produce early queueing and utilization results.
3. Keep workload and hardware abstractions replaceable.
4. Add xPU/NMP, MoE, scheduling, passive-fabric details, traces, and ASTRA-Sim incrementally.

The first working version should **not** model detailed xPU internals, NMP internals, wavelength allocation, or advanced scheduling.

The initial simulator only needs:

- a generic hardware abstraction;
- generic compute, memory, transfer, and wait tasks;
- latency + bandwidth hardware models;
- an agentic event/state model;
- Poisson arrivals and configurable random distributions;
- a simple reference execution policy;
- end-to-end latency, queueing, throughput, and utilization metrics.

---

# 1. High-Level Architecture

```text
Agentic Workload Generator
    ↓ creates Agent / Invocation events

Agentic Runtime
    ↓ creates generic ComputeTask / MemoryTask / TransferTask

Discrete-Event Engine
    ↓ executes tasks on abstract hardware resources

Simple Hardware Backend
    - compute service rate + fixed latency
    - memory bandwidth + fixed latency
    - link bandwidth + fixed latency

Metrics
    - request latency
    - queueing delay
    - throughput
    - resource utilization
```

Later extensions replace individual modules without rewriting the core:

```text
Simple Compute Model
    → xPU/NMP analytical service models

Simple Link Model
    → passive-fabric reservation model
    → ASTRA-Sim backend

Generic Invocation
    → dense execution model
    → MoE execution model

Reference Policy
    → scheduling baselines
    → joint scheduling policy
```

---

# 2. Core Design Rules

## 2.1 The event engine owns time

The simulator advances only through events:

```text
pop earliest event
→ update simulation time
→ update state
→ create new events
→ repeat
```

No cycle-by-cycle simulation.

## 2.2 Workload and hardware remain independent

The workload layer does not know whether a task runs on a GPU, xPU, NMP, CPU, or generic accelerator.

The hardware layer does not know whether a task belongs to normal LLM serving, an agent turn, a tool-return event, or a MoE expert.

## 2.3 Use generic task objects first

The first implementation uses only:

- `ComputeTask`
- `MemoryTask`
- `TransferTask`
- `WaitTask`

Specialized xPU, NMP, KV, and MoE tasks are added later.

## 2.4 First version uses a reference policy, not a research scheduler

The initial execution policy is intentionally simple:

- FCFS admission;
- static device mapping;
- no migration;
- no network-aware placement;
- no batching optimization.

It exists only to make the simulator runnable and testable.

---

# 3. Repository Structure

```text
simulator/
├── README.md
├── pyproject.toml
├── configs/
│   ├── hardware/
│   ├── workloads/
│   └── experiments/
│
├── src/
│   └── agentic_sim/
│       ├── core/
│       │   ├── event.py
│       │   ├── event_engine.py
│       │   ├── task.py
│       │   ├── resource.py
│       │   └── simulation.py
│       │
│       ├── workload/
│       │   ├── agent.py
│       │   ├── invocation.py
│       │   ├── arrival.py
│       │   ├── distributions.py
│       │   └── workload_generator.py
│       │
│       ├── runtime/
│       │   ├── runtime.py
│       │   ├── reference_policy.py
│       │   └── task_builder.py
│       │
│       ├── hardware/
│       │   ├── hardware.py
│       │   ├── device.py
│       │   ├── link.py
│       │   ├── simple_backend.py
│       │   └── topology.py
│       │
│       ├── metrics/
│       │   ├── collector.py
│       │   ├── request_metrics.py
│       │   ├── resource_metrics.py
│       │   └── export.py
│       │
│       └── cli/
│           ├── run.py
│           └── sweep.py
│
├── experiments/
│   ├── phase1_framework/
│   ├── phase2_heterogeneous_hw/
│   ├── phase3_moe/
│   ├── phase4_scheduling/
│   ├── phase5_fabric/
│   └── phase6_validation/
│
└── tests/
    ├── test_event_engine.py
    ├── test_resources.py
    ├── test_arrivals.py
    ├── test_agent_state.py
    ├── test_simple_backend.py
    └── test_end_to_end.py
```

---

# 4. Phase 1: Minimal End-to-End Framework

## Goal

Build the smallest complete simulator that can execute agentic workloads on abstract hardware and produce meaningful queueing results.

This phase is the highest priority.

Do not implement:

- detailed xPU/NMP models;
- MoE;
- dynamic batching optimization;
- KV migration;
- passive optical reservation;
- ASTRA-Sim.

## 4.1 Event engine

Implement:

```python
class EventType(Enum):
    AGENT_ARRIVAL = auto()
    INVOCATION_READY = auto()
    TASK_READY = auto()
    TASK_START = auto()
    TASK_FINISH = auto()
    TOOL_WAIT_START = auto()
    TOOL_RETURN = auto()
    AGENT_FINISH = auto()
```

```python
@dataclass(order=True)
class Event:
    time: float
    sequence_id: int
    event_type: EventType = field(compare=False)
    payload: object = field(compare=False)
```

Requirements:

- priority queue based on event time;
- deterministic tie-breaking;
- fixed random-seed support;
- no task may execute before its dependencies;
- simulation stops when all agents finish or a configured end time is reached.

## 4.2 Generic task abstraction

```python
class TaskType(Enum):
    COMPUTE = auto()
    MEMORY = auto()
    TRANSFER = auto()
    WAIT = auto()


@dataclass
class Task:
    task_id: int
    task_type: TaskType
    agent_id: int
    invocation_id: int
    ready_time: float
    dependencies: list[int]
    metadata: dict
```

Examples:

```python
# Compute metadata
{
    "work_units": 1.0,
    "device_class": "compute"
}
```

```python
# Transfer metadata
{
    "bytes": 1048576,
    "src": 0,
    "dst": 1
}
```

## 4.3 Generic hardware abstraction

### Device

```python
@dataclass
class Device:
    device_id: int
    device_class: str
    service_rate: float
    fixed_latency: float
    queue_policy: str = "fcfs"
```

Initial compute service time:

```text
T_compute = fixed_latency + work_units / service_rate
```

### Memory resource

```python
@dataclass
class MemoryResource:
    memory_id: int
    capacity_bytes: int
    bandwidth_bytes_per_sec: float
    fixed_latency: float
```

Initial memory service time:

```text
T_memory = fixed_latency + bytes / bandwidth
```

### Link

```python
@dataclass
class Link:
    src: int
    dst: int
    bandwidth_bytes_per_sec: float
    fixed_latency: float
```

Initial transfer service time:

```text
T_transfer = fixed_latency + bytes / bandwidth
```

No wavelength, TX/RX, route, or shared-segment details in Phase 1.

## 4.4 Simple topology

Support a configurable graph:

```yaml
hardware:
  devices:
    - id: 0
      class: compute
      service_rate: 1.0e12
      fixed_latency_us: 5

    - id: 1
      class: memory
      service_rate: 1.0e12
      fixed_latency_us: 5

  links:
    - src: 0
      dst: 1
      bandwidth_gbps: 400
      fixed_latency_us: 1
```

Initially support:

- direct link lookup;
- optional default all-to-all link;
- serialized transfers per link;
- no alternate routing.

## 4.5 Agentic state model

```python
class AgentState(Enum):
    NOT_ARRIVED = auto()
    READY = auto()
    RUNNING = auto()
    TOOL_WAIT = auto()
    FINISHED = auto()
```

```python
@dataclass
class Agent:
    agent_id: int
    arrival_time: float
    state: AgentState

    num_turns: int
    current_turn: int

    input_tokens_per_turn: list[int]
    output_tokens_per_turn: list[int]
    tool_wait_per_turn: list[float]

    current_context_tokens: int
    start_time: float | None
    finish_time: float | None
```

State progression:

```text
arrival
→ invocation
→ optional tool wait
→ next invocation
→ ...
→ finish
```

The first version does not need token-by-token prefill/decode execution.

Each invocation can be represented as a small generic task graph:

```text
compute
→ transfer
→ compute
→ optional tool wait
```

## 4.6 Workload distributions

Provide reusable distributions:

- constant;
- uniform;
- exponential;
- lognormal;
- discrete categorical;
- bounded integer.

### Arrival process

Implement Poisson arrivals:

```text
inter-arrival time ~ Exponential(rate = λ)
```

Example:

```yaml
workload:
  num_agents: 10000

  arrival:
    type: poisson
    rate_per_second: 20

  turns:
    type: discrete
    values: [1, 2, 3, 4, 5]
    probabilities: [0.1, 0.2, 0.3, 0.25, 0.15]

  input_tokens:
    type: lognormal
    mean: 7.0
    sigma: 0.8
    min: 128
    max: 32768

  output_tokens:
    type: lognormal
    mean: 5.0
    sigma: 0.7
    min: 16
    max: 4096

  tool_wait_ms:
    type: lognormal
    mean: 6.0
    sigma: 1.0
    min: 10
    max: 30000
```

All random draws must be reproducible using a configured seed.

## 4.7 Invocation cost model

Avoid detailed Transformer modeling in Phase 1.

Map invocation properties to abstract work:

```text
compute_work =
    a0
  + a1 × input_tokens
  + a2 × output_tokens
  + a3 × (context_tokens+input_tokens) × output_tokens
```

```text
transfer_bytes =
    b0
  + b1 × input_tokens
  + b2 × output_tokens
  + b3 × context_tokens
```

The coefficients are configurable.

The purpose is not hardware accuracy. The purpose is to create workload pressure that changes with:

- input length;
- output length;
- context growth;
- number of turns;
- tool-wait behavior.

## 4.8 Reference execution policy

Implement one replaceable policy:

```text
FCFS
+ static device assignment
+ no migration
+ no batching optimization
```

```python
class ExecutionPolicy(ABC):
    @abstractmethod
    def place_task(
        self,
        task: Task,
        hardware_state: HardwareState,
    ) -> Placement:
        ...
```

## 4.9 Phase 1 metrics

Agent-level:

- arrival time;
- start time;
- finish time;
- total latency;
- active execution time;
- tool-wait time;
- queueing time;
- number of turns;
- final context length.

Hardware-level:

- device busy time;
- device utilization;
- link busy time;
- link utilization;
- queue length over time;
- maximum queue length;
- bytes transferred.

System-level:

- completed agents per second;
- invocation throughput;
- average latency;
- p50/p95/p99 latency;
- average number of active agents;
- average number of waiting agents;
- offered load vs. completed load.

## 4.10 Phase 1 experiments

The first simulator should produce:

1. throughput vs. Poisson arrival rate;
2. p50/p95/p99 latency vs. arrival rate;
3. utilization vs. arrival rate;
4. active-agent count over time;
5. tool-wait-agent count over time;
6. effect of tool-wait distribution;
7. effect of number of turns;
8. effect of context growth;
9. queue buildup near saturation;
10. sensitivity to compute latency and link bandwidth.

These are framework-validation results, not final architecture results.

## 4.11 Phase 1 acceptance criteria

Phase 1 is complete when:

- the simulator runs end-to-end;
- Poisson agent arrivals are supported;
- configurable per-agent distributions are supported;
- agents execute multiple turns;
- tool waits suspend and resume agents;
- context length grows across turns;
- abstract compute and transfer tasks execute on hardware resources;
- queueing emerges naturally;
- latency and utilization metrics are exported;
- an arrival-rate sweep produces a saturation curve;
- all results are reproducible with a fixed seed.

---

# 5. Phase 2: Add Heterogeneous xPU/NMP Semantics

## Goal

Replace generic devices with simple xPU and NMP abstractions.

This phase adds architectural meaning but still uses analytical latency models.

Do not add detailed optical fabric or advanced scheduling yet.

## 5.1 xPU abstraction

Add task types:

- prefill compute;
- projection;
- FFN;
- optional generic dense compute.

Initial model:

```text
T_xPU = fixed_overhead + max(
    FLOPs / effective_compute,
    bytes / effective_memory_bandwidth
)
```

Support batch size as an input.

## 5.2 NMP abstraction

Add task types:

- decode attention;
- KV read;
- KV write;
- optional KV restore.

Initial model:

```text
T_NMP = fixed_overhead + max(
    attention_FLOPs / effective_NMP_compute,
    KV_bytes / effective_NMP_bandwidth
)
```

No bank/channel timing model yet.

## 5.3 Split execution flow

Replace the generic invocation graph with:

```text
prefill on xPU
→ KV placement on NMP
→ NMP attention
→ activation transfer
→ xPU FFN
→ optional return transfer
```

A simplified layer multiplier is sufficient.

## 5.4 Attention and FFN batch sizes

Add configurable:

- attention micro-batch size;
- FFN batch size;
- maximum aggregation delay.

At this point, implement only the mechanism, not an optimized scheduler.

## 5.5 Phase 2 experiments

- xPU/NMP ratio;
- xPU compute rate;
- NMP bandwidth;
- context length;
- attention micro-batch size;
- FFN batch size;
- serial vs. pipelined execution;
- xPU/NMP utilization balance;
- ideal transfer vs. finite link bandwidth.

## 5.6 Phase 2 acceptance criteria

- generic devices can be replaced by xPU and NMP;
- split attention/FFN execution runs end-to-end;
- xPU and NMP have separate queues and utilization;
- context length affects NMP service time;
- batch size affects xPU service time;
- the Phase 1 workload generator remains unchanged.

---

# 6. Phase 3: Add Basic MoE Semantics

## Goal

Add MoE shortly after xPU/NMP, but keep the first MoE model simple.

Do not add advanced expert scheduling yet.

## 6.1 Minimal MoE model

Add:

- number of experts;
- top-k;
- token-to-expert assignment;
- static expert placement;
- dispatch bytes;
- expert compute;
- combine bytes.

Initial routing options:

- uniform random;
- Zipf/skewed;
- domain-conditioned categorical distribution.

## 6.2 Simple execution

```text
router
→ dispatch
→ expert compute
→ combine
```

Use static expert placement and FCFS per expert device.

No expert replication or migration initially.

## 6.3 Agentic and MoE separation

The agent workload creates invocations.

The model layer decides whether an invocation is dense or MoE.

The agent state machine must not contain expert logic.

## 6.4 Phase 3 experiments

- dense vs. MoE;
- routing skew;
- top-k;
- expert count;
- expert imbalance;
- dispatch/combine traffic;
- tool-return bursts and expert queue buildup;
- domain-conditioned routing.

## 6.5 Phase 3 acceptance criteria

- dense and MoE use the same event engine;
- static expert placement works;
- routing skew creates measurable expert imbalance;
- MoE network traffic is separated from normal activation traffic;
- agentic turns can invoke either a dense or MoE model.

---

# 7. Phase 4: Add Scheduling Baselines

## Goal

Only after the simulator and hardware abstractions are stable, add scheduling policies.

Scheduling is not required for the first working version.

## 7.1 Agent-level baselines

1. FCFS;
2. shortest remaining work;
3. earliest deadline first;
4. least-loaded device;
5. KV-locality first.

## 7.2 xPU/NMP policies

1. static xPU/NMP pairing;
2. least-loaded xPU;
3. least-loaded NMP;
4. fixed attention batch;
5. fixed FFN batch;
6. timeout-based FFN aggregation.

## 7.3 MoE policies

1. static expert placement;
2. least-loaded expert replica;
3. hot-expert replication;
4. expert micro-batch aggregation.

## 7.4 Joint policies

Only after isolated baselines work, add:

- KV + compute aware;
- MoE + network aware;
- agent + KV + expert aware.

## 7.5 Phase 4 acceptance criteria

- policies share one interface;
- all policies run on identical workloads;
- policy behavior is deterministic for a fixed seed;
- queueing, utilization, and SLO metrics differ across policies;
- policy ablations run automatically.

---

# 8. Phase 5: Add Passive-Fabric Details

## Goal

Replace the simple latency + bandwidth link model with progressively richer passive-fabric semantics.

This phase must not block Phase 1–4.

## 8.1 Fidelity levels

### Level 0

```text
fixed latency + fixed bandwidth
```

### Level 1

```text
per-source TX capacity
+ per-destination RX capacity
```

### Level 2

```text
fixed paths
+ shared path-segment contention
```

### Level 3

```text
waveguide resources
+ wavelength groups
```

### Level 4

```text
multiple wavelengths per flow
+ shared bandwidth
+ fragmentation
```

## 8.2 Traffic classes

- activation;
- KV transfer;
- MoE dispatch;
- MoE combine;
- prefix replication.

## 8.3 Phase 5 experiments

- simple link vs. TX/RX contention;
- fixed-path hotspot;
- many-to-one traffic;
- wavelength count;
- endpoint bandwidth;
- MoE and KV traffic interference;
- network-aware placement.

## 8.4 Phase 5 acceptance criteria

- the simple link model remains selectable;
- each fidelity level is independently selectable;
- higher-fidelity models preserve the same task interface;
- contention changes completion order;
- fabric results can be compared against the simple model.

---

# 9. Phase 6: Trace Support and ASTRA-Sim Validation

## Goal

Only after the top-down simulator produces stable results, add trace input and ASTRA-Sim integration.

## 9.1 Trace support

Add parsers for:

- arrival timestamps;
- per-turn input/output tokens;
- tool-call timing;
- expert routing;
- model choice.

The trace parser must produce the same internal `Agent` and `Invocation` objects as the synthetic generator.

## 9.2 ASTRA-Sim integration

ASTRA-Sim acts as a backend for representative epochs.

The runtime exports:

```text
compute operations
memory operations
communication operations
dependencies
device mapping
```

Use ASTRA-Sim for:

- representative pipeline epochs;
- network/topology validation;
- MoE communication epochs;
- KV restore bursts;
- selected high-fidelity sensitivity points.

Do not make ASTRA-Sim the owner of the agentic state machine.

## 9.3 Validation strategy

Compare the fast backend and ASTRA-Sim for:

- epoch makespan;
- operation ordering;
- communication overlap;
- topology sensitivity;
- bottleneck classification;
- scheduler ordering.

The main goal is trend and ordering validation, not exact numerical equality.

---

# 10. Implementation Order for Codex

## Pull Request 1: Framework Skeleton

Implement:

- repository layout;
- configuration loader;
- event types;
- event queue;
- simulation clock;
- generic tasks;
- generic resources;
- metrics skeleton;
- unit tests.

No workload yet.

## Pull Request 2: Agentic Synthetic Workload

Implement:

- `Agent`;
- `AgentState`;
- `Invocation`;
- Poisson arrivals;
- configurable distributions;
- multi-turn progression;
- tool waits;
- context growth;
- workload unit tests.

No hardware specialization yet.

## Pull Request 3: Simple Hardware Backend

Implement:

- generic compute devices;
- generic memory devices;
- latency + bandwidth links;
- FCFS queues;
- static placement;
- end-to-end execution;
- request and resource metrics.

At the end of PR3, the first complete simulator must run.

## Pull Request 4: Phase 1 Experiments

Implement:

- CLI runner;
- parameter sweeps;
- CSV/JSON export;
- arrival-rate sweep;
- saturation curve;
- latency percentiles;
- utilization plots;
- example configs.

This is the first result milestone.

## Pull Request 5: xPU/NMP Abstractions

Implement:

- xPU analytical model;
- NMP analytical model;
- split attention/FFN execution;
- separate queues;
- simple batching parameters;
- xPU/NMP experiments.

## Pull Request 6: Basic MoE

Implement:

- MoE config;
- router;
- static expert placement;
- dispatch;
- expert compute;
- combine;
- imbalance metrics.

## Pull Request 7: Scheduling Baselines

Implement:

- scheduler interface;
- agent baselines;
- xPU/NMP baselines;
- MoE baselines;
- automated ablations.

## Pull Request 8: Passive-Fabric Details

Implement fidelity levels:

- simple link;
- TX/RX;
- fixed path;
- waveguide;
- wavelength.

## Pull Request 9: Trace Input

Implement:

- trace schema;
- trace parser;
- synthetic/trace workload equivalence;
- trace validation tests.

## Pull Request 10: ASTRA-Sim Backend

Implement:

- representative epoch export;
- ASTRA runner;
- result parser;
- cross-validation scripts.

---

# 11. Phase 1 Configuration Example

```yaml
seed: 1
simulation_end_s: 100

hardware:
  devices:
    - id: 0
      class: compute
      service_rate: 1.0e12
      fixed_latency_us: 5

    - id: 1
      class: memory
      service_rate: 5.0e11
      fixed_latency_us: 10

  links:
    - src: 0
      dst: 1
      bandwidth_gbps: 400
      fixed_latency_us: 1

    - src: 1
      dst: 0
      bandwidth_gbps: 400
      fixed_latency_us: 1

workload:
  num_agents: 10000

  arrival:
    type: poisson
    rate_per_second: 20

  turns:
    type: discrete
    values: [1, 2, 3, 4]
    probabilities: [0.1, 0.3, 0.4, 0.2]

  input_tokens:
    type: lognormal
    mean: 7.0
    sigma: 0.8
    min: 128
    max: 32768

  output_tokens:
    type: lognormal
    mean: 5.0
    sigma: 0.7
    min: 16
    max: 4096

  tool_wait_ms:
    type: lognormal
    mean: 6.0
    sigma: 1.0
    min: 10
    max: 30000

invocation_model:
  compute:
    base_work: 1.0e9
    per_input_token: 2.0e7
    per_output_token: 4.0e7
    context_output_term: 2.0e4

  transfer:
    base_bytes: 4096
    per_input_token: 2048
    per_output_token: 4096
    per_context_token: 1024

policy:
  type: static_fcfs
  compute_device: 0
  memory_device: 1
```

---

# 12. Phase 1 Output Format

```text
results/<experiment>/<run_id>/
├── config.yaml
├── summary.json
├── agents.csv
├── invocations.csv
├── resources.csv
├── queue_timeseries.csv
└── timeline.json
```

Minimum summary:

```json
{
  "completed_agents": 0,
  "agent_throughput_per_s": 0.0,
  "invocation_throughput_per_s": 0.0,
  "mean_latency_ms": 0.0,
  "p50_latency_ms": 0.0,
  "p95_latency_ms": 0.0,
  "p99_latency_ms": 0.0,
  "mean_queueing_ms": 0.0,
  "mean_tool_wait_ms": 0.0,
  "compute_utilization": 0.0,
  "memory_utilization": 0.0,
  "link_utilization": 0.0,
  "max_compute_queue": 0,
  "max_memory_queue": 0
}
```

---

# 13. Immediate Codex Task

Implement only Pull Requests 1–3.

Deliver:

- a working Python package;
- event engine;
- generic task and resource abstractions;
- Poisson arrivals;
- configurable random distributions;
- multi-turn agents;
- tool waits;
- context growth;
- simple compute/memory/link hardware;
- static FCFS placement;
- end-to-end simulation;
- basic metrics;
- unit and integration tests;
- one example configuration.

Do not implement yet:

- xPU-specific equations;
- NMP-specific equations;
- MoE;
- KV migration;
- advanced scheduling;
- passive optical resource reservation;
- wavelength allocation;
- trace input;
- ASTRA-Sim.

The first target is a clean simulator framework that can show:

```text
arrival rate
→ queue buildup
→ utilization saturation
→ tail-latency growth
```

while also capturing:

```text
multi-turn agents
+ tool waits
+ context growth
```

All later architecture features must plug into this framework without changing the event engine or workload interfaces.