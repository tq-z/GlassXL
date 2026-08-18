# Agentic Sim

Agentic Sim is a deterministic discrete-event simulator for coarse token-unit
prefill/decode (PD) and decode attention/FFN (AFD) disaggregation. The current
V1 is a breaking reconstruction: it accepts only the strict three-file schema
and has no Phase 1/1.5 compatibility path.

## Implemented model

- Physical devices declare `prefill`, `decode_attn`, and/or `decode_ffn`
  capabilities. Every physical device has one FCFS queue shared by all of its
  compute and communication endpoint work.
- Initial context `C` is warm KV in the unbounded backing pool. Incremental
  input `P` is prefilled, and oracle output `O` is generated one token per
  synchronized decode step.
- Stateful and stateless prefill use deterministic greedy batching and explicit
  pool-load, compute, and P2P stages.
- Attention placement supports `round_robin` and `least_active_context`;
  residency supports `active_only` and deterministic `lru_cache`.
- Attention DP replicas synchronize into one global FFN step. Dispatch and
  combine each pay the configured full logical all-to-all, including when
  attention and FFN are co-located.
- Expert loads conserve `B * top_k`, use the configured non-replicated expert
  ownership, and execute as one atomic job per FFN device.
- Episode completion writes unsynchronized `P + O` to the pool. Dirty KV remains
  capacity-pinned until the modeled write completes, and the next turn joins
  pool completion with tool return.

V1 rates summarize the whole model and are invoked once per decode step;
`model.num_layers` does not multiply latency or traffic.

## Architecture

```text
hardware.yaml + workload.yaml + mapping.yaml
                    |
                    v
        strict typed configuration
                    |
                    v
        seeded session generation
                    |
                    v
              AgenticRuntime
       +------------+-------------+
       |            |             |
 PrefillController  AttentionAdmission  DecodeController
       |            |             |
       +------------+-------------+
                    |
       device / interconnect / KV-pool queues
                    |
         deterministic event calendar
                    |
              MetricsCollector
```

## Install and test

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

## Run the documented example

```bash
.venv/bin/python examples/run_simulation.py \
  --hardware configs/example/hardware.yaml \
  --workload configs/example/workload.yaml \
  --mapping configs/example/mapping.yaml
```

The runner prints a JSON system summary plus record counts and physical/logical
resource metrics. Programmatic use is equally direct:

```python
from agentic_sim import Simulation, load_config

config = load_config(
    "configs/example/hardware.yaml",
    "configs/example/workload.yaml",
    "configs/example/mapping.yaml",
)
result = Simulation(config).run()
print(result.summary)
print(result.turns[0].to_dict())
```

See [configuration.md](docs/configuration.md) for every YAML field and
[metrics_reference.md](docs/metrics_reference.md) for result fields, units, and
populations. [implementation_summary.md](docs/implementation_summary.md)
records the frozen behavior and verified example results.

## Generate roofline-calibrated configs

The lightweight offline estimator converts effective device FLOP/s, memory
bandwidth, and fabric byte rates into simulator-native token rates. It writes
`hardware.yaml`, `mapping.yaml`, and a grouped per-layer/all-layer
`roofline_report.yaml`:

```bash
.venv/bin/python scripts/generate_config.py \
  --model Kimi-K2 \
  --device-flops-per-second 1e15 \
  --device-memory-bandwidth-bytes-per-second 3e12 \
  --dp 2 --ep 4 \
  --prefill-tokens 4096 --decode-context-tokens 32768 \
  --attn-kv-capacity-bytes 8e10 \
  --ffn-all-to-all-bandwidth-bytes-per-second 4e11 \
  --ffn-all-to-all-start-latency-us 5 \
  --ffn-all-to-all-min-payload-bytes 16384 \
  --kv-p2p-bandwidth-bytes-per-second 2e11 \
  --kv-p2p-start-latency-us 5 \
  --kv-pool-bandwidth-bytes-per-second 1e11 \
  --kv-pool-start-latency-us 10 \
  --output-dir generated/kimi-k2
```

Inputs use decimal SI units and are treated as effective rates. Existing output
files are protected unless `--force` is passed. See
[roofline_model.md](docs/roofline_model.md) for equations, supported model
sources, token conversions, and intentional omissions.


For a commented reusable input file, use:

```bash
.venv/bin/python scripts/generate_config.py \
  --input-yaml configs/example/roofline_generator.yaml
```

## V1 non-goals

V1 does not model layer-wise execution, trace-derived routing, expert
replication, asynchronous AFD overlap, speculative decoding, active attention
migration, packet-level networks, cold-start prefill, or byte conversion.
