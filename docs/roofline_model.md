# Lightweight MoE roofline model

This is an offline calibration tool for `agentic_sim`. It deliberately trades
fidelity for a small deterministic model requiring effective device FLOP/s,
effective memory bandwidth, an operating point, and network byte rates. The
reusable API is in `roofline_model/`; the wrapper is
`scripts/generate_config.py`.

The generator produces:

- `hardware.yaml`: whole-model prefill/attention rates, per-EP-device FFN
  latency curves, KV-pool rates, and interconnect rates.
- `mapping.yaml`: one prefill device, deterministic attention DP placement, and
  balanced contiguous routed-expert ownership across EP.
- `roofline_report.yaml`: identity, inputs, byte conversions, grouped per-layer
  and all-layer estimates, curves, bottlenecks, and assumptions.

It does not generate a workload. Simulator V1 consumes each rate once per
whole-model phase; `mapping.model.num_layers` is identity only. Grouped layer
results therefore live in the report, while only their latency-summed result
enters `hardware.yaml`.

## YAML input

For repeatable runs, put generator inputs in a flat YAML file using the CLI
destination names. A fully commented example is
[`configs/example/roofline_generator.yaml`](../configs/example/roofline_generator.yaml).

```bash
.venv/bin/python scripts/generate_config.py \
  --input-yaml configs/example/roofline_generator.yaml
```

All required numeric inputs shown in the example may also be supplied as flags.
Prefill, decode attention, and decode FFN each accept their own effective
per-device FLOP/s and memory-bandwidth rates:

```text
prefill_device_flops_per_second
prefill_device_memory_bandwidth_bytes_per_second
decode_attn_device_flops_per_second
decode_attn_device_memory_bandwidth_bytes_per_second
decode_ffn_device_flops_per_second
decode_ffn_device_memory_bandwidth_bytes_per_second
```

The corresponding CLI names use hyphens. The legacy shared
`device_flops_per_second` and
`device_memory_bandwidth_bytes_per_second` inputs remain compatibility
shorthands: each supplies any phase-specific value that was omitted. An
explicit phase value always wins. When both YAML and the command line provide
the same input, the explicit command-line flag takes precedence. `--config`
and `--config-yaml` are aliases for `--input-yaml`. Unknown YAML keys are
rejected, and the same positivity/finite-value checks as the CLI are applied
after merging.

## Supported model registry

The registry is hard-coded so generation is offline and reproducible.
`DeepSeek-V4` means the Pro 1.6T checkpoint. Main layers are grouped when their
attention and FFN behavior is the same.

| Alias | Grouped main layers | Routed / top-k / shared | Official config |
|---|---|---:|---|
| `GLM5.2` | 3 fresh-index DSA+dense; 18 fresh-index DSA+MoE; 57 reused-index DSA+MoE | 256 / 8 / 1 | [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) |
| `DeepSeek-V4` | 31 HCA+MoE; 30 CSA+MoE | 384 / 6 / 1 | [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) |
| `Kimi K3` | 1 KDA+dense; 68 KDA+MoE; 24 gated-MLA+MoE | 896 / 16 / 2 | [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) |
| `Qwen3.8-2.4T` | 69 DeltaNet+MoE; 23 gated-full-attention+MoE | 512 / 10 / 1 | [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json) |
| `GLM-4.5` | 3 GQA+dense; 89 GQA+MoE | 160 / 8 / 1 | [GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json) |
| `DeepSeek-V3` | 3 MLA+dense; 58 MLA+MoE | 256 / 8 / 1 | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) |
| `Kimi-K2` | 1 MLA+dense; 60 MLA+MoE | 384 / 8 / 1 | [Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json) |
| `QWEN3-235B` | 94 GQA+MoE | 128 / 8 / 0 | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json) |

Vision towers and MTP/speculative auxiliary layers are excluded. Kimi-K3 uses
its 3,584-element latent routed width for expert traffic; its main hidden width
is 7,168.

## Equations and calibration

Every operation uses the lower-bound roofline:

```text
compute_time = FLOPs / effective_FLOPs_per_second
memory_time  = bytes / effective_memory_bandwidth_bytes_per_second
time         = max(compute_time, memory_time)
```

There is no implicit utilization factor. Precision widths default to two bytes
for weights, activations, and KV, and can be overridden independently. The
roofline uses the phase's supplied effective rates, so one device can have
different calibrated utilization for prefill, attention, and expert GEMMs.

### Prefill

`P` is the number of new prefill tokens and `C` is their existing context
(`decode_context_tokens` in the generator input). Each layer includes
attention projections and active FFN GEMMs. Full attention counts:

```text
attention_pairs = P * C + P * (P + 1) / 2
```

Sparse/capped attention applies its window, compression, and top-k rules over
positions `C + 1` through `C + P`. KDA and DeltaNet use recurrent work and do
not rescan the prior context. The final prefill rate uses compute time:

```text
prefill_tokens_per_second = P / sum(layer_count * compute_time_per_layer(P, C))
```


### Decode attention

At reference context `C`, a layer includes projection weights, context-growing
KV reads, sparse index scans where relevant, and fixed recurrent-state traffic
for KDA/DeltaNet. Group latencies are summed before conversion:

```text
attention_context_tokens_per_second =
    C / sum(layer_count * roofline_time_per_layer(C))
```

This effective average is not an arithmetic mean of rates. `hardware.yaml`
stores the per-device value; the report also gives the ideal `DP * rate`.

### Decode FFN

Routed MoE work models gate, up, and down SwiGLU matrix multiplies separately.
The weight term loads all experts active at that point on the local EP device.
Dense early layers and shared experts are folded into every device curve using
`local_routed_activations / top_k`, with ideal even EP sharding of their
weights.

Breakpoints include `(0, 0)`, every integer local active-expert transition,
integer neighbors of roofline crossovers, and two tail points. The simulator
uses exact values at breakpoints, linear interpolation, and the final segment
for extrapolation. Dispatch and combine are excluded because the simulator
charges the all-to-all separately.

## Byte-to-token conversion

Token units differ by fabric and must not be mixed:

```text
routed_activation_bytes = routed_expert_input_width * activation_bytes
all_to_all_tokens/s      = all_to_all_bytes/s / routed_activation_bytes
min_package_tokens       = ceil(min_payload_bytes / routed_activation_bytes)
```

An effective KV token includes context-growing KV across relevant layers. MLA
uses its compressed latent cache. KDA/DeltaNet recurrent and convolution states
are fixed with context and are amortized at the reference context:

```text
effective_KV_bytes/token =
    growing_KV_elements_per_token * KV_bytes
  + fixed_state_elements_per_session * KV_bytes / C
```

KV P2P/pool byte rates are divided by this size, and attention KV capacity is
the floor of capacity bytes divided by it. Pool capacity remains `null` because
finite backing-pool capacity is not representable in schema V1.

## Placement and output safety

Default IDs are prefill `0`, attention `1..DP`, then `EP` FFN devices. Routed
experts are assigned contiguously and as evenly as possible. With
`--colocate-attn-ffn`, attention and FFN share IDs; this requires `DP == EP`.
`EP` may not exceed the routed expert count. Existing output files are never
replaced unless `--force` is supplied.

## Intentional approximations and omissions

- “QPS” uses simulator-native units: new prefill tokens/s, routed FFN
  activations, and attention context tokens/s.
- Routing is perfectly balanced and follows deterministic quotient/remainder
  routing. Expert skew, capacity factors, dropped tokens, and hot experts are
  ignored.
- Kernel launches, occupancy, fusion, router/norm/activation cost,
  quantization/dequantization metadata, tensor/pipeline parallelism, and
  communication-compute overlap are ignored.
- Network topology, participant-dependent collective scaling, endpoint NIC
  limits beyond simulator queues, and packet behavior are ignored.
- Sparse attention and recurrent state are collapsed to the selected reference
  context. Compressed MLA KV is assumed even if a simple reference runtime
  caches expanded K/V.
- Dense/shared FFN work assumes ideal EP sharding. Layer-varying expert
  ownership is not representable in the current mapping schema.
- Device memory for weights/workspaces, model loading, KV fragmentation, and
  backing-pool capacity are not modeled.
- The result is a coarse optimistic calibration/lower bound, not a kernel
  benchmark, production capacity forecast, or quantization model.
