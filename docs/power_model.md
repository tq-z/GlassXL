# Per-die power traces (`power_model`)

`power_model` turns a PanAgretti simulation into per-die power over time for a
split of GPU prefill chips and NMP decode nodes, with expert FFN and all-to-all
energy placed on chips by real MoE routing. It is the input for thermal
simulation; it does no thermal modelling itself. `agentic_sim` never imports it
and its default behaviour is unchanged.

```text
sessions.json ─► AgenticRuntime + PowerCollector ─► activity.npz ─► price(power.yaml) ─► power.npz, dies.csv, chips.csv
                       ▲ routing source (real decode rows)                ▲ coefficients + static power
generate_config.py ─► hardware.yaml, mapping.yaml, roofline_report.yaml (deployment block, skew-aware FFN curves)
```

## Quick start

```bash
# H200 prefill + perf_sim-default NMP nodes, PIC statics from Wang et al. CICC 2024.
python scripts/generate_config.py --input-yaml configs/power/h200_perf_sim_generator.yaml \
    --routing-traces ../moe_routing --force
python scripts/power_trace.py --config-dir configs/power/h200_perf_sim \
    --power configs/power/h200_perf_sim.yaml \
    --sessions-json configs/glassml/workload/sessions.json --num-sessions 300 \
    --arrival-rate 4 --seed 11 --horizon 120 --routing traces \
    --check-parity --check-completeness --out results/power/h200_perf_sim_traces

# PanAgretti paper hardware (B200-like prefill, paper NMP geometry).
python scripts/generate_config.py --input-yaml configs/power/smoke_generator.yaml \
    --routing-traces ../moe_routing --output-dir configs/power/smoke_routed --force
python scripts/power_trace.py --config-dir configs/power/smoke_routed \
    --power configs/power/panagretti_b200.yaml ... --out results/power/paper_b200

# Re-price with edited coefficients, without re-simulating.
python scripts/power_trace.py --from-activity results/power/h200_perf_sim_traces/activity.npz \
    --config-dir configs/power/h200_perf_sim --power my_power.yaml --out results/power/repriced

# Sweeps on any GPU/NMP split (see "Sweeps" below).
python scripts/power_sweep.py grid --gpus 16 --nodes 16 --stacks 6 --out results/power/grid_1to6_16x6
python scripts/power_sweep.py splits --out results/power/splits

# (Once) fit NMP coefficients and measure node rates against perf_sim kernels.
python scripts/power_crosscheck.py --geometry perf_sim_default --stacks-per-node 4 \
    --measure-rates --out results/power/perf_sim_crosscheck_default.json
```

## Simulator timing options used by the power scripts

`power_model.simulate` and `scripts/power_trace.py` construct the runtime with
`AgenticRuntime(..., prefill_transfers_at_completion=True)`. The simulator's
default (still `False`) books each prefill batch's attention admissions and KV
P2P transfers at *launch*, with the compute finish as a future ready time, and on
the prefill device's queue. Because a P2P reservation also occupies the
destination NMP device's FCFS queue (no backfill), every decode step issued before
that transfer starts waits behind it, and the lock-step decode barrier stalls the
whole panel, for up to 22 s in the 300-session H200 run. With the option on,
transfers are booked by a `PREFILL_COMPLETE` event at compute finish and leave
through the prefill device's copy engine (they do not queue behind its next,
reserved-ahead compute). `--legacy-prefill-transfers` reproduces the old booking.
`run_summary.json` reports `checks.stall.longest_decode_step_s`.

The attention admission controller also no longer deadlocks. With LRU residency,
a waiter's cached copy is protected from eviction; when nothing is active or
draining (so no pool write or eviction will ever call `retry` again) and the FCFS
head is blocked only by protected copies, the head now evicts those clean,
recoverable copies. FCFS order is unchanged; runs that never deadlocked are
bit-identical. `checks.stall.admission_deadlock_releases` counts these releases.

## What is modelled

The collector sees every simulator event (prefill batches, decode steps,
P2P KV transfers, pool loads/offloads) and bins seven activity counters per
device: `flop_gemm, flop_attn, mem_rd, mem_wr, act, io_tx, io_rx`. Counters come
from `roofline_model/activity.py`, which reuses the roofline estimator's model
math with bytes split by kind.

| Event | Activity | Time window |
|---|---|---|
| Decode step, attention | per session: KV reads, QK/AV FLOPs, projection FLOPs, activations, KV append; projection weights once per device | [latest attention start, combine finish] (real hardware interleaves layers behind a per-layer barrier) |
| Decode step, FFN | per MoE layer and device: routed tokens `x`, distinct experts hit `k` (weights `3·k·rw·ei`, FLOPs `6·x·rw·ei`); shared/dense work at the even EP share | same step window |
| Decode step, all-to-all | remote (token, expert) copies × `routed_width × act_bytes`, dispatch + combine | same step window |
| Prefill batch | FLOPs `a_tok·T + a_pair·Σ(P·C + P(P+1)/2)`; weights once per batch (expected expert coverage); context KV once per session; TP all-reduce bytes | compute reservation |
| KV P2P, pool load/offload | bytes = tokens × KV bytes/token at both endpoints; pool traffic on a `kv_pool` pseudo-die | their reservations |

Pricing (per bin, `P = E/dt + static`):

```text
NMP  dramJ  = (mem_rd + mem_wr)·8·e_dram / dram_dies                        + static p_dram (each die)
     logic  = flop_gemm·e_mac_gemm/2 + flop_attn·e_mac_attn/2 + mem_rd·8·e_operand_fill
              + act·8·e_sram + (io_tx + io_rx)·8·e_io                        + static p_logic + p_io
     pic    = (io_tx + io_rx)·8·e_pic                                        + static p_pic
GPU  per chip (device activity / TP):
     xpu    = FLOPs·e_flop                                                   + static p_xpu
     hbm    = (mem_rd + mem_wr + act)·8·e_hbm                                + static p_hbm
     pic    = (io_tx + io_rx)·8·e_link                                       + static p_pic + p_io
```

Die names: `nmp017.dram0..N`, `nmp017.logic`, `nmp017.pic`, `gpu0_3.xpu`,
`gpu0_3.hbm`, `gpu0_3.pic`, `kv_pool`. On the NMP node the optical-I/O drivers
(EIC) sit on the logic die, following PanAgretti's 344.7 W logic-die budget; on a
GPU chip they sit with the PIC, which is flip-chip bonded to its driver EIC
(Wang et al. Fig. 2d).

## Presets and provenance

Coefficients live in `power_model/coefficients.py`; each carries a provenance
string and `power.yaml` overrides them by name. Values no source provides are
placeholders, used only with `allow_placeholders: true` and listed in every output.

| Preset | Kind | Basis |
|---|---|---|
| `panagretti_nmp` | NMP | PanAgretti Table I node (16 PUs × 32 × 32×32 MACs, 4 dies × 16 TB/s); coefficients fitted to perf_sim at that geometry (0.6575 pJ/MAC, 0.01646 pJ/bit operand fill); PIC 3 W from the paper |
| `perf_sim_default_nmp` | NMP | four unpatched perf_sim stacks (16 ch × 16 banks × 16×16 MACs, 8 tRCD tiers, 32 GiB each); fit at that geometry: 0.659 pJ/MAC, 0.01655 pJ/bit, 0.4291 pJ/bit DRAM (FFN/projection within 0.5%, attention within 1.8% of total energy, logic-die-only residual up to 2.6%) |
| `b200_like` | GPU | PanAgretti B200-like prefill node, 2.25 PFLOP/s, 8 TB/s, 1000 W |
| `h200` | GPU | perf_sim's xPU entry: 980 TFLOPS, 600 W (perf_sim's all-in busy power), 3.2 pJ/bit host access energy; H200 HBM3e 4.8 TB/s |

PIC statics (`perf_sim_default_nmp`, `h200`) come from Wang et al., "Silicon
Photonics Chip I/O for Ultra High-Bandwidth and Energy-Efficient Die-to-Die
Connectivity", IEEE CICC 2024, Table 1 (with substrate undercut): modulators 206
+ filters 59.6 + (de-)interleavers 28 fJ/b of thermal tuning at the paper's
1.024 Tb/s link = 0.30 W per link, × 63 Tx/Rx link pairs per node = 18.94 W.
Heater power is per ring, so it does not scale with PanAgretti's 32 Gb/s channels;
the comb source is off-panel and excluded. This is 6.3× the ~3 W PanAgretti
assumes (citing a later TCPMT 2025 paper); without undercut Table 1 gives 35.35 W.
Table 1 is thermal tuning only, so traffic-proportional optical energy (`e_io`,
`e_pic`, `e_link`) stays a 0 placeholder and the `a2a`/`tp` categories are 0 J.

`e_flop` for a GPU, when omitted, is derived so that full effective compute plus
full effective HBM bandwidth draws `tdp_w` (the package: xpu + hbm).

## Real routing

`power_model/routing.py` loads decode rows only (`[input_tokens_total,
row_end)` of each round) from `moe_routing` (Qwen3-235B, 20 SWE-bench sessions,
47,376 rows). Each capture file is the final call of a segment, so earlier
rounds' decode rows are read back from that array at the same positions. Each
simulated session replays a contiguous stream from a seeded offset. This keeps
expert popularity and the 0.44 token-to-token reuse, while batching comes from the
simulator. Ownership comes from `mapping.yaml`. Routing changes energy attribution
only: simulator output is identical with routing on or off.

Timing stays consistent through the generator. `power_model/skew.py` measures
the real FFN time `Σ_l max_d t(x_ld, k_ld)` from i.i.d. decode rows and writes
per-device curves so the simulator's equal-routing barrier returns that time. The
construction is exact at every batch up to 512, and beyond it whenever each device
owns at most 8 contiguous experts (EP ≥ 16 with the generator's placement); it is
within 0.6% otherwise. Past `--skew-max-batch` the curves extrapolate with the
last measured straggler slope. The report's `max_abs_error_*` fields check the
construction in-sample; an out-of-sample re-measurement agrees within ±1.5%, and
the runtime `ffn_routed` timing ratio is the independent check. The balanced
roofline is 2.3× too slow at batch 1 and 3.4× too fast at batch 4096.

## Sweeps (`scripts/power_sweep.py`)

A panel is P H200 chips (TP groups) plus N NMP nodes of S perf_sim-default stacks,
for any registry model (`--model`, default `qwen3-235b`; `glm-4.5` has been run).
The prefill TP is the smallest power of two whose HBM holds the FP16 weights (TP4
for Qwen3-235B's 468 GB, TP8 for GLM-4.5's 702 GB), and node KV capacity is DRAM
minus the replicated attention projections and the node's EP share of dense,
shared and routed expert weights. Runs without `--routing-traces` (and every model
but Qwen3-235B, which alone has traces) use the even expert split and balanced FFN
curves; compare models only with the same mode, since skew-aware and balanced curves
differ by 2–3× in FFN time depending on batch.
Node attention runs at S × one stack's measured rate (KV striped over the node's
banks, as in PanAgretti). Node FFN uses perf_sim's own multi-stack convention,
expert tensor parallelism S with the input all-gather, measured per S (about half
of perfect S-fold scaling). KV capacity is node DRAM minus the replicated
attention weights and the node's share of experts.

- `grid`: B sessions all at context C prefill P tokens and decode O tokens
  together. It reports the steady window with all B active (TPOT, throughput,
  per-die power, J/token), the prefill phase (TTFT, GPU power), and marks cells
  whose KV cannot be resident at once as infeasible.
- `splits`: GPU:stack ratios and node sizes on the TraceLab Codex workload, run
  to completion; speedups are relative to `--baseline` (default 1 H200 : 6 stacks).

## Outputs

- `activity.npz`: `activity[T, device, 7]`, by-category tallies, pool bytes. Re-price with `--from-activity`.
- `power.npz`: `t_edges_s`, `power_w[T, die]` (float32), die names, `static_w`,
  `energy_j[die, category + static]` (whole horizon) and `meta_json` (coefficients,
  provenance, placeholders).
- `dies.csv`, `chips.csv`: mean, p95 and peak power after `warmup_s`; energy columns cover the whole horizon.
- `run_summary.json`: simulator summary, energy totals, timing ratios, checks, routing statistics.

## Checks (in `run_summary.json`)

- `conservation`: binned energy = category tallies; power × time = dynamic + static.
- `io_balance`: on-panel bytes sent = received, except pool traffic.
- `completeness` (`--check-completeness`): the collector saw every reservation the simulator made.
- `parity` (`--check-parity`): the simulation is identical with the plain collector.
- `stall`: no session left waiting for KV capacity; longest decode step; admission deadlock releases.
- `anchors`: full-load die powers. The paper's DRAM 58.9 W, logic 344.7 W and PIC
  3 W are enforced only for `panagretti_nmp`; other presets report the deviation.
  The GPU package check is an identity when `e_flop` is derived and is then not
  counted; `anchors.ok` is `null` when no independent anchor applies.
- `timing_ratio_simulated_over_activity`: `prefill` is 1.0 by construction;
  `ffn_routed` ≈ 1.0 with skew-aware curves; `attn` shows the linear-context
  attention model's error (below); in equal mode `ffn` differs from 1 by design.

## Limitations

- **Static power dominates and is mostly placeholder.** In the 300-session H200
  run static is ~60% of energy; only the PIC heaters are sourced. Logic leakage
  (10% of the die), DRAM background, EIC drivers, GPU idle and HBM background are
  placeholders. PIC heater power also falls as the PIC warms (≈0.7 W/K per node),
  which a thermal loop should close.
- **Attention timing.** The simulator's attention time is linear in context, but
  real time is affine (projection weights plus KV per token). Energy per step is
  exact; power in short-context bins can be high by up to 1/`attn p5`.
- **Prefill.** Latency is one roofline over the whole batch (full overlap of
  compute-bound attention and memory-bound expert streaming): about 12% optimistic
  on the chip peak. Expert coverage assumes uniform routing (real prefill routing
  hits ~0.6 of experts at 181 tokens); run-level prefill energy is ~0.4% high.
  `act` is the estimator's activation term; on a GPU it is priced as HBM traffic,
  which undercounts unfused kernels by a few W on the hbm die.
- **HBM energy placement.** All of `e_hbm` (including PHY/controller) is on the hbm die.
- **Placement.** `least_active_context` breaks ties by lowest device id, so
  low-id NMP nodes carry more attention: a spatial gradient that follows numbering.
- **All-to-all bytes** count every remote (token, expert) copy (~4% above
  per-destination dedup at EP56, 20-50% at EP 16/8). Equal-routing FFN energy uses
  the simulator's deterministic loads and is a biased reference.
- **Capacity.** One perf_sim-default node (4 stacks) holds ~602K Qwen3 KV tokens;
  `power_trace.py` truncates sessions whose context would exceed it (`session_fit`).
  A 1M-token context needs ≥ 8-stack nodes or FP8 KV. The global KV P2P resource is
  a single FCFS queue.
- **Data scope.** Routing traces cover Qwen3-235B only, captured at batch 1, with
  no gate weights. Other models fall back to equal routing.
- **Not modelled:** LM head, embeddings and router (~0.3% of prefill FLOPs, ~1-2%
  of NMP decode dynamic energy at 100K+ contexts, up to ~5% at 8K); traffic
  between the dies of one NMP node; TP all-reduce time (its energy is included);
  temperature-dependent leakage (static is constant).
