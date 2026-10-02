# Power traces for thermal simulation

Turns a PanAgretti simulation into per-die power over time (10 ms bins) for a panel of
GPU prefill chips and NMP decode nodes. Output: `power.npz` with `power_w[t, die]`,
one column per die (`nmp017.logic`, `nmp017.dram0..3`, `nmp017.pic`, `gpu0_3.xpu`,
`gpu0_3.hbm`, `gpu0_3.pic`). This is the input for a thermal model (3D-ICE); no
thermal modelling happens here.

Detailed docs: [docs/power_model.md](../docs/power_model.md) (how it works),
[docs/power_takeaways.md](../docs/power_takeaways.md) (results),
[docs/power_audit.md](../docs/power_audit.md) (what was verified).

## What was added (new files, no effect on PanAgretti)

| File | Role |
|---|---|
| `power_model/collector.py` | Observes simulator events, counts FLOPs and bytes per device |
| `power_model/binner.py` | Spreads each event's counters into fixed time bins |
| `power_model/pricing.py` | Counters × energy coefficients + static power → watts per die |
| `power_model/coefficients.py` | Coefficient presets with provenance (perf_sim fit, Wang et al. PIC, placeholders) |
| `power_model/routing.py`, `skew.py` | Real Qwen3 expert routing (energy placement) and matching FFN timing curves |
| `power_model/checks.py`, `simulate.py`, `layout.py` | Invariant checks, run wrapper, die layout |
| `roofline_model/activity.py` | FLOP/byte counters per operation, reusing the roofline estimator's math |
| `scripts/power_trace.py` | Run one trace: `activity.npz`, `power.npz`, `dies.csv`, `chips.csv`, `run_summary.json` |
| `scripts/power_sweep.py` | Batch × context grid and GPU:NMP split study |
| `scripts/power_crosscheck.py` | Fit NMP coefficients against perf_sim kernels (run once) |
| `scripts/power_report.py` | Figures and tables for the takeaways |
| `configs/power/*.yaml` | Coefficient configs (`panagretti_b200`, `h200_perf_sim`, `h200_perf_sim_glm45`) and generator inputs |
| `tests/test_power_*.py`, `tests/test_roofline_activity.py` | Tests |

## What changed in PanAgretti itself

Defaults are unchanged (golden tests). Everything below is opt-in or fixes a crash/hang.

1. **New prefill latency model** `roofline_context` (selected by config): prefill time
   depends on new tokens *and* context, so simulated time matches the energy model.
2. **`AgenticRuntime(prefill_transfers_at_completion=True)`** (opt-in; the power scripts
   use it): books prefill → decode KV transfers when prefill finishes instead of at
   launch. The default booking stalled all decode for seconds behind future transfers
   (2.4× lower throughput); `--legacy-prefill-transfers` reproduces the old behaviour.
3. **Two fixes on paths that previously failed:** attention admission no longer crashes
   on a waiter whose ready time is in the future, and no longer deadlocks when every
   node is full of cached sessions. Runs that completed before are bit-identical.
4. **Generator flags** (`--prefill-devices`, `--prefill-tp`, `--prefill-model`,
   `--attn-proj-reference-batch`, `--routing-traces`): default output is byte-identical.

## Commands

```bash
# 0. (once) fit NMP coefficients and node rates against perf_sim
python scripts/power_crosscheck.py --geometry perf_sim_default --stacks-per-node 4 --measure-rates \
    --out results/power/perf_sim_crosscheck_default.json

# 1. build a panel config: 8 H200 prefill chips + 56 NMP nodes, Qwen3-235B, real routing
python scripts/generate_config.py --input-yaml configs/power/h200_perf_sim_generator.yaml \
    --routing-traces ../moe_routing --force

# 2. simulate 300 Codex sessions for 120 s and write the power trace
python scripts/power_trace.py --config-dir configs/power/h200_perf_sim --power configs/power/h200_perf_sim.yaml \
    --sessions-json configs/glassml/workload/sessions.json --num-sessions 300 --arrival-rate 4 --seed 11 \
    --horizon 120 --routing traces --check-parity --check-completeness --out results/power/h200_perf_sim_traces

# 3. re-price an existing run with edited coefficients (no re-simulation)
python scripts/power_trace.py --from-activity results/power/h200_perf_sim_traces/activity.npz \
    --config-dir configs/power/h200_perf_sim --power my_power.yaml --out results/power/repriced

# 4. sweeps on any GPU : NMP split (--model glm-4.5 for GLM; TP is chosen automatically)
python scripts/power_sweep.py grid   --gpus 16 --nodes 16 --stacks 6 --out results/power/grid_1to6_16x6
python scripts/power_sweep.py splits --no-routing --jobs 4 --out results/power/splits

# 5. figures and tables
python scripts/power_report.py --out docs/figures/power_takeaways
```

Set `OMP_NUM_THREADS=1` for the sweeps. Tests: `pytest tests/test_power_model.py tests/test_roofline_activity.py`
(`tests/test_power_routing.py` needs `../moe_routing`).

Inputs not in the repository: `configs/glassml/workload/sessions.json` (46 MB, the
TraceLab Codex sessions; rebuild with `scripts/build_workload.py` from the trace) and
`../moe_routing` (the captured Qwen3-235B routing). Without the routing directory,
use `--routing equal` and omit `--routing-traces`.

## Read the output

- `power.npz`: `t_edges_s` (bin edges), `power_w[t, die]`, `die_names`, `static_w` per die,
  `energy_j[die, category]`, `meta_json` (coefficients, provenance, placeholders).
- `dies.csv` / `chips.csv`: mean, p95, peak power per die and per chip.
- `run_summary.json`: simulator summary, energy by category, timing ratios, checks.

## Caveats in one breath

Static power (leakage, DRAM background, optical drivers, GPU idle) is placeholder except
the PIC heaters (Wang et al. CICC 2024, 18.9 W per node); the H200 numbers come from
perf_sim's H100 entry; real routing exists for Qwen3-235B only (other models use the even
split); the power scripts rely on the opt-in transfer booking (item 2 above).
