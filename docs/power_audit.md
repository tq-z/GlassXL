# Power-trace implementation audit (2026-09-28)

A close review of everything behind the per-die power traces: the activity
counters, the simulator-core changes, the config generator, the power pipeline,
real routing, the coefficients and the produced traces. Eight reviewers each
owned one slice and listed both what they verified correct and what they found
wrong. An independent skeptic then tried to refute every medium or high finding
(one per finding) and every low finding (in a batch per slice), with reproducing
scripts. Raw results: `results/power/audit/audit_verified.json` (61 findings, 103
verified-correct items; local, `results/` is not tracked).

Status legend:
- **fixed**: changed in the code, with tests where practical.
- **fixed (power runs)**: resolved by `AgenticRuntime(prefill_transfers_at_completion=True)`, which `power_model.simulate` and `scripts/power_trace.py` turn on. The simulator default is unchanged.
- **documented**: a real modelling limit, stated in `docs/power_model.md` and the provenance strings.
- **open**: known and not yet addressed; the impact on the traces is small.

## Scorecard

| Slice | Verified correct | High | Medium | Low | Refuted |
|---|---|---|---|---|---|
| Activity counters (`roofline_model/activity.py`) | 10 | 0 | 0 | 6 | 1 |
| Simulator core (prefill model, admission) | 12 | 1 | 0 | 3 | 0 |
| Config generator | 14 | 0 | 0 | 6 | 0 |
| Binner / pricing / layout / checks | 14 | 0 | 3 | 9 | 0 |
| Collector / simulate / power_trace | 11 | 1 | 1 | 6 | 0 |
| Real routing + skew curves | 12 | 0 | 0 | 6 | 0 |
| Coefficients + perf_sim crosscheck | 16 | 0 | 1 | 9 | 0 |
| End-to-end numeric audit | 14 | 1 | 1 | 6 | 0 |
| **Total** | **103** | **3** | **6** | **51** | **1** |

Severity is after verification. Medium findings that verification downgraded to low (9) are counted as low.
The three high findings are the same defect, found independently by three reviewers.

## What was wrong, and what happened to it

### High

| Finding | Status |
|---|---|
| **Decode stalls behind KV P2P transfers booked ahead of the clock.** At prefill launch the simulator submits attention admissions and books each session's KV P2P transfer with the compute finish as a future ready time. The reservation also occupies the destination NMP device's FCFS queue (no backfill), so every decode step issued before the transfer starts waits behind it, and the lock-step decode barrier stalls all nodes. In the 300-session H200 run: steps up to 21.8 s, 62-79% of the 120 s inside stall windows, NMP decode throttled about 2x, GPU prefill load about 30-40% low. Pre-existing simulator behaviour (the earlier retry fix made it reachable instead of crashing). | **fixed (power runs)**. New `AgenticRuntime(..., prefill_transfers_at_completion=True)`: a `PREFILL_COMPLETE` event books admissions and transfers at compute finish, and transfers leave through the prefill device's copy engine instead of queueing behind its reserved-ahead compute. Longest decode step in the same run: 67 ms. Throughput 1,070 → 2,582 tok/s; inter-token 51.5 → 8.3 ms. Regression test `test_prefill_transfers_at_completion_do_not_stall_running_decode`. `--legacy-prefill-transfers` reproduces the old booking; the simulator default is unchanged. |

### Medium

| Finding | Status |
|---|---|
| Some non-integer horizons (e.g. 16.01 s at 10 ms) create a zero-width last bin: a NaN row in `power_w` and failed conservation. | **fixed**: robust bin count in `binner._bin_count`; the no-horizon edge case (zero-length event on the final edge) is folded back in. |
| `anchors.ok` could never be False for the perf_sim/H200 presets: paper targets did not apply and the GPU check is an identity when `e_flop` is derived. | **fixed**: identity checks carry a note and no `ok`; `anchors.ok` is `null` when no independent anchor applies. |
| `docs/power_model.md` stale (GPU pic die, presets, anchors, capacity, fit residuals). | **fixed**: rewritten. |
| NMP node FFN rate assumed perfect 4x scaling over four perf_sim stacks; perf_sim's own multi-stack convention (expert TP = stacks, with input all-gather) runs compute-bound FFN about 2x slower. | **fixed**: node FFN rates measured in perf_sim at TP = S for S = 1, 2, 4, 6, 8 (`scripts/power_sweep.py NODE_FFN`, `configs/power/h200_perf_sim_generator.yaml`). Attention keeps S x stack (KV striped over the node's banks, as in PanAgretti). |
| Decode energy spread over [first attention start, combine finish], so stalled steps smeared energy over seconds. | **fixed**: window is now [latest attention start, combine finish] (real hardware interleaves layers behind a per-layer barrier); with the stall fixed, steps are milliseconds. |
| Traces are ~75% static, ~60% placeholder static; the NMP leakage placeholder (34.5 W) was scaled from the paper's logic die, twice the MACs of the perf_sim die. | **fixed / documented**: leakage placeholder is now 10% of this preset's full-load logic die (20.3 W; a 15% share would be ~36 W). Static is now ~59% of energy in the H200 run; the placeholder share is stated in the docs. |

### Low (verified real)

| Finding | Status |
|---|---|
| `retry()` could return admissions with non-monotonic ready times, reordering P2P transfers. | fixed (power runs): admissions are no longer future-dated. |
| LRU victims evicted at the release event for a waiter that becomes ready later. | fixed (power runs): same reason. |
| Retry tests thin. | fixed: added the FCFS deadlock regression test; the old test still documents the capacity case. |
| Stall check blind to mid-run deadlock and to stalled steps. | fixed: `stall` now reports `longest_decode_step_s` and `admission_deadlock_releases`. A new admission deadlock (all nodes full of protected cached copies, FCFS head blocked) is now resolved by releasing those copies (`test_drained_panel_releases_protected_copies_instead_of_deadlocking`). |
| Skew-curve tail past `--skew-max-batch` used the balanced slope (0.65x the real time at 2x the range). | fixed: extrapolates with the device's last measured straggler slope. |
| `act` docstring said "on-chip buffers" while GPU prices it as HBM traffic. | fixed (docstring). |
| `power_crosscheck.py` claimed perf_sim has no multi-stack notion. | fixed (docstring). |
| "~1%" for LM head/embedding/router overstated for decode. | fixed (doc): ~0.3% of prefill FLOPs, ~1-2% of NMP decode dynamic energy at 100K+ contexts, up to ~5% at 8K. |
| H200 preset pairs perf_sim's SXM-class 980 TFLOPS with its 600 W busy power. | documented in provenance: if the part is an H200 SXM (700 W), the prefill xPU die heat is ~17% (~100 W/chip) low. |
| PIC 18.94 W is 6.3x PanAgretti's ~3 W; w/o-undercut alternative 35.35 W not shown. | documented in provenance and docs. |
| `e_hbm` = 3.2 pJ/bit is perf_sim's HBM2e-class host access energy, used as an HBM3e proxy. | documented in provenance. |
| Fit residuals quoted on total energy understate the logic-die residual. | documented (logic-die residual up to 2.6%). |
| Heater power depends on temperature (~0.7 W/K per node). | documented. |
| Prefill expert coverage assumes uniform routing (real: ~0.6 at 181 tokens). | documented; run-level prefill energy +0.4%, HBM die +3-4%. |
| Prefill latency is one roofline over the whole batch (full overlap). | documented; ~12% on the chip-level peak, xPU peaks unchanged. |
| GPU `act` priced at HBM energy undercounts unfused kernels. | documented; +2-10 W on the hbm die during large prefills. |
| All GPU HBM energy (incl. PHY/controller) on the hbm die. | documented. |
| NMP drivers on the logic die, GPU drivers on the pic die. | documented (custom logic die integrates them; a stock H200 cannot). |
| Least-active-context placement ties break to the lowest device id → low-id hot gradient. | documented; a spatial pattern the thermal model will reproduce. |
| All-to-all bytes counted per (token, expert) copy. | documented (~4% above per-destination dedup at EP56). |
| Equal-routing FFN uses the simulator's deterministic loads (biased reference). | documented; real routing is the default for Qwen3. |
| GPU HBM traffic can exceed the effective bandwidth when KV transfers overlap prefill. | documented; with the copy engine, 40 of 96,000 chip-bins exceed the 600 W package TDP (max 620.6 W). |
| Skew thinning exact only for ≤ 8 contiguous experts per device; validation in-sample. | documented (exact for the shipped EP ≥ 16 configs). |
| Decode rows of earlier rounds come from the segment's final-call array. | documented. |
| Short-context attention runs faster than perf_sim (fixed ~0.3 µs kernel overhead; up to 1.8x at 1K). | documented. |
| Optical I/O energy is fully static (a2a/tp = 0 J). | documented: Wang Table 1 is thermal tuning only. |
| Operand fill charged on DMA reads; `conservation` power clause is an identity; `prefill_chips_per_device` not cross-checked with TP; energy columns ignore `warmup_s`; GEMM rate carries a per-expert tail; MLA `pair_flops` mixes decode/prefill widths; generator tests read the personal default YAML; roofline_report keeps balanced-routing sections; KV capacity uses the average expert footprint; `--from-activity` trusts the chip count. | open: each below 1% of the traces or latent (not triggered by the shipped configs). |
| Found after the audit: on a KV-starved panel (28 H200 + 5 × 8-stack nodes) the run still ended with sessions waiting (137 of 298 completed), a second admission stall mode beyond the one now released. | open: far from any useful operating point, but strict FCFS admission deserves a redesign. |
| Test gaps (new presets, collector attribution formulas, skew thinning/tail, row pointers). | partly fixed (pic die, deadlock, stall, preset/anchor and zero-width-bin tests); rest open. |

### Refuted

- "`ffn_layer_time` ignores `nonrouted_tokens`": the real-routing path never calls it; `layer_folded_time` already uses the even EP share.

## What is right (verified)

Condensed from the 103 verified-correct items; each was checked by an independent
script, by hand, or against the source papers.

- **Model counters.** Qwen3-235B coefficients recomputed from the published config match exactly: 71,303,168 projection params/layer, KV 192,512 B/token, `a_tok` 4.179e10, `a_pair` 3,080,192, `W_fix` 13.405 GB, `W_rt` 454.19 GB. Decode attention per session, per context token and projection-once-per-device; causal pair count `P·C + P(P+1)/2`; FFN terms vs `_ffn_layer_work` for every supported model; the simulator's `roofline_context` latency is bit-identical to the collector's over 20,000 random batches.
- **Simulator core.** Default configs bit-identical wherever the previous code could run; the retry fix cannot double-book KV capacity; FCFS order and `capacity_wait_s` correct; loader validation and the unchanged aggregate path.
- **Generator.** Default outputs byte-identical to the pre-change code; the H200 config regenerates byte-identically; prefill/decode ids, TP rates, deployment block, YAML/CLI parity; `equal_local_tokens` equals `ExpertMapping.equal_loads`.
- **Binner and pricing.** Overlap-fraction spreading conserves activity to 3.6e-15 over 20,000 fuzzed events; NMP and GPU joule matrices; `e_flop` derivation excludes the pic die; die/column mapping; dies.csv/chips.csv statistics.
- **Collector.** No double counting across the base-class call graph (P2P, pool and TP bytes reproduce the simulator token counters exactly); P2P/pool bytes on the right devices; TP all-reduce `2(c-1)/c`; all-to-all copies; pre-step contexts and one KV append per token; parity and `--from-activity` bit-identical.
- **Routing.** Decode rows `[input_tokens_total, row_end)` are exactly the generated tokens for all 350 rounds; ownership agrees with the generator and the simulator; `layer_device_loads`; all-to-all remote-copy logic; `layer_folded_time` equals the estimator to 7e-16; the skew construction is exact at every batch 1-4096 for the H200 EP56 placement and agrees with an out-of-sample re-measurement within ±1.5%.
- **Coefficients.** Wang et al. Table 1 → 0.3006 W per 64-channel link → 18.94 W per node: converting at Wang's 1.024 Tb/s (not 2.048) is right because heater power is per ring; ×63 is right because the per-link figure covers Tx and Rx; excluding the off-panel comb is right; treating heaters as static is right to first order. perf_sim default geometry (65,536 MACs/stack, 24.2 TB/s tier-averaged, 32 GiB), the harmonic tier average, the e_mac/e_operand-fill fit (e_fill 0.01655 vs perf_sim's RF write 0.01679 pJ/bit), `e_dram` 0.4291, derived H200 `e_flop` 0.6246 pJ/FLOP.
- **End to end.** Per-die ΣP·dt = energy_j to 2.8e-8; no NaN, negative or sub-static bins; physical ceilings hold (GPU package ≤ TDP in the audited runs, NMP dies well below full-load power); decode attention, FFN (equal and real routing), prefill and transfer energies hand-computed from first principles match the collector with relative error 0; deterministic outputs.

---

# Addendum (2026-10-01): GLM-4.5 as a second model

Three reviewers (model spec, model-generic code paths, produced numbers) with one
refuter per medium/high finding. Raw results: `results/power/audit/audit_glm45.json`.

| Slice | Verified correct | High | Medium | Low |
|---|---|---|---|---|
| GLM-4.5 spec and derived quantities | 14 | 0 | 1 | 2 |
| Model-generic code paths (dense + shared layers, 160 experts, TP8) | 11 | 0 | 2 | 4 |
| Produced numbers vs Qwen | 12 | 1 | 1 (+1 downgraded to low) | 6 |

**Verified correct (highlights).** Registry entry matches the published config field
by field; every derived constant (KV 376,832 B/token, 702.34 GB FP16, a_tok 6.40e10,
a_pair 4.52e6, per-node resident weights 37.18 GB, capacities 266K/368K/513K tokens)
reproduces by hand and byte-for-byte in every generated config; `required_tp` 8/4;
the Qwen path is unchanged (byte-identical regeneration); dense/shared branches,
160-expert placement, PowerCollector invariants and prefill/attention/FFN energies
reproduce from first principles on a tiny GLM run; all trace checks pass; physical
ceilings hold; the split TP skip/dedup logic behaves as intended.

| Finding | Status |
|---|---|
| **High.** The GLM `1to4_16x4` split drained its calendar with 96 sessions stuck (134 of 231 completed, 10 deadlock releases) yet `splits.json` reported it as the fastest config (1.83×) — the open admission-stall mode, now hit by a shipped sweep. | **fixed**: `power_report` flags stalled runs and excludes them from the common session set (which also corrected the Qwen table: 286 sessions instead of 131); `split_run` now records a `stall` block (waiters, deadlock releases, longest step). The stall itself remains open. |
| **Medium.** Dense-layer and shared-expert weights are EP-sharded with no collective (estimator convention); a DP-attention deployment replicates them per node. First exposed by GLM (Qwen has neither). Node KV capacity ~5% high; under replication the 256K row and the 128K × 64 cell of the 56 × 4 grid would not fit; per-node FFN traffic +42%, ~60 µs/step; J/token ≤ 3%. | documented (§5.4 of the takeaways); changing it is a one-line estimator decision. |
| **Medium.** GLM grids (balanced FFN curves, even split) were compared with the shipped Qwen grids (skew-aware curves, real routing). The balanced curve is slower at batch ≤ 8 and up to 21–31% faster at batch 256; FFN energy under the even split is 1.6× the real-routing figure. My first write-up misattributed the FFN-share difference to GLM's dense/shared layers and inverted the bias direction. | **fixed** in the takeaways: like-for-like Qwen-balanced column added; FFN rows compared even split to even split. |
| Medium → low. Split speedups and J/token compare different session populations (per-config capacity fitting) and makespans. | documented; the common-set metric addresses the population part. |
| Low. All-to-all launch latency counts 92 layers while energy counts 89 MoE layers (generator uses `num_layers`): GLM TPOT 0.12 ms (1–3%) high. | documented; left because the generator's goldens would change. |
| Low. Capacity uses the average experts per node (nodes with 3 experts have 0.6–1.4% less room). | documented. |
| Low. perf_sim rate constants were measured on Qwen shapes (GQA 16, 4096 × 1536 experts). | documented. |
| Low. Linear-context attention timing error is larger for GLM (p5 0.76 vs 0.85). | documented. |
| Low. GLM trace carries 11% of its energy past the horizon in one in-flight 24 s prefill batch (single TP8 group). | documented. |
| Low. Hot-spot ratio 1.72× median node (placement tie-break). | documented. |
| Low. GPU-heavy GLM splits are KV-starved; a terse refusal for too-small panels; misleading error when traces are applied to GLM. | open (usability). |
