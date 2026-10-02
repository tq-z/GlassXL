#!/usr/bin/env python3
"""Cross-check the analytical NMP energy model against perf_sim (Stratum) kernels.

Runs perf_sim's kernels at one of two geometries and compares each kernel's
DRAM and non-DRAM energy with this repository's activity x coefficient model on
identical work:

  * ``panagretti`` (default): the PanAgretti paper node as ONE perf_sim stack
    (16 PUs x 32 matrix units x 32x32 MACs = 524,288 MACs at 1 GHz, 512 banks,
    ~64 TB/s);
  * ``perf_sim_default``: perf_sim's unpatched DRAM3D/config.py stack (16 ch x
    16 banks x 16x16 MAC engines = 65,536 MACs at 1 GHz, 8 tRCD tiers), with a
    node made of ``--stacks-per-node`` such stacks.

It then fits, once, the effective coefficients that absorb what the analytical
model folds away (register files, SRAM, reducers, softmax units, ring NoC,
row-granularity DRAM reads) and prints a power.yaml ``nmp.overrides`` block that
freezes them. With ``--measure-rates`` it also measures the stack's effective
roofline rates (GEMM and attention MAC/s, streaming bytes/s averaged over the
DRAM tiers) and prints them per node for the config generator.

  python scripts/power_crosscheck.py --out results/power/perf_sim_crosscheck.json
  python scripts/power_crosscheck.py --geometry perf_sim_default --stacks-per-node 4 \
      --measure-rates --out results/power/perf_sim_crosscheck_default.json

perf_sim quirks handled:
  * fast=False everywhere (fast=True copies channel 0's energy to all channels);
  * pnm_interface is called directly: sweeps/panagretti/kernels.py chdirs on
    import and its caches ignore patched geometry;
  * projections are compared on the shape perf_sim models (it assumes
    q_dim == hidden, which is half of Qwen3-235B's projection);
  * kernels run on one perf_sim stack at expert/attention tensor parallelism 1
    (perf_sim's own multi-stack convention is TP = stacks; see
    scripts/power_sweep.py NODE_FFN for node-level FFN rates measured that way);
  * the NoC read x write product (noc.py:119) only affects nanojoules;
  * perf_sim's attention kernel has no KV-append write; neither does this comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Qwen3-235B-A22B shapes (roofline_model/specs.py).
HIDDEN = 4096
EXPERT_INTERMEDIATE = 1536
Q_HEADS = 64
KV_HEADS = 4
HEAD_DIM = 128
GQA = Q_HEADS // KV_HEADS

# PanAgretti node geometry expressed with perf_sim's knobs; 0.5 ns tRCD gives
# 512 banks x 4 KiB / tRC = 64 TB/s.
GEOMETRY = dict(tRCD_tiers=[0.5e-9] * 8, n_ch=16, n_bank1ch=32, n_dp_tap=32, n_dp_pe=32, clk_period=1e-9)
# name -> (perf_sim knobs, peak MAC/s of one stack, peak DRAM bytes/s of one stack)
GEOMETRIES = {
    "panagretti": (GEOMETRY, 5.24288e14, 64e12),
    # Unpatched perf_sim: 16 ch x 16 banks x 16 PEs x 16 taps at 1 GHz; peak bytes/s
    # is 256 banks x 4 KiB rows over the tier-averaged tRC (43.3 ns).
    "perf_sim_default": ({}, 16 * 16 * 16 * 16 * 1e9, 256 * 4096 / 43.33e-9),
}
ROWS_PER_TIER = 4096
N_TIERS = 8
E_DRAM_PERF_SIM_PJ = 0.429
E_MAC_PERF_SIM_PJ = 0.604
E_SRAM_PJ = 0.0094

ATTENTION_CONTEXTS = (1024, 4096, 16384, 65536, 131072)
FFN_TOKENS_PER_EXPERT = (1, 2, 4, 8, 16, 32, 64)
FFN_EXPERTS = (1, 2, 3)
PROJECTION_TOKENS = (1, 4, 16, 64)


def analytical_attention(context: int) -> dict[str, float]:
    return {
        "macs": 2.0 * Q_HEADS * HEAD_DIM * context,
        "dram_bytes": context * 2 * KV_HEADS * HEAD_DIM * 2.0,
        "act_bytes": 4.0 * HIDDEN * 2.0,
    }


def analytical_ffn(experts: int, tokens: int) -> dict[str, float]:
    matrix = HIDDEN * EXPERT_INTERMEDIATE
    return {
        "macs": 3.0 * experts * tokens * matrix,
        "dram_bytes": 3.0 * experts * matrix * 2.0,
        "act_bytes": 3.0 * experts * tokens * (HIDDEN + EXPERT_INTERMEDIATE) * 2.0,
    }


def analytical_projection(tokens: int, mode: str) -> dict[str, float]:
    columns = HIDDEN + 2 * HIDDEN // GQA if mode == "proj_qkv" else HIDDEN
    return {
        "macs": float(tokens * HIDDEN * columns),
        "dram_bytes": HIDDEN * columns * 2.0,
        "act_bytes": tokens * (HIDDEN + columns) * 2.0,
    }


def _import_pnm(perf_sim: Path):
    for path in (perf_sim, perf_sim / "sweeps"):
        sys.path.insert(0, str(path))
    import harness  # noqa: E402

    harness.ensure_repo_cwd()
    harness.install_vllm_stub()
    import DRAM3D.LogicDieSimulator.pnm_interface as pnm  # noqa: E402

    return harness, pnm


def _hardware(harness, knobs: dict):
    return harness.patched_hw(**knobs) if knobs else contextlib.nullcontext()


def run_kernels(perf_sim: Path, knobs: dict) -> list[dict]:
    harness, pnm = _import_pnm(perf_sim)

    rows: list[dict] = []

    def record(kernel: str, point: dict, work: dict, result) -> None:
        seconds, total, parts = result
        rows.append(
            {
                "kernel": kernel,
                **point,
                **work,
                "perf_sim_s": seconds,
                "perf_sim_dram_j": parts.e_dram,
                "perf_sim_nondram_j": total - parts.e_dram,
                "perf_sim_parts_j": {
                    name: getattr(parts, name)
                    for name in (
                        "e_dram", "e_matrix_rf_pe", "e_tensor_core_pe", "e_psum_sram_pe",
                        "e_reducer_pu", "e_spe_pu", "e_shared_sram_pu", "e_router",
                    )
                },
            }
        )

    with _hardware(harness, knobs):
        groups = [(16 // KV_HEADS, 1)] * KV_HEADS
        for context in ATTENTION_CONTEXTS:
            result = pnm.sim_attn_layer(
                "mixtral", HEAD_DIM, GQA, context, 0, groups,
                [(0,)] * KV_HEADS, [(0,)] * KV_HEADS,
                fast=False, report_detailed_energy=True,
            )
            record("attention", {"context_tokens": context}, analytical_attention(context), result)
        for experts in FFN_EXPERTS:
            for tokens in FFN_TOKENS_PER_EXPERT:
                expert_tokens = [(expert,) for expert in range(experts) for _ in range(tokens)]
                result = pnm.sim_moe_layer(
                    "mixtral", expert_tokens, [0] * experts, HIDDEN, EXPERT_INTERMEDIATE, 1, "silu",
                    fast=False, report_detailed_energy=True,
                )
                record(
                    "ffn",
                    {"experts": experts, "tokens_per_expert": tokens},
                    analytical_ffn(experts, tokens),
                    result,
                )
        for mode in ("proj_qkv", "proj_o"):
            for tokens in PROJECTION_TOKENS:
                result = pnm.sim_attn_proj_qkvo(
                    "mixtral", tokens, HIDDEN, GQA, 0, mode, 1, fast=False, report_detailed_energy=True,
                )
                record(mode, {"tokens": tokens}, analytical_projection(tokens, mode), result)
    return rows


def measure_rates(perf_sim: Path, knobs: dict) -> dict:
    """Effective roofline rates of one stack, with data spread evenly over the tiers.

    Streaming bytes/s comes from batch-1 GEMVs (weights read once, memory-bound)
    placed at each tier's first row; the tier average is harmonic because time,
    not bandwidth, adds up. MAC/s comes from compute-bound kernels: FFN at 1024
    tokens per expert (GEMM) and attention at 128K context (GQA 16).
    """

    harness, pnm = _import_pnm(perf_sim)
    matrix = HIDDEN * EXPERT_INTERMEDIATE
    qkv_columns = HIDDEN + 2 * HIDDEN // GQA
    tiers: list[dict] = []
    with _hardware(harness, knobs):
        for tier in range(N_TIERS):
            row = tier * ROWS_PER_TIER
            ffn_s, *_ = pnm.sim_moe_layer(
                "mixtral", [(0,)], [row], HIDDEN, EXPERT_INTERMEDIATE, 1, "silu", fast=False,
                report_detailed_energy=True,
            )
            qkv_s, *_ = pnm.sim_attn_proj_qkvo(
                "mixtral", 1, HIDDEN, GQA, row, "proj_qkv", 1, fast=False, report_detailed_energy=True,
            )
            tiers.append(
                {
                    "tier": tier,
                    "ffn_batch1_bytes_per_s": 3.0 * matrix * 2.0 / ffn_s,
                    "qkv_batch1_bytes_per_s": HIDDEN * qkv_columns * 2.0 / qkv_s,
                }
            )
        ffn_s, *_ = pnm.sim_moe_layer(
            "mixtral", [(0,)] * 1024, [0], HIDDEN, EXPERT_INTERMEDIATE, 1, "silu", fast=False,
            report_detailed_energy=True,
        )
        groups = [(16 // KV_HEADS, 1)] * KV_HEADS
        attention_s, *_ = pnm.sim_attn_layer(
            "mixtral", HEAD_DIM, GQA, 131072, 0, groups, [(0,)] * KV_HEADS, [(0,)] * KV_HEADS,
            fast=False, report_detailed_energy=True,
        )
    samples = [value for tier in tiers for value in (tier["ffn_batch1_bytes_per_s"], tier["qkv_batch1_bytes_per_s"])]
    return {
        "tiers": tiers,
        "bytes_per_s": len(samples) / sum(1.0 / value for value in samples),
        "gemm_macs_per_s": 3.0 * matrix * 1024 / ffn_s,
        "attention_macs_per_s": 2.0 * Q_HEADS * HEAD_DIM * 131072 / attention_s,
    }


def _fit_logic(rows: list[dict]) -> tuple[float, float]:
    """Least-squares (relative error) fit of on-logic-die energy.

    nondram - act*8*e_sram = e_mac*MACs + e_fill*8*dram_read_bytes

    e_fill is the per-bit cost of writing each streamed weight/KV operand into
    a PE register file; it is why small-batch GEMVs cost more per MAC.
    """

    import numpy as np

    design = np.array(
        [
            [row["macs"] * 1e-12, row["dram_bytes"] * 8 * 1e-12]
            for row in rows
        ]
    )
    target = np.array(
        [row["perf_sim_nondram_j"] - row["act_bytes"] * 8 * E_SRAM_PJ * 1e-12 for row in rows]
    )
    weights = 1.0 / target
    solution, *_ = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)
    return float(solution[0]), float(solution[1])


def fit(rows: list[dict], peak_macs_per_s: float, peak_bytes_per_s: float, rates: dict | None = None) -> dict:
    for row in rows:
        row["dram_ratio"] = row["perf_sim_dram_j"] / (row["dram_bytes"] * 8 * E_DRAM_PERF_SIM_PJ * 1e-12)
        row["effective_pj_per_mac"] = (
            (row["perf_sim_nondram_j"] - row["act_bytes"] * 8 * E_SRAM_PJ * 1e-12) / row["macs"] * 1e12
        )
        row["roofline_s"] = max(row["macs"] / peak_macs_per_s, row["dram_bytes"] / peak_bytes_per_s)
        if rates is not None:
            macs_per_s = rates["attention_macs_per_s"] if row["kernel"] == "attention" else rates["gemm_macs_per_s"]
            row["effective_roofline_s"] = max(row["macs"] / macs_per_s, row["dram_bytes"] / rates["bytes_per_s"])
            row["latency_ratio_perf_sim_over_effective_roofline"] = row["perf_sim_s"] / row["effective_roofline_s"]
    gemm = [row for row in rows if row["kernel"] != "attention"]
    attention = [row for row in rows if row["kernel"] == "attention"]
    dram_overhead = statistics.median(row["dram_ratio"] for row in rows)
    e_mac, e_fill = _fit_logic(rows)
    e_mac_gemm = e_mac_attn = e_mac
    for row in rows:
        dram = row["dram_bytes"] * 8 * E_DRAM_PERF_SIM_PJ * dram_overhead * 1e-12
        nondram = (
            row["macs"] * e_mac * 1e-12
            + row["dram_bytes"] * 8 * e_fill * 1e-12
            + row["act_bytes"] * 8 * E_SRAM_PJ * 1e-12
        )
        row["fitted_dram_residual"] = dram / row["perf_sim_dram_j"] - 1
        row["fitted_nondram_residual"] = nondram / row["perf_sim_nondram_j"] - 1
        row["fitted_total_residual"] = (dram + nondram) / (row["perf_sim_dram_j"] + row["perf_sim_nondram_j"]) - 1
        row["latency_ratio_perf_sim_over_roofline"] = row["perf_sim_s"] / row["roofline_s"]
    worst = {
        kernel: {
            "dram": max(abs(row["fitted_dram_residual"]) for row in rows if row["kernel"] == kernel),
            "nondram": max(abs(row["fitted_nondram_residual"]) for row in rows if row["kernel"] == kernel),
            "total": max(abs(row["fitted_total_residual"]) for row in rows if row["kernel"] == kernel),
        }
        for kernel in sorted({row["kernel"] for row in rows})
    }
    timing = {}
    if rates is not None:
        for kernel in sorted({row["kernel"] for row in rows}):
            ratios = [row["latency_ratio_perf_sim_over_effective_roofline"] for row in rows if row["kernel"] == kernel]
            timing[kernel] = {"min": min(ratios), "median": statistics.median(ratios), "max": max(ratios)}
    return {
        "effective_roofline_timing_ratio_by_kernel": timing,
        "dram_overhead_vs_0.429_pj": dram_overhead,
        "e_mac_gemm_pj": e_mac_gemm,
        "e_mac_attn_pj": e_mac_attn,
        "e_operand_fill_pj_per_bit": e_fill,
        "single_parameter_pj_per_mac": {
            "gemm_median": statistics.median(row["effective_pj_per_mac"] for row in gemm),
            "attention_median": statistics.median(row["effective_pj_per_mac"] for row in attention),
        },
        "unfitted_e_mac_pj": E_MAC_PERF_SIM_PJ,
        "worst_abs_residual_by_kernel": worst,
        "checkpoint_d": {
            "dram_within_5pct": all(item["dram"] <= 0.05 for item in worst.values()),
            "total_within_10pct": all(item["total"] <= 0.10 for item in worst.values()),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perf-sim", type=Path, default=REPOSITORY_ROOT.parent / "perf_sim")
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "results" / "power" / "perf_sim_crosscheck.json")
    parser.add_argument("--geometry", choices=sorted(GEOMETRIES), default="panagretti")
    parser.add_argument("--stacks-per-node", type=int, default=1, help="perf_sim stacks in one decode node")
    parser.add_argument("--measure-rates", action="store_true", help="also measure effective roofline rates")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    perf_sim = args.perf_sim.resolve()
    knobs, peak_macs, peak_bytes = GEOMETRIES[args.geometry]

    rows = run_kernels(perf_sim, knobs)
    rates = measure_rates(perf_sim, knobs) if args.measure_rates else None
    summary = fit(rows, peak_macs, peak_bytes, rates)
    overrides = {
        "e_mac_gemm_pj": round(summary["e_mac_gemm_pj"], 4),
        "e_mac_attn_pj": round(summary["e_mac_attn_pj"], 4),
        "e_operand_fill_pj_per_bit": round(summary["e_operand_fill_pj_per_bit"], 5),
        "e_dram_pj_per_bit": round(E_DRAM_PERF_SIM_PJ * summary["dram_overhead_vs_0.429_pj"], 4),
    }
    node = None
    if rates is not None:
        stacks = args.stacks_per_node
        node = {
            "stacks_per_node": stacks,
            "peak_macs_per_s": peak_macs * stacks,
            "peak_dram_bytes_per_s": peak_bytes * stacks,
            "effective_dram_bytes_per_s": rates["bytes_per_s"] * stacks,
            "decode_ffn_device_flops_per_second": 2.0 * rates["gemm_macs_per_s"] * stacks,
            "decode_attn_device_flops_per_second": 2.0 * rates["attention_macs_per_s"] * stacks,
            "efficiency_gemm": rates["gemm_macs_per_s"] / peak_macs,
            "efficiency_attention": rates["attention_macs_per_s"] / peak_macs,
        }
    document = {
        "date": datetime.date.today().isoformat(),
        "perf_sim": str(perf_sim),
        "geometry_name": args.geometry,
        "geometry": {key: value for key, value in knobs.items() if key != "tRCD_tiers"}
        | ({"tRCD_s": knobs["tRCD_tiers"][0]} if "tRCD_tiers" in knobs else {"tRCD_s": "perf_sim tiers"}),
        "model": "qwen3-235b (one layer, one perf_sim stack)",
        "fit": summary,
        "suggested_power_yaml_nmp_overrides": overrides,
        "rates_per_stack": rates,
        "node": node,
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps({"fit": summary, "nmp_overrides": overrides, "node": node}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
