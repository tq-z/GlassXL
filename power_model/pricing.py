"""Price binned activity into per-die power and write the output files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .binner import ACTIVITY_KINDS, CATEGORIES, KIND_INDEX, BinnedActivity
from .coefficients import PowerConfig
from .layout import DeviceLayout, Die

PJ = 1e-12


@dataclass(frozen=True)
class PowerTrace:
    t_edges_s: np.ndarray
    dies: tuple[Die, ...]
    power_w: np.ndarray
    static_w: np.ndarray
    energy_j: np.ndarray
    beyond_horizon_j: np.ndarray
    meta: dict[str, Any]

    @property
    def die_names(self) -> tuple[str, ...]:
        return tuple(die.name for die in self.dies)

    @property
    def horizon_s(self) -> float:
        return float(self.t_edges_s[-1])

    def dynamic_energy_j(self) -> float:
        widths = np.diff(self.t_edges_s)
        return float(((self.power_w - self.static_w[None, :]) * widths[:, None]).sum())


def gpu_effective_rates_per_chip(
    power: PowerConfig, report: dict[str, Any] | None
) -> tuple[float, float, str]:
    """Effective FLOP/s and HBM bytes/s of one prefill chip, as the simulator runs it."""

    deployment = (report or {}).get("deployment") or {}
    hardware = deployment.get("prefill_device_hardware")
    tp = deployment.get("prefill_tp")
    if hardware and tp:
        return (
            float(hardware["flops_per_second"]) / int(tp),
            float(hardware["memory_bandwidth_bytes_per_second"]) / int(tp),
            "roofline_report.yaml deployment",
        )
    return (
        float(power.gpu["flops_per_second"]),
        float(power.gpu["hbm_bytes_per_second"]),
        "gpu preset peak rates (no roofline report)",
    )


def resolve_e_flop_pj(power: PowerConfig, report: dict[str, Any] | None) -> tuple[float, str]:
    """Energy per FLOP; derived when omitted so that full compute + full HBM draws TDP.

    Calibrating at the roofline ridge (both resources saturated) keeps every
    modelled operating point at or below TDP; compute-only work then draws less.
    """

    if power.gpu.get("e_flop_pj") is not None:
        return float(power.gpu["e_flop_pj"]), power.provenance.get("gpu.e_flop_pj", "power.yaml")
    flops, hbm, source = gpu_effective_rates_per_chip(power, report)
    hbm_w = 8.0 * power.gpu["e_hbm_pj_per_bit"] * PJ * hbm
    budget = power.gpu["tdp_w"] - power.gpu["p_xpu_static_w"] - power.gpu["p_hbm_static_w"] - hbm_w
    if budget <= 0.0:
        raise ValueError(
            "gpu tdp_w must exceed static power plus full-bandwidth HBM power to derive e_flop_pj"
        )
    return budget / flops / PJ, (
        "derived: (tdp_w - static - HBM power at effective bandwidth) / effective FLOP/s "
        f"per chip ({source})"
    )


def device_coefficients(
    kind: str,
    chips: int,
    dram_dies: int,
    power: PowerConfig,
    e_flop_pj: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Joules per unit activity [7, components] and static watts [components]."""

    k = KIND_INDEX
    if kind == "nmp":
        nmp = power.nmp
        components = dram_dies + 2
        matrix = np.zeros((len(ACTIVITY_KINDS), components))
        logic, pic = dram_dies, dram_dies + 1
        for die in range(dram_dies):
            matrix[k["mem_rd"], die] = 8.0 * nmp["e_dram_pj_per_bit"] * PJ / dram_dies
            matrix[k["mem_wr"], die] = 8.0 * nmp["e_dram_pj_per_bit"] * PJ / dram_dies
        matrix[k["flop_gemm"], logic] = nmp["e_mac_gemm_pj"] * PJ / 2.0
        matrix[k["flop_attn"], logic] = nmp["e_mac_attn_pj"] * PJ / 2.0
        matrix[k["act"], logic] = 8.0 * nmp["e_sram_pj_per_bit"] * PJ
        # DRAM reads also stream operands (weights, KV) into the PE register files.
        matrix[k["mem_rd"], logic] = 8.0 * (
            nmp["e_dram_interface_pj_per_bit"] + nmp["e_operand_fill_pj_per_bit"]
        ) * PJ
        matrix[k["mem_wr"], logic] = 8.0 * nmp["e_dram_interface_pj_per_bit"] * PJ
        for io in ("io_tx", "io_rx"):
            matrix[k[io], logic] = 8.0 * nmp["e_io_pj_per_bit"] * PJ
            matrix[k[io], pic] = 8.0 * nmp["e_pic_pj_per_bit"] * PJ
        static = np.array(
            [nmp["p_dram_static_w"]] * dram_dies
            + [nmp["p_logic_static_w"] + nmp["p_io_static_w"], nmp["p_pic_static_w"]]
        )
        return matrix, static
    if kind == "gpu":
        gpu = power.gpu
        matrix = np.zeros((len(ACTIVITY_KINDS), 3 * chips))
        for chip in range(chips):
            xpu, hbm, pic = 3 * chip, 3 * chip + 1, 3 * chip + 2
            for flop in ("flop_gemm", "flop_attn"):
                matrix[k[flop], xpu] = e_flop_pj * PJ / chips
            for io in ("io_tx", "io_rx"):
                matrix[k[io], pic] = 8.0 * gpu["e_link_pj_per_bit"] * PJ / chips
            for memory in ("mem_rd", "mem_wr", "act"):
                matrix[k[memory], hbm] = 8.0 * gpu["e_hbm_pj_per_bit"] * PJ / chips
        # The panel-link drivers (EIC) are flip-chip bonded on the PIC, so both sit on the pic die.
        static = np.array(
            [gpu["p_xpu_static_w"], gpu["p_hbm_static_w"], gpu["p_pic_static_w"] + gpu["p_io_static_w"]] * chips
        )
        return matrix, static
    raise ValueError(f"unknown device kind {kind!r}")


def price(
    binned: BinnedActivity,
    layout: DeviceLayout,
    power: PowerConfig,
    report: dict[str, Any] | None = None,
) -> PowerTrace:
    if tuple(binned.device_ids) != tuple(layout.device_ids):
        raise ValueError("binned devices differ from the layout")
    e_flop_pj, e_flop_source = resolve_e_flop_pj(power, report)
    widths = binned.bin_widths_s
    n_bins = len(widths)
    n_dies = len(layout.dies)
    energy_t = np.zeros((n_bins, n_dies))
    static = np.zeros(n_dies)
    by_category = np.zeros((n_dies, len(CATEGORIES)))
    beyond = np.zeros(n_dies)
    for row, device_id in enumerate(layout.device_ids):
        columns = np.asarray(layout.device_dies(device_id))
        matrix, device_static = device_coefficients(
            layout.kinds[device_id], layout.chips[device_id], layout.dram_dies, power, e_flop_pj
        )
        energy_t[:, columns] = binned.activity[:, row, :] @ matrix
        static[columns] = device_static
        by_category[columns, :] = (binned.by_category[row] @ matrix).T
        beyond[columns] = binned.beyond_horizon[row] @ matrix
    pool = n_dies - 1
    pool_j_per_byte = 8.0 * power.pool["e_pool_pj_per_bit"] * PJ
    energy_t[:, pool] = binned.pool_bytes * pool_j_per_byte
    by_category[pool, :] = binned.pool_by_category * pool_j_per_byte
    beyond[pool] = binned.pool_beyond_horizon * pool_j_per_byte

    power_w = energy_t / widths[:, None] + static[None, :]
    energy_j = np.concatenate([by_category, (static * binned.horizon_s)[:, None]], axis=1)
    meta = {
        "coefficients": power.as_meta(),
        "gpu_e_flop_pj": e_flop_pj,
        "gpu_e_flop_source": e_flop_source,
        "energy_columns": [*CATEGORIES, "static"],
    }
    return PowerTrace(
        t_edges_s=binned.t_edges_s,
        dies=layout.dies,
        power_w=power_w,
        static_w=static,
        energy_j=energy_j,
        beyond_horizon_j=beyond,
        meta=meta,
    )


def _window(trace: PowerTrace, warmup_s: float) -> np.ndarray:
    starts = trace.t_edges_s[:-1]
    mask = starts >= warmup_s
    if not mask.any():
        raise ValueError("warmup_s leaves no bins to summarize")
    return mask


def _stats(values: np.ndarray, widths: np.ndarray) -> tuple[float, float, float]:
    mean = float((values * widths).sum() / widths.sum())
    return mean, float(np.percentile(values, 95)), float(values.max())


def write_outputs(
    out_dir: Path,
    trace: PowerTrace,
    *,
    warmup_s: float,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write power.npz, dies.csv and chips.csv; return a JSON-able summary."""

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {**trace.meta, **(extra_meta or {})}
    np.savez_compressed(
        out_dir / "power.npz",
        t_edges_s=trace.t_edges_s,
        power_w=trace.power_w.astype(np.float32),
        static_w=trace.static_w,
        energy_j=trace.energy_j,
        beyond_horizon_j=trace.beyond_horizon_j,
        die_names=np.asarray(trace.die_names),
        die_chip=np.asarray([die.chip for die in trace.dies]),
        die_kind=np.asarray([die.kind for die in trace.dies]),
        die_device=np.asarray([-1 if die.device_id is None else die.device_id for die in trace.dies]),
        meta_json=np.asarray(json.dumps(meta, default=str)),
    )
    mask = _window(trace, warmup_s)
    widths = np.diff(trace.t_edges_s)[mask]
    power = trace.power_w[mask]
    columns = [*CATEGORIES, "static"]
    with (out_dir / "dies.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["die", "chip", "kind", "component", "device_id", "mean_w", "p95_w", "peak_w", "static_w",
             "energy_j", *(f"energy_j_{name}" for name in columns)]
        )
        for index, die in enumerate(trace.dies):
            mean, p95, peak = _stats(power[:, index], widths)
            writer.writerow(
                [die.name, die.chip, die.kind, die.component,
                 "" if die.device_id is None else die.device_id,
                 f"{mean:.6g}", f"{p95:.6g}", f"{peak:.6g}", f"{trace.static_w[index]:.6g}",
                 f"{trace.energy_j[index].sum():.6g}",
                 *(f"{value:.6g}" for value in trace.energy_j[index])]
            )
    chips: dict[str, list[int]] = {}
    for index, die in enumerate(trace.dies):
        chips.setdefault(die.chip, []).append(index)
    chip_rows = []
    with (out_dir / "chips.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["chip", "kind", "device_id", "mean_w", "p95_w", "peak_w", "energy_j"])
        for chip, indices in chips.items():
            die = trace.dies[indices[0]]
            total = power[:, indices].sum(axis=1)
            mean, p95, peak = _stats(total, widths)
            energy = float(trace.energy_j[indices].sum())
            chip_rows.append({"chip": chip, "kind": die.kind, "mean_w": mean, "peak_w": peak})
            writer.writerow(
                [chip, die.kind, "" if die.device_id is None else die.device_id,
                 f"{mean:.6g}", f"{p95:.6g}", f"{peak:.6g}", f"{energy:.6g}"]
            )
    energy_by_category = {
        name: float(trace.energy_j[:, column].sum()) for column, name in enumerate(columns)
    }
    by_kind: dict[str, dict[str, float]] = {}
    for row in chip_rows:
        entry = by_kind.setdefault(row["kind"], {"chips": 0, "mean_w_sum": 0.0, "mean_w_max": 0.0, "peak_w_max": 0.0})
        entry["chips"] += 1
        entry["mean_w_sum"] += row["mean_w"]
        entry["mean_w_max"] = max(entry["mean_w_max"], row["mean_w"])
        entry["peak_w_max"] = max(entry["peak_w_max"], row["peak_w"])
    return {
        "horizon_s": trace.horizon_s,
        "warmup_s": warmup_s,
        "bins": int(len(trace.t_edges_s) - 1),
        "dies": len(trace.dies),
        "energy_j_by_category": energy_by_category,
        "energy_j_total": float(trace.energy_j.sum()),
        "energy_j_beyond_horizon": float(trace.beyond_horizon_j.sum()),
        "chips_by_kind": by_kind,
        "placeholders": list(trace.meta["coefficients"]["placeholders"]),
        "gpu_e_flop_pj": trace.meta["gpu_e_flop_pj"],
    }
