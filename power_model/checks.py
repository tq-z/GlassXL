"""Invariant checks for binned activity and priced power traces."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from .binner import ACTIVITY_KINDS, CATEGORIES, KIND_INDEX, BinnedActivity
from .coefficients import PowerConfig
from .layout import DeviceLayout
from .pricing import PowerTrace, device_coefficients, gpu_effective_rates_per_chip, resolve_e_flop_pj


def _close(a: float, b: float, rel: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-6)


def conservation(binned: BinnedActivity, trace: PowerTrace) -> dict[str, Any]:
    """Binned activity and energy must reconcile with the per-category tallies."""

    widths = np.diff(trace.t_edges_s)
    total = float((trace.power_w * widths[:, None]).sum())
    static = float(trace.static_w.sum() * trace.horizon_s)
    dynamic = trace.dynamic_energy_j()
    categories = float(trace.energy_j[:, : len(CATEGORIES)].sum())
    activity_binned = binned.activity.sum(axis=(0, 1))
    activity_categories = binned.by_category.sum(axis=(0, 1))
    activity_ok = all(
        _close(float(a), float(b)) for a, b in zip(activity_binned, activity_categories)
    )
    return {
        "ok": _close(dynamic, categories) and _close(total, dynamic + static) and activity_ok,
        "total_j": total,
        "dynamic_j": dynamic,
        "static_j": static,
        "category_j": categories,
        "activity_matches_categories": activity_ok,
    }


def io_balance(binned: BinnedActivity) -> dict[str, Any]:
    """Every byte sent on-panel is received on-panel, except pool traffic."""

    tx = float(binned.by_category[:, :, KIND_INDEX["io_tx"]].sum())
    rx = float(binned.by_category[:, :, KIND_INDEX["io_rx"]].sum())
    offload = float(binned.pool_by_category[CATEGORIES.index("pool_offload")])
    load = float(binned.pool_by_category[CATEGORIES.index("pool_load")])
    return {
        "ok": math.isclose(tx - rx, offload - load, rel_tol=0.0, abs_tol=1e-9 * max(tx, rx, 1.0)),
        "tx_bytes": tx,
        "rx_bytes": rx,
        "pool_offload_bytes": offload,
        "pool_load_bytes": load,
    }


def completeness(collector, runtime) -> dict[str, Any]:
    """The collector must have seen every reservation the simulator made."""

    expected = Counter(
        (item.device_id, item.operation.value, item.start_time, item.finish_time)
        for device in runtime.topology.devices.values()
        for item in device.reservations
    )
    seen = collector.seen_counter()
    missing = expected - seen
    extra = seen - expected
    return {
        "ok": not missing and not extra,
        "reservations": int(sum(expected.values())),
        "missing": int(sum(missing.values())),
        "extra": int(sum(extra.values())),
        "missing_by_operation": dict(Counter(key[1] for key in missing.elements())),
    }


# PanAgretti paper Section V-D full-load die powers and tolerances. They describe
# the paper's node geometry, so they are enforced only for the preset that models it.
PAPER_NMP_TARGETS = {
    "panagretti_nmp": {"dram": (58.9, 0.05), "logic": (344.7, 0.10), "pic": (3.0, 0.05)},
}
PAPER_REFERENCE = {"dram": 58.9, "logic": 344.7, "pic": 3.0}


def _anchor(value: float, name: str, targets: dict[str, tuple[float, float]] | None) -> dict[str, Any]:
    if targets is None:
        # Another geometry: report the paper figure for comparison only.
        reference = PAPER_REFERENCE[name]
        return {"value": value, "paper_reference": reference, "deviation": value / reference - 1}
    target, tolerance = targets[name]
    return {
        "value": value,
        "target": target,
        "deviation": value / target - 1,
        "ok": abs(value / target - 1) <= tolerance,
    }


def anchors(power: PowerConfig, layout: DeviceLayout, report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Price one second of full-rate activity; compare with the paper's die powers where they apply."""

    nmp = power.nmp
    dies = int(nmp["dram_dies"])
    e_flop_pj, _ = resolve_e_flop_pj(power, report)
    matrix, static = device_coefficients("nmp", 1, dies, power, e_flop_pj)
    full = np.zeros(len(ACTIVITY_KINDS))
    full[KIND_INDEX["flop_gemm"]] = 2.0 * nmp["macs_per_second"]
    full[KIND_INDEX["mem_rd"]] = nmp["dram_bytes_per_second"]
    full[KIND_INDEX["io_tx"]] = nmp["io_bytes_per_second"]
    full[KIND_INDEX["io_rx"]] = nmp["io_bytes_per_second"]
    dynamic = full @ matrix
    dram_die = float(dynamic[0] + static[0])
    logic_no_leakage = float(dynamic[dies] + nmp["p_io_static_w"])
    pic = float(dynamic[dies + 1] + static[dies + 1])
    targets = PAPER_NMP_TARGETS.get(power.nmp_preset)
    result: dict[str, Any] = {
        "nmp_preset": power.nmp_preset,
        "nmp_dram_die_w": _anchor(dram_die, "dram", targets),
        "nmp_logic_die_w_excl_leakage": {
            **_anchor(logic_no_leakage, "logic", targets),
            "note": (
                "full MAC rate + operand fill at full DRAM bandwidth + optical-I/O drivers, excluding "
                "leakage; the paper's synthesized 344.7 W implies ~0.62 pJ/MAC, the perf_sim fit ~0.66"
            ),
        },
        "nmp_pic_w": _anchor(pic, "pic", targets),
        "nmp_node_full_load_w": float(dynamic.sum() + static.sum()),
    }
    gpu_matrix, gpu_static = device_coefficients("gpu", 1, dies, power, e_flop_pj)
    # TDP covers the GPU package (xpu die + HBM); the panel-link PIC/EIC die is separate.
    package = [0, 1]
    flops, hbm, source = gpu_effective_rates_per_chip(power, report)
    full = np.zeros(len(ACTIVITY_KINDS))
    full[KIND_INDEX["flop_gemm"]] = flops
    compute_only = float((full @ gpu_matrix)[package].sum() + gpu_static[package].sum())
    full[KIND_INDEX["mem_rd"]] = hbm
    gpu_total = float((full @ gpu_matrix)[package].sum() + gpu_static[package].sum())
    result["gpu_pic_idle_w"] = float(gpu_static[2])
    gpu_anchor = {
        "value": gpu_total,
        "target": power.gpu["tdp_w"],
        "effective_flops_per_chip": flops,
        "effective_hbm_bytes_per_chip": hbm,
        "rates_source": source,
        "e_flop_pj": e_flop_pj,
    }
    if power.gpu.get("e_flop_pj") is None:
        # e_flop was derived to make exactly this point draw TDP: an identity, not a check.
        gpu_anchor["note"] = "identity: e_flop_pj is derived from tdp_w at this operating point"
    else:
        gpu_anchor["ok"] = abs(gpu_total / power.gpu["tdp_w"] - 1) <= 0.10
    result["gpu_full_compute_and_hbm_w"] = gpu_anchor
    result["gpu_compute_only_w"] = compute_only
    result["gpu_idle_w"] = float(gpu_static[package].sum())
    enforced = [item["ok"] for item in result.values() if isinstance(item, dict) and "ok" in item]
    # None: no independent anchor applies to these presets (nothing was validated).
    result["ok"] = all(enforced) if enforced else None
    return result
