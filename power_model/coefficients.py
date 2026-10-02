"""Energy coefficients and static power for NMP decode nodes and GPU prefill chips.

Every value carries its provenance. ``None`` marks a value only the hardware
owners can supply (static/leakage power, I/O energy per bit, GPU HBM energy).
With ``allow_placeholders: true`` a documented placeholder is used instead and
listed in every output, so no placeholder can silently pass for a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from roofline_model import PrecisionSpec


@dataclass(frozen=True)
class Coefficient:
    value: float | None
    source: str
    placeholder: float | None = None
    placeholder_note: str = ""


NMP_PRESETS: dict[str, dict[str, Coefficient]] = {
    "panagretti_nmp": {
        "dram_dies": Coefficient(4, "PanAgretti paper Table I: four M3D-DRAM dies per decode node"),
        "macs_per_second": Coefficient(
            5.24288e14, "paper Table I: 16 PUs x 32 matrix units x 32x32 MACs at 1 GHz"
        ),
        "dram_bytes_per_second": Coefficient(64e12, "paper: 4 dies x 16 TB/s internal bandwidth"),
        "io_bytes_per_second": Coefficient(
            15.75e12, "paper Section III-C: 63 links x 2 Tb/s per direction"
        ),
        "e_dram_pj_per_bit": Coefficient(
            0.43,
            "paper Table I DRAM access energy; perf_sim kernels at PanAgretti geometry match "
            "bytes x 0.429 pJ/bit within 0.4% (no row-granularity overhead)",
        ),
        "e_dram_interface_pj_per_bit": Coefficient(
            0.0,
            "assumed 0: the paper's 344.7 W logic die is explained by MACs + SRAM + 17.7 W I/O, "
            "leaving no separable DRAM PHY/controller energy",
        ),
        "e_mac_gemm_pj": Coefficient(
            0.6575,
            "scripts/power_crosscheck.py fit to perf_sim (Stratum) kernels at PanAgretti geometry, "
            "2026-09-27: FP16 MAC 0.604 pJ + register-file/psum/reducer/router overheads; "
            "FFN/projection kernels within 0.6%, attention within 4.8% (total energy)",
        ),
        "e_mac_attn_pj": Coefficient(0.6575, "same joint fit as e_mac_gemm_pj (softmax units included)"),
        "e_operand_fill_pj_per_bit": Coefficient(
            0.01646,
            "same fit: each DRAM-read weight/KV bit written into a PE register file "
            "(perf_sim matrix RF write 16.8 fJ/bit); why small-batch GEMVs cost more per MAC",
        ),
        "e_sram_pj_per_bit": Coefficient(
            0.0094, "perf_sim component_info.py psum/shared SRAM (IEEE 10873439 scaled to 0.7 V)"
        ),
        "e_io_pj_per_bit": Coefficient(
            None,
            "optical-I/O driver energy per bit on the logic die",
            0.0,
            "all 17.7 W of optical-I/O drivers treated as static (p_io_static_w)",
        ),
        "e_pic_pj_per_bit": Coefficient(
            None,
            "photonic-chip modulator/receiver energy per bit",
            0.0,
            "PIC power treated as static heaters/tuning (p_pic_static_w)",
        ),
        "p_dram_static_w": Coefficient(
            None,
            "per-die DRAM refresh/periphery/background power",
            3.86,
            "paper 58.9 W per die minus 0.43 pJ/bit x 128 Tb/s dynamic = 3.86 W residual",
        ),
        "p_logic_static_w": Coefficient(
            None,
            "logic-die leakage + clock tree",
            34.5,
            "~10% of the paper's 344.7 W logic die (typical 7 nm leakage share)",
        ),
        "p_io_static_w": Coefficient(
            None,
            "optical-I/O drivers when idle",
            17.7,
            "paper Section V-D: 17.7 W optical-I/O drivers, treated as always on",
        ),
        "p_pic_static_w": Coefficient(
            3.0, "paper Section V-D: PIC interleaver, modulator and filter heaters ~3 W"
        ),
    },
    "perf_sim_default_nmp": {
        "dram_dies": Coefficient(
            4, "PanAgretti node: four M3D-DRAM dies; each is one unpatched perf_sim stack (32 GiB)"
        ),
        "macs_per_second": Coefficient(
            2.62144e14,
            "perf_sim DRAM3D/config.py defaults: 16 ch x 16 banks x 16 PEs x 16 taps at 1 GHz "
            "= 65,536 MACs per stack, x 4 stacks (peak)",
        ),
        "dram_bytes_per_second": Coefficient(
            9.68e13,
            "perf_sim defaults: 256 banks x 4 KiB rows / tier-averaged tRC 43.3 ns = 24.2 TB/s "
            "per stack, x 4 (peak)",
        ),
        "io_bytes_per_second": Coefficient(
            15.75e12, "PanAgretti Section III-C: 63 links x 2 Tb/s per direction"
        ),
        "e_dram_pj_per_bit": Coefficient(
            0.4291,
            "perf_sim CfgDRAM3D.energy_per_bit_int 0.429 pJ/bit; scripts/power_crosscheck.py "
            "--geometry perf_sim_default measures bytes x 0.429 within 0.4%",
        ),
        "e_dram_interface_pj_per_bit": Coefficient(
            0.0, "perf_sim charges no separate DRAM PHY/controller energy (bank access energy only)"
        ),
        "e_mac_gemm_pj": Coefficient(
            0.659,
            "scripts/power_crosscheck.py --geometry perf_sim_default fit, 2026-09-27: FP16 MAC "
            "0.604 pJ + psum/reducer/router/softmax overheads; FFN/projection kernels within 0.5%, "
            "attention within 1.8% of total energy (logic-die-only residual up to 2.6%)",
        ),
        "e_mac_attn_pj": Coefficient(0.659, "same joint fit as e_mac_gemm_pj"),
        "e_operand_fill_pj_per_bit": Coefficient(
            0.01655, "same fit: register-file write of each DRAM-read weight/KV bit"
        ),
        "e_sram_pj_per_bit": Coefficient(
            0.0094, "perf_sim component_info.py psum/shared SRAM (IEEE 10873439 scaled to 0.7 V)"
        ),
        "e_io_pj_per_bit": Coefficient(
            None,
            "optical-I/O driver (EIC) energy per bit on the logic die; neither perf_sim nor "
            "Wang et al. CICC 2024 reports it",
            0.0,
            "all 17.7 W of PanAgretti optical-I/O drivers treated as static (p_io_static_w)",
        ),
        "e_pic_pj_per_bit": Coefficient(
            None,
            "PIC modulator/receiver switching energy per bit; Wang et al. CICC 2024 Table 1 "
            "reports only thermal-tuning power, which is static",
            0.0,
            "no traffic-proportional PIC energy",
        ),
        "p_dram_static_w": Coefficient(
            None,
            "per-die DRAM refresh/periphery/background power (perf_sim models none)",
            3.86,
            "PanAgretti 58.9 W per die minus 0.43 pJ/bit x 128 Tb/s dynamic = 3.86 W residual",
        ),
        "p_logic_static_w": Coefficient(
            None,
            "logic-die leakage + clock tree (perf_sim models none)",
            20.3,
            "~10% of this preset's ~203 W full-load logic die (typical 7 nm leakage share); "
            "a 15% share, as in perf_sim's vendored thermal model, would give ~36 W",
        ),
        "p_io_static_w": Coefficient(
            None,
            "optical-I/O drivers (EIC) when idle; not in Wang et al. CICC 2024",
            17.7,
            "PanAgretti Section V-D: 17.7 W optical-I/O drivers, treated as always on",
        ),
        "p_pic_static_w": Coefficient(
            18.94,
            "Wang et al. CICC 2024 Table 1 (w/ substrate undercut): modulator 206 + filter 59.6 + "
            "(de-)interleaver 28 fJ/b thermal tuning at the 1.024 Tb/s link rate = 0.30 W per "
            "64-ring link, x 63 Tx/Rx link pairs per node. Heater power is per ring, so it does "
            "not scale with PanAgretti's 32 Gb/s channels; the comb source is off-panel. This is "
            "6.3x PanAgretti's own ~3 W; without undercut Table 1 gives 35.3 W. Heater power "
            "falls as the die warms, so a thermal loop should replace this constant",
        ),
    },
}

GPU_PRESETS: dict[str, dict[str, Coefficient]] = {
    "b200_like": {
        "flops_per_second": Coefficient(2.25e15, "paper Section V-A: B200-like prefill node 2.25 PFLOP/s"),
        "hbm_bytes_per_second": Coefficient(8e12, "paper Section V-A: 8 TB/s HBM"),
        "tdp_w": Coefficient(1000.0, "B200 SXM board power (assumed; used for e_flop derivation and anchors)"),
        "e_flop_pj": Coefficient(
            None,
            "derived when omitted: a compute-bound prefill at the device's effective FLOP/s draws tdp_w",
        ),
        "e_hbm_pj_per_bit": Coefficient(
            None,
            "HBM access energy incl. PHY",
            4.0,
            "HBM3e literature range ~3.5-4.5 pJ/bit",
        ),
        "e_link_pj_per_bit": Coefficient(
            None,
            "GPU-side fabric I/O energy per bit",
            1.0,
            "generic co-packaged I/O figure",
        ),
        "p_xpu_static_w": Coefficient(
            None,
            "GPU die idle/leakage power",
            150.0,
            "typical B200-class idle draw",
        ),
        "p_hbm_static_w": Coefficient(
            None,
            "HBM background power per GPU",
            20.0,
            "3D-ICE-Glass fixed prefill HBM power, treated as static",
        ),
        "p_io_static_w": Coefficient(
            None,
            "optical-I/O drivers (EIC) of the chip's panel link",
            17.7,
            "same optical interface as a PanAgretti NMP node (17.7 W drivers, always on)",
        ),
        "p_pic_static_w": Coefficient(
            3.0, "PanAgretti Section V-D PIC heaters ~3 W, assuming the same interface as an NMP node"
        ),
    },
    "h200": {
        "flops_per_second": Coefficient(
            9.8e14,
            "perf_sim utils.get_xpu_config('H100') 980 TFLOPS dense FP16, an SXM-class figure "
            "(H200 shares the GH100 die; perf_sim multiplies by 2**40, a +10% unit slip, decimal "
            "used here; H200 NVL is ~835 TFLOPS)",
        ),
        "hbm_bytes_per_second": Coefficient(
            4.8e12, "H200 HBM3e 4.8 TB/s (perf_sim has no H200 entry; its H100 entry is 3.35 TB/s)"
        ),
        "tdp_w": Coefficient(
            600.0,
            "perf_sim XPUConfig.compute_power 600 W, which perf_sim charges as all-in busy power "
            "(= H200 NVL board power). Paired with SXM-class FLOP/s this is conservative per FLOP; "
            "H200 SXM is 700 W",
        ),
        "e_flop_pj": Coefficient(
            None,
            "derived when omitted: full effective compute + full effective HBM draws tdp_w",
        ),
        "e_hbm_pj_per_bit": Coefficient(
            3.2,
            "perf_sim CfgDRAM3D.energy_per_bit_ext / XPUConfig.io_eng_per_bit: 3.2 pJ/bit, "
            "perf_sim's HBM2e-class end-to-end host access energy, used as a proxy for H200 "
            "HBM3e access (array + PHY + controller, all placed on the hbm die)",
        ),
        "e_link_pj_per_bit": Coefficient(
            None,
            "panel-link switching energy per bit (EIC + PIC); Wang et al. CICC 2024 reports only "
            "static thermal tuning",
            0.0,
            "no traffic-proportional optical-I/O energy, as for the NMP nodes",
        ),
        "p_xpu_static_w": Coefficient(
            None,
            "GPU die idle/leakage power (perf_sim models busy power only)",
            90.0,
            "15% of the 600 W TDP, the same share as the B200 placeholder",
        ),
        "p_hbm_static_w": Coefficient(
            None,
            "HBM background power per GPU",
            20.0,
            "3D-ICE-Glass fixed prefill HBM power, treated as static",
        ),
        "p_io_static_w": Coefficient(
            None,
            "optical-I/O drivers (EIC) of the chip's panel link; not in Wang et al. CICC 2024",
            17.7,
            "same optical interface as a PanAgretti NMP node (17.7 W drivers, always on)",
        ),
        "p_pic_static_w": Coefficient(
            18.94,
            "Wang et al. CICC 2024 Table 1 (w/ undercut) thermal tuning, 0.30 W per 64-ring link "
            "x 63 links: a prefill node has the same 63 Tx/Rx panel links as a decode node",
        ),
    },
}

POOL_PRESETS: dict[str, dict[str, Coefficient]] = {
    "off_panel": {
        "e_pool_pj_per_bit": Coefficient(
            None,
            "KV backing-pool access + link energy per bit (location not yet decided)",
            0.0,
            "pool traffic is reported in bytes; its energy is left at zero until the pool is placed",
        ),
    }
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "model",
    "precision",
    "prefill_chips_per_device",
    "device_kinds",
    "gpu",
    "nmp",
    "kv_pool",
    "trace",
    "allow_placeholders",
}


@dataclass(frozen=True)
class PowerConfig:
    model: str
    precision: PrecisionSpec | None
    prefill_chips_per_device: int | None
    device_kinds: dict[int, str]
    nmp: dict[str, float]
    gpu: dict[str, float | None]
    pool: dict[str, float]
    dt_s: float
    warmup_s: float
    allow_placeholders: bool
    provenance: dict[str, str] = field(default_factory=dict)
    placeholders: tuple[str, ...] = ()
    nmp_preset: str = "panagretti_nmp"
    gpu_preset: str = "b200_like"

    def as_meta(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "nmp_preset": self.nmp_preset,
            "gpu_preset": self.gpu_preset,
            "nmp": dict(self.nmp),
            "gpu": dict(self.gpu),
            "kv_pool": dict(self.pool),
            "provenance": dict(self.provenance),
            "placeholders": list(self.placeholders),
            "dt_s": self.dt_s,
            "warmup_s": self.warmup_s,
        }


def _resolve(
    section: str,
    presets: dict[str, dict[str, Coefficient]],
    raw: Any,
    default_preset: str,
    allow_placeholders: bool,
    derived: frozenset[str] = frozenset(),
) -> tuple[dict[str, float | None], dict[str, str], list[str]]:
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError(f"{section}: must be a mapping")
    unknown = sorted(set(raw) - {"preset", "overrides"})
    if unknown:
        raise ValueError(f"{section}: unknown key(s): {', '.join(unknown)}")
    preset_name = raw.get("preset", default_preset)
    if preset_name not in presets:
        raise ValueError(f"{section}.preset must be one of {', '.join(presets)}")
    preset = presets[preset_name]
    overrides = raw.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{section}.overrides: must be a mapping")
    unknown = sorted(set(overrides) - set(preset))
    if unknown:
        raise ValueError(f"{section}.overrides: unknown coefficient(s): {', '.join(unknown)}")
    values: dict[str, float | None] = {}
    provenance: dict[str, str] = {}
    placeholders: list[str] = []
    missing: list[str] = []
    for name, coefficient in preset.items():
        key = f"{section}.{name}"
        if name in overrides and overrides[name] is not None:
            value = overrides[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key}: must be a finite non-negative number")
            values[name] = float(value)
            provenance[key] = "power.yaml override"
        elif coefficient.value is not None:
            values[name] = float(coefficient.value)
            provenance[key] = f"{preset_name}: {coefficient.source}"
        elif name in derived:
            values[name] = None
            provenance[key] = f"{preset_name}: {coefficient.source}"
        elif allow_placeholders and coefficient.placeholder is not None:
            values[name] = float(coefficient.placeholder)
            provenance[key] = (
                f"PLACEHOLDER {coefficient.placeholder:g} ({coefficient.placeholder_note}); "
                f"needed: {coefficient.source}"
            )
            placeholders.append(key)
        else:
            missing.append(f"{key} ({coefficient.source})")
    if missing:
        raise ValueError(
            "power.yaml is missing required value(s); supply overrides or set "
            "allow_placeholders: true -> " + "; ".join(missing)
        )
    return values, provenance, placeholders


def load_power_config(path: str | Path) -> PowerConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: root must be a mapping")
    unknown = sorted(set(document) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown key(s): {', '.join(unknown)}")
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    if not isinstance(document.get("model"), str):
        raise ValueError(f"{path}: model must be a roofline_model alias")
    allow = document.get("allow_placeholders", False)
    if not isinstance(allow, bool):
        raise ValueError(f"{path}: allow_placeholders must be true or false")

    precision = None
    if document.get("precision") is not None:
        raw = document["precision"]
        if not isinstance(raw, dict) or set(raw) - {"weight_bytes", "activation_bytes", "kv_bytes"}:
            raise ValueError(f"{path}: precision takes weight_bytes, activation_bytes, kv_bytes")
        precision = PrecisionSpec(**{key: float(value) for key, value in raw.items()})

    chips = document.get("prefill_chips_per_device")
    if chips is not None and (isinstance(chips, bool) or not isinstance(chips, int) or chips <= 0):
        raise ValueError(f"{path}: prefill_chips_per_device must be a positive integer")

    kinds_raw = document.get("device_kinds") or {}
    if not isinstance(kinds_raw, dict):
        raise ValueError(f"{path}: device_kinds must map device id -> gpu|nmp")
    device_kinds: dict[int, str] = {}
    for key, kind in kinds_raw.items():
        if kind not in ("gpu", "nmp"):
            raise ValueError(f"{path}: device_kinds[{key}] must be gpu or nmp")
        device_kinds[int(key)] = kind

    trace = document.get("trace") or {}
    if not isinstance(trace, dict) or set(trace) - {"dt_s", "warmup_s"}:
        raise ValueError(f"{path}: trace takes dt_s and warmup_s")
    dt_s = float(trace.get("dt_s", 0.01))
    warmup_s = float(trace.get("warmup_s", 0.0))
    if not math.isfinite(dt_s) or dt_s <= 0.0 or not math.isfinite(warmup_s) or warmup_s < 0.0:
        raise ValueError(f"{path}: trace.dt_s must be > 0 and warmup_s >= 0")

    nmp, nmp_prov, nmp_ph = _resolve("nmp", NMP_PRESETS, document.get("nmp"), "panagretti_nmp", allow)
    gpu, gpu_prov, gpu_ph = _resolve(
        "gpu", GPU_PRESETS, document.get("gpu"), "b200_like", allow, derived=frozenset({"e_flop_pj"})
    )
    pool, pool_prov, pool_ph = _resolve("kv_pool", POOL_PRESETS, document.get("kv_pool"), "off_panel", allow)
    if nmp["dram_dies"] is None or nmp["dram_dies"] < 1 or nmp["dram_dies"] != int(nmp["dram_dies"]):
        raise ValueError(f"{path}: nmp.dram_dies must be a positive integer")
    return PowerConfig(
        model=document["model"],
        precision=precision,
        prefill_chips_per_device=chips,
        device_kinds=device_kinds,
        nmp={key: float(value) for key, value in nmp.items()},
        gpu=gpu,
        pool={key: float(value) for key, value in pool.items()},
        dt_s=dt_s,
        warmup_s=warmup_s,
        allow_placeholders=allow,
        provenance={**nmp_prov, **gpu_prov, **pool_prov},
        placeholders=tuple(nmp_ph + gpu_ph + pool_ph),
        nmp_preset=(document.get("nmp") or {}).get("preset", "panagretti_nmp"),
        gpu_preset=(document.get("gpu") or {}).get("preset", "b200_like"),
    )
