"""Map simulator devices to physical chips and dies.

A simulator prefill device is a tensor-parallel group of GPU chips, each with
its xPU die, HBM and photonic chip; each decode device (attention and/or FFN)
is one NMP node with its DRAM dies, logic die and photonic chip. Roles come from ``mapping.yaml``; ``device_kinds`` in
``power.yaml`` overrides them for mixed configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_sim.config.models import SimulationConfig

from .coefficients import PowerConfig

NMP_LOGIC_COMPONENTS = ("logic", "pic")
GPU_COMPONENTS = ("xpu", "hbm", "pic")


@dataclass(frozen=True)
class Die:
    name: str
    device_id: int | None
    chip: str
    kind: str
    component: str


@dataclass(frozen=True)
class DeviceLayout:
    device_ids: tuple[int, ...]
    kinds: dict[int, str]
    chips: dict[int, int]
    dram_dies: int
    dies: tuple[Die, ...]

    @property
    def die_names(self) -> tuple[str, ...]:
        return tuple(die.name for die in self.dies)

    def device_dies(self, device_id: int) -> tuple[int, ...]:
        return tuple(index for index, die in enumerate(self.dies) if die.device_id == device_id)

    @classmethod
    def from_config(
        cls,
        sim_config: SimulationConfig,
        power: PowerConfig,
        report: dict[str, Any] | None = None,
    ) -> DeviceLayout:
        mapping = sim_config.mapping
        prefill_ids = set(mapping.prefill.device_ids)
        decode_ids = set(mapping.decode_attn.device_ids) | set(mapping.decode_ffn.device_ids)
        device_ids = tuple(sorted(device.id for device in sim_config.hardware.devices))
        kinds: dict[int, str] = {}
        for device_id in device_ids:
            if device_id in power.device_kinds:
                kinds[device_id] = power.device_kinds[device_id]
            elif device_id in prefill_ids and device_id in decode_ids:
                raise ValueError(
                    f"device {device_id} serves both prefill and decode; set device_kinds in power.yaml"
                )
            elif device_id in prefill_ids:
                kinds[device_id] = "gpu"
            elif device_id in decode_ids:
                kinds[device_id] = "nmp"
        unknown = sorted(set(power.device_kinds) - set(device_ids))
        if unknown:
            raise ValueError(f"power.yaml device_kinds names unknown device(s): {unknown}")

        chips_per_prefill = power.prefill_chips_per_device
        if chips_per_prefill is None:
            deployment = (report or {}).get("deployment") or {}
            chips_per_prefill = int(deployment.get("prefill_tp", 1))
        dram_dies = int(power.nmp["dram_dies"])

        dies: list[Die] = []
        chips: dict[int, int] = {}
        used = tuple(device_id for device_id in device_ids if device_id in kinds)
        for device_id in used:
            if kinds[device_id] == "nmp":
                chips[device_id] = 1
                chip = f"nmp{device_id:03d}"
                for component in (*(f"dram{j}" for j in range(dram_dies)), *NMP_LOGIC_COMPONENTS):
                    dies.append(Die(f"{chip}.{component}", device_id, chip, "nmp", component))
            else:
                chips[device_id] = chips_per_prefill
                for index in range(chips_per_prefill):
                    chip = f"gpu{device_id}_{index}"
                    for component in GPU_COMPONENTS:
                        dies.append(Die(f"{chip}.{component}", device_id, chip, "gpu", component))
        dies.append(Die("kv_pool", None, "kv_pool", "pool", "pool"))
        return cls(
            device_ids=used,
            kinds={device_id: kinds[device_id] for device_id in used},
            chips=chips,
            dram_dies=dram_dies,
            dies=tuple(dies),
        )
