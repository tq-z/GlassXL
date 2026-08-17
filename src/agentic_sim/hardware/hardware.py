"""Hardware state exposed to future runtime controllers."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_sim.config.models import HardwareConfig
from agentic_sim.hardware.topology import Topology


@dataclass(frozen=True)
class HardwareState:
    topology: Topology

    @classmethod
    def from_config(cls, config: HardwareConfig) -> HardwareState:
        return cls(Topology.from_config(config))
