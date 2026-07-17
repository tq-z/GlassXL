"""Hardware state exposed to execution policies."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_sim.hardware.topology import Topology


@dataclass(frozen=True)
class HardwareState:
    topology: Topology
