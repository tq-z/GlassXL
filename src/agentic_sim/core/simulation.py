"""Top-level assembly for the reconstructed PD/AFD simulation."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_sim.config import SimulationConfig, validate_simulation_config
from agentic_sim.metrics import (
    AttentionAdmissionMetric,
    AttentionDeviceMetric,
    DecodeStepMetric,
    PrefillBatchMetric,
    PrefillSessionMetric,
    ResourceMetric,
    SessionMetric,
    TurnMetric,
)
from agentic_sim.runtime.runtime import AgenticRuntime
from agentic_sim.runtime.session import SessionState


@dataclass(frozen=True)
class SimulationResult:
    summary: dict[str, int | float]
    sessions: tuple[SessionMetric, ...]
    turns: tuple[TurnMetric, ...]
    attention_admissions: tuple[AttentionAdmissionMetric, ...]
    attention_devices: tuple[AttentionDeviceMetric, ...]
    prefill_batches: tuple[PrefillBatchMetric, ...]
    prefill_sessions: tuple[PrefillSessionMetric, ...]
    decode_steps: tuple[DecodeStepMetric, ...]
    resources: tuple[ResourceMetric, ...]
    session_states: tuple[SessionState, ...]


class Simulation:
    """Run only the strict typed three-file V1 configuration."""

    def __init__(self, config: SimulationConfig) -> None:
        if not isinstance(config, SimulationConfig):
            raise TypeError("Simulation requires a typed three-file SimulationConfig")
        validate_simulation_config(config)
        self.config = config

    def run(self) -> SimulationResult:
        runtime = AgenticRuntime(self.config)
        snapshot = runtime.run()
        return SimulationResult(
            summary=snapshot.summary,
            sessions=snapshot.sessions,
            turns=snapshot.turns,
            attention_admissions=snapshot.attention_admissions,
            attention_devices=snapshot.attention_devices,
            prefill_batches=snapshot.prefill_batches,
            prefill_sessions=snapshot.prefill_sessions,
            decode_steps=snapshot.decode_steps,
            resources=snapshot.resources,
            session_states=tuple(runtime.sessions.values()),
        )
