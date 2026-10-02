"""Run one simulation with the power collector attached."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentic_sim.config.models import SimulationConfig
from agentic_sim.core.event import EventType
from agentic_sim.runtime.runtime import AgenticRuntime
from roofline_model import ModelSpec, PrecisionSpec

from .binner import ActivityBinner, BinnedActivity
from .coefficients import PowerConfig
from .collector import PowerCollector
from .layout import DeviceLayout
from .pricing import PowerTrace, price


@dataclass
class SimulationRun:
    runtime: AgenticRuntime
    snapshot: Any
    collector: PowerCollector
    layout: DeviceLayout
    binned: BinnedActivity
    trace: PowerTrace
    wall_s: float


def simulate(
    sim_config: SimulationConfig,
    power: PowerConfig,
    model: ModelSpec,
    precision: PrecisionSpec,
    *,
    report: dict[str, Any] | None = None,
    session_specs: Sequence[Any] | None = None,
    routing: Any | None = None,
    horizon_s: float | None = None,
    track_reservations: bool = False,
    prefill_transfers_at_completion: bool = True,
    collector_class: type[PowerCollector] = PowerCollector,
) -> SimulationRun:
    """``horizon_s`` bins [0, horizon_s]; pass the same value as simulation_end_s.

    ``prefill_transfers_at_completion`` (on by default here) books prefill KV
    transfers on the event clock; the simulator's default books them at launch,
    which stalls lock-step decode behind future transfers (see AgenticRuntime).
    """

    layout = DeviceLayout.from_config(sim_config, power, report)
    binner = ActivityBinner(power.dt_s, layout.device_ids, horizon_s)
    runtime = AgenticRuntime(
        sim_config,
        session_specs=session_specs,
        prefill_transfers_at_completion=prefill_transfers_at_completion,
    )
    collector = collector_class(
        sim_config,
        layout,
        binner,
        model,
        precision,
        report=report,
        routing=routing,
        track_reservations=track_reservations,
    )
    runtime.metrics = collector
    runtime.metrics.capture_attention(0.0, runtime.attention)
    started = time.time()
    snapshot = runtime.run()
    wall_s = time.time() - started
    for event in runtime.engine.pending(EventType.PREFILL_COMPLETE):
        collector.record_unfinished_prefill(event.payload)
    binned = binner.finalize(runtime.engine.now)
    trace = price(binned, layout, power, report)
    return SimulationRun(runtime, snapshot, collector, layout, binned, trace, wall_s)
