from __future__ import annotations

from copy import deepcopy

import pytest

from agentic_sim.core.event import EventType
from agentic_sim.core.event_engine import EventEngine
from agentic_sim.core.simulation import Simulation
from agentic_sim.hardware.device import DecodeSettings, Device, PrefillSettings
from agentic_sim.metrics.collector import MetricsCollector
from agentic_sim.runtime.decode_batcher import DecodeBatchController
from agentic_sim.runtime.phase_state import InvocationPhase, InvocationSession
from agentic_sim.runtime.prefill_batcher import PrefillBatchController
from agentic_sim.workload.invocation import Invocation


def session(session_id: int, prefill: int, output: int = 1) -> InvocationSession:
    invocation = Invocation(session_id, session_id, 0, prefill, output, 0, 0.0)
    return InvocationSession(invocation, prefill, prefill * 10, output, InvocationPhase.WAITING_PREFILL)


def test_prefill_arrival_during_service_waits_for_next_batch() -> None:
    engine = EventEngine()
    metrics = MetricsCollector()
    device = Device(0, "compute", 0.0, prefill=PrefillSettings(10.0, 10.0, 8, 100))
    finishes: dict[int, float] = {}
    controller = PrefillBatchController(
        engine,
        device,
        device.prefill,
        metrics,
        lambda *args: None,
        lambda current: finishes.setdefault(current.session_id, engine.now),
    )
    engine.register_handler(EventType.PREFILL_BATCH_START, lambda event: controller.handle_start())
    engine.register_handler(EventType.PREFILL_BATCH_FINISH, lambda event: controller.handle_finish(event.payload))
    engine.register_handler(EventType.AGENT_ARRIVAL, lambda event: controller.enqueue(session(1, 2)))
    controller.enqueue(session(0, 10))
    engine.schedule(0.25, EventType.AGENT_ARRIVAL)
    engine.run()

    assert [item.request_count for item in metrics.prefill_batches] == [1, 1]
    assert metrics.prefill_batches[1].start_time_s == pytest.approx(metrics.prefill_batches[0].finish_time_s)
    assert finishes == {0: pytest.approx(1.0), 1: pytest.approx(1.2)}


def test_decode_waiter_fills_slot_at_boundary() -> None:
    engine = EventEngine()
    metrics = MetricsCollector()
    device = Device(1, "compute", 0.0, decode=DecodeSettings(10.0, 20.0, 2))
    completed: list[int] = []
    controller = DecodeBatchController(
        engine,
        device,
        device.decode,
        metrics,
        lambda *args: None,
        lambda *args: None,
        lambda current: completed.append(current.session_id),
    )
    engine.register_handler(EventType.DECODE_ITERATION_START, lambda event: controller.handle_start())
    engine.register_handler(EventType.DECODE_STEP_FINISH, lambda event: controller.handle_finish(event.payload))
    controller.enqueue(session(0, 1, 1))
    controller.enqueue(session(1, 1, 2))
    controller.enqueue(session(2, 1, 1))
    engine.run()

    assert [item.active_sessions for item in metrics.decode_iterations] == [2, 2]
    assert completed == [0, 1, 2]
    assert metrics.max_decode_queue == 3


def phase15_retention_config(minimal_config: dict, retention: str) -> dict:
    config = deepcopy(minimal_config)
    config["hardware"]["devices"] = [
        {
            "id": 0,
            "class": "compute",
            "fixed_latency_s": 0.01,
            "prefill": {
                "single_request_tokens_per_second": 20.0,
                "max_system_tokens_per_second": 40.0,
                "max_batch_requests": 4,
                "max_batch_tokens": 100,
            },
        },
        {
            "id": 1,
            "class": "compute",
            "fixed_latency_s": 0.01,
            "decode": {
                "single_session_tokens_per_second": 10.0,
                "max_system_tokens_per_second": 20.0,
                "max_active_sessions": 4,
            },
        },
    ]
    config["hardware"]["kv_ingress"] = {
        "fixed_latency_s": 0.01,
        "per_session_max_bandwidth_bytes_per_sec": 100.0,
        "max_system_bandwidth_bytes_per_sec": 200.0,
    }
    config["hardware"]["links"] = [
        link for link in config["hardware"]["links"] if (link["src"], link["dst"]) != (0, 1)
    ]
    config["invocation_model"]["kv"]["retention"] = retention
    config["workload"]["num_agents"] = 2
    return config


@pytest.mark.parametrize("retention", ["same_device", "memory", "far_memory", "discard"])
def test_phase15_preserves_all_retention_modes(minimal_config: dict, retention: str) -> None:
    result = Simulation(phase15_retention_config(minimal_config, retention)).run()
    assert result.summary["completed_agents"] == 2
    assert result.summary["ingress_flows"] == 4
    assert len(result.prefill_batches) == result.summary["prefill_batches"]
    assert len(result.ingress_flows) == result.summary["ingress_flows"]
    assert len(result.decode_iterations) == result.summary["decode_iterations"]


def test_phase15_fixed_seed_is_deterministic(minimal_config: dict) -> None:
    config = phase15_retention_config(minimal_config, "discard")
    first = Simulation(deepcopy(config)).run()
    second = Simulation(deepcopy(config)).run()
    assert first.summary == second.summary
    assert first.prefill_batches == second.prefill_batches
    assert first.ingress_flows == second.ingress_flows
    assert first.decode_iterations == second.decode_iterations
