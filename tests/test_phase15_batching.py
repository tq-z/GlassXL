from __future__ import annotations

from copy import deepcopy

import pytest

from agentic_sim.config import ConfigurationError, validate_config
from agentic_sim.core.event import EventType
from agentic_sim.core.event_engine import EventEngine
from agentic_sim.core.simulation import Simulation
from agentic_sim.hardware.bandwidth_pool import BandwidthFlow, BandwidthPool
from agentic_sim.hardware.device import DecodeSettings, Device, PrefillSettings
from agentic_sim.metrics.collector import MetricsCollector
from agentic_sim.runtime.decode_batcher import DecodeBatchController
from agentic_sim.runtime.phase_state import InvocationPhase, InvocationSession
from agentic_sim.runtime.prefill_batcher import PrefillBatchController
from agentic_sim.workload.invocation import Invocation


def make_session(session_id: int, prefill_tokens: int, output_tokens: int = 1, kv_bytes: int = 100) -> InvocationSession:
    invocation = Invocation(session_id, session_id, 0, prefill_tokens, output_tokens, 0, 0.0)
    return InvocationSession(
        invocation=invocation,
        effective_prefill_tokens=prefill_tokens,
        prompt_kv_bytes=kv_bytes,
        remaining_output_tokens=output_tokens,
        phase=InvocationPhase.WAITING_PREFILL,
    )


def test_prefill_greedy_batch_uses_batch_size_dependent_equation() -> None:
    engine = EventEngine()
    metrics = MetricsCollector()
    device = Device(0, "compute", 0.01, prefill=PrefillSettings(10.0, 15.0, 4, 100))
    completed: list[int] = []
    controller = PrefillBatchController(
        engine,
        device,
        device.prefill,
        metrics,
        lambda session, queue, service: None,
        lambda session: completed.append(session.session_id),
    )
    engine.register_handler(EventType.PREFILL_BATCH_START, lambda event: controller.handle_start())
    engine.register_handler(EventType.PREFILL_BATCH_FINISH, lambda event: controller.handle_finish(event.payload))
    controller.enqueue(make_session(0, 4))
    controller.enqueue(make_session(1, 6))
    engine.run()

    assert completed == [0, 1]
    assert len(metrics.prefill_batches) == 1
    batch = metrics.prefill_batches[0]
    assert batch.request_count == 2
    assert batch.total_tokens == 10
    assert batch.finish_time_s - batch.start_time_s == pytest.approx(0.01 + max(6 / 10, 10 / 15))


def test_prefill_oversized_head_runs_alone_without_blocking() -> None:
    engine = EventEngine()
    metrics = MetricsCollector()
    device = Device(0, "compute", 0.0, prefill=PrefillSettings(10.0, 20.0, 8, 10))
    controller = PrefillBatchController(engine, device, device.prefill, metrics, lambda *args: None, lambda session: None)
    engine.register_handler(EventType.PREFILL_BATCH_START, lambda event: controller.handle_start())
    engine.register_handler(EventType.PREFILL_BATCH_FINISH, lambda event: controller.handle_finish(event.payload))
    controller.enqueue(make_session(0, 12))
    controller.enqueue(make_session(1, 3))
    engine.run()

    assert [batch.request_count for batch in metrics.prefill_batches] == [1, 1]
    assert [batch.total_tokens for batch in metrics.prefill_batches] == [12, 3]


def test_continuous_decode_removes_and_admits_only_at_boundaries() -> None:
    engine = EventEngine()
    metrics = MetricsCollector()
    device = Device(1, "compute", 0.01, decode=DecodeSettings(10.0, 15.0, 2))
    completed: list[int] = []
    controller = DecodeBatchController(
        engine,
        device,
        device.decode,
        metrics,
        lambda session, queue: None,
        lambda session, time, service: None,
        lambda session: completed.append(session.session_id),
    )
    engine.register_handler(EventType.DECODE_ITERATION_START, lambda event: controller.handle_start())
    engine.register_handler(EventType.DECODE_STEP_FINISH, lambda event: controller.handle_finish(event.payload))
    late = make_session(2, 1, output_tokens=1)
    engine.register_handler(EventType.AGENT_ARRIVAL, lambda event: controller.enqueue(late))
    controller.enqueue(make_session(0, 1, output_tokens=2))
    controller.enqueue(make_session(1, 1, output_tokens=1))
    engine.schedule(0.05, EventType.AGENT_ARRIVAL)
    engine.run()

    assert completed == [1, 0, 2]
    assert [item.active_sessions for item in metrics.decode_iterations] == [2, 2]
    assert metrics.decode_iterations[0].finish_time_s == pytest.approx(0.01 + max(0.1, 2 / 15))
    assert late.decode_started_at == pytest.approx(metrics.decode_iterations[0].finish_time_s)


def test_bandwidth_pool_rebalances_and_completes_flows_independently() -> None:
    engine = EventEngine()
    completed: list[tuple[int, float]] = []
    pool = BandwidthPool(
        engine,
        fixed_latency=0.1,
        per_session_max_bandwidth=100.0,
        max_system_bandwidth=150.0,
        on_flow_complete=lambda flow, time: completed.append((flow.flow_id, time)),
        on_active_change=lambda time, count, bandwidth: None,
    )
    engine.register_handler(EventType.KV_INGRESS_DATA_READY, lambda event: pool.handle_data_ready(event.payload, engine.now))
    engine.register_handler(EventType.KV_INGRESS_FINISH, lambda event: pool.handle_completion(event.payload, engine.now))
    pool.submit("short", 75, 0.0)
    pool.submit("long", 150, 0.0)
    engine.run()

    assert completed == [(0, pytest.approx(1.1)), (1, pytest.approx(1.85))]
    assert pool.active_count == 0


def test_bandwidth_pool_honors_per_session_cap() -> None:
    engine = EventEngine()
    completed: list[float] = []
    pool = BandwidthPool(engine, 0.2, 50.0, 1000.0, lambda flow, time: completed.append(time), lambda *args: None)
    engine.register_handler(EventType.KV_INGRESS_DATA_READY, lambda event: pool.handle_data_ready(event.payload, engine.now))
    engine.register_handler(EventType.KV_INGRESS_FINISH, lambda event: pool.handle_completion(event.payload, engine.now))
    pool.submit(None, 100, 0.0)
    engine.run()
    assert completed == [pytest.approx(2.2)]


def preferred_config(minimal_config: dict) -> dict:
    config = deepcopy(minimal_config)
    config["hardware"]["devices"] = [
        {
            "id": 0,
            "class": "compute",
            "fixed_latency_s": 0.01,
            "prefill": {
                "single_request_tokens_per_second": 10.0,
                "max_system_tokens_per_second": 30.0,
                "max_batch_requests": 8,
                "max_batch_tokens": 100,
                "layout": "packed",
                "batch_wait_us": 0,
            },
        },
        {
            "id": 1,
            "class": "compute",
            "fixed_latency_s": 0.01,
            "decode": {
                "single_session_tokens_per_second": 10.0,
                "max_system_tokens_per_second": 30.0,
                "max_active_sessions": 8,
            },
        },
    ]
    config["hardware"]["kv_ingress"] = {
        "fixed_latency_s": 0.01,
        "per_session_max_bandwidth_bytes_per_sec": 100.0,
        "max_system_bandwidth_bytes_per_sec": 200.0,
    }
    config["hardware"]["links"] = []
    config["invocation_model"]["kv"]["retention"] = "discard"
    config["workload"].update(
        {
            "num_agents": 5,
            "arrival": {"type": "poisson", "rate_per_second": 1_000_000_000.0},
            "turns": {"type": "constant", "value": 1},
            "input_tokens": {"type": "constant", "value": 2},
            "output_tokens": {"type": "constant", "value": 3},
        }
    )
    return config


def test_phase15_runs_end_to_end_with_batch_and_bandwidth_metrics(minimal_config: dict) -> None:
    result = Simulation(preferred_config(minimal_config)).run()
    assert result.summary["completed_agents"] == 5
    assert result.summary["mean_prefill_batch_requests"] > 1.0
    assert result.summary["mean_decode_batch_size"] > 1.0
    assert result.summary["peak_concurrent_ingress_flows"] > 1
    assert result.summary["decode_tokens_emitted"] == 15
    assert result.summary["total_ingress_bytes"] == 100
    assert result.summary["mean_ttft_ms"] > 0.0
    assert result.summary["mean_inter_token_ms"] > 0.0
    assert all(metric.first_token_time_s is not None for metric in result.invocations)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("prefill", "max_batch_requests"), 0, "max_batch_requests"),
        (("prefill", "layout"), "padded", "packed"),
        (("decode", "max_active_sessions"), 0, "max_active_sessions"),
    ],
)
def test_invalid_phase15_batching_configuration_is_rejected(
    minimal_config: dict, path: tuple[str, str], value: object, message: str
) -> None:
    config = preferred_config(minimal_config)
    device_index = 0 if path[0] == "prefill" else 1
    config["hardware"]["devices"][device_index][path[0]][path[1]] = value
    with pytest.raises(ConfigurationError, match=message):
        validate_config(config)


def test_invalid_ingress_cap_is_rejected(minimal_config: dict) -> None:
    config = preferred_config(minimal_config)
    config["hardware"]["kv_ingress"]["max_system_bandwidth_bytes_per_sec"] = 0
    with pytest.raises(ConfigurationError, match="max_system_bandwidth"):
        validate_config(config)
