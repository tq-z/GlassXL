from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from agentic_sim import Simulation, load_config
from agentic_sim.config.models import (
    AttentionPlacement,
    AttentionResidency,
    ConstantDistributionConfig,
    DistributionType,
)
from agentic_sim.runtime.runtime import AgenticRuntime
from agentic_sim.workload import SessionSpec, TurnSpec


ROOT = Path(__file__).parents[1]


def example_config():
    return load_config(
        ROOT / "configs/example/hardware.yaml",
        ROOT / "configs/example/workload.yaml",
        ROOT / "configs/example/mapping.yaml",
    )


def deterministic_config():
    config = example_config()
    constant = lambda value: ConstantDistributionConfig(DistributionType.CONSTANT, value)
    source = replace(
        config.workload.source,
        num_sessions=2,
        turns=constant(2),
        context_tokens=constant(4),
        prefill_tokens=constant(2),
        output_tokens=constant(2),
        tool_wait_ms=constant(100),
    )
    return replace(
        config,
        workload=replace(config.workload, simulation_end_s=None, source=source),
    )


def test_end_to_end_is_deterministic_and_reconciles_all_session_tokens() -> None:
    config = deterministic_config()

    first = Simulation(config).run()
    second = Simulation(config).run()

    assert first.summary == second.summary
    assert first.sessions == second.sessions
    assert first.turns == second.turns
    assert first.decode_steps == second.decode_steps
    assert first.summary["completed_sessions"] == 2
    assert first.summary["completed_turns"] == 4
    assert first.summary["emitted_tokens"] == 8
    assert first.summary["prefill_compute_tokens"] == 8
    assert first.summary["kv_pool_load_tokens"] == 12
    assert first.summary["kv_pool_offload_tokens"] == 16
    assert first.summary["kv_p2p_tokens"] == 16
    assert first.summary["routed_activation_tokens"] == 16
    assert first.summary["all_to_all_send_tokens"] == 16
    assert tuple(item.final_context_tokens for item in first.sessions) == (12, 12)
    assert tuple(item.generated_tokens for item in first.sessions) == (4, 4)
    assert all(item.is_finished for item in first.session_states)
    assert all(
        item.kv.pool.snapshot == item.kv.current_snapshot
        for item in first.session_states
    )


def test_section_18_metrics_cover_stages_capacity_experts_and_resources() -> None:
    result = Simulation(deterministic_config()).run()

    assert len(result.turns) == 4
    assert all(item.finish_time_s is not None for item in result.turns)
    assert all(item.ttft_s is not None for item in result.turns)
    assert all(item.prefill_service_s > 0.0 for item in result.turns)
    assert all(item.kv_pool_load_s > 0.0 for item in result.turns)
    assert all(item.kv_p2p_s > 0.0 for item in result.turns)
    assert {item.load_kind for item in result.prefill_sessions} == {"full_context", "delta"}
    assert sum(item.stateful_affinity_hit for item in result.prefill_sessions) == 2
    assert sum(item.cache_hit for item in result.attention_admissions) == 2
    assert all(item.capacity_utilization <= 1.0 for item in result.attention_devices)
    assert all(
        sum(tokens for _, tokens, _ in item.expert_loads)
        == item.routed_activation_tokens
        for item in result.decode_steps
    )
    assert all(item.dispatch_latency_s == item.combine_latency_s for item in result.decode_steps)
    assert {item.resource_type for item in result.resources} == {
        "physical_device",
        "ffn_all_to_all",
        "kv_p2p",
        "kv_pool",
    }
    assert all(not hasattr(item, "bytes_processed") for item in result.resources)


def test_capacity_pressure_retries_then_evicts_only_recoverable_clean_kv() -> None:
    config = example_config()
    devices = tuple(
        replace(
            item,
            capabilities=replace(
                item.capabilities,
                decode_attn=replace(
                    item.capabilities.decode_attn,
                    kv_capacity_tokens=6,
                ),
            ),
        )
        if item.id == 1
        else item
        for item in config.hardware.devices
    )
    attention_mapping = replace(
        config.mapping.decode_attn,
        device_ids=(1,),
        placement=AttentionPlacement.ROUND_ROBIN,
        residency=AttentionResidency.LRU_CACHE,
    )
    config = replace(
        config,
        hardware=replace(config.hardware, devices=devices),
        mapping=replace(config.mapping, decode_attn=attention_mapping),
        workload=replace(config.workload, simulation_end_s=None),
    )
    specs = tuple(
        SessionSpec(
            session_id,
            0.0,
            4,
            (TurnSpec(0, 0, 2, 0.0),),
        )
        for session_id in range(2)
    )

    runtime = AgenticRuntime(config, specs)
    result = runtime.run()

    assert result.summary["completed_sessions"] == 2
    assert result.summary["attention_retries"] == 1
    assert result.summary["lru_evictions"] == 1
    assert all(item.capacity_utilization <= 1.0 for item in result.attention_devices)
    assert all(session.is_finished for session in runtime.sessions.values())


def test_legacy_simulation_and_retired_modules_are_not_supported() -> None:
    with pytest.raises(TypeError, match="typed three-file"):
        Simulation({})  # type: ignore[arg-type]

    assert importlib.util.find_spec("agentic_sim.runtime.task_builder") is None
    assert importlib.util.find_spec("agentic_sim.runtime.reference_policy") is None
    assert importlib.util.find_spec("agentic_sim.metrics.request_metrics") is None
