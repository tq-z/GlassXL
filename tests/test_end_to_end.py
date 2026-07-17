from copy import deepcopy

import pytest

from agentic_sim.config import ConfigurationError
from agentic_sim.core.simulation import Simulation
from agentic_sim.workload.agent import AgentState


def resource_by_id(result, resource_id: str):
    return next(resource for resource in result.resources if resource.resource_id == resource_id)


def test_multiturn_retained_simulation_runs_end_to_end(minimal_config: dict) -> None:
    result = Simulation(minimal_config).run()
    assert result.summary["completed_agents"] == 2
    assert result.summary["invocation_throughput_per_s"] > 0.0
    assert result.summary["mean_tool_wait_ms"] == 50.0
    assert result.summary["average_kv_wait_agents"] > 0.0
    assert len(result.invocations) == 4
    assert all(agent.state is AgentState.FINISHED for agent in result.all_agents)
    assert all(agent.current_context_tokens == 6 for agent in result.all_agents)
    assert resource_by_id(result, "compute:0").busy_time_s > 0.0
    assert resource_by_id(result, "compute:1").busy_time_s > 0.0
    assert resource_by_id(result, "memory:2").bytes_processed > 0
    assert resource_by_id(result, "memory:3").bytes_processed == 0
    assert resource_by_id(result, "link:0->1").bytes_processed > 0


def test_tool_wait_and_kv_store_overlap(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["workload"]["num_agents"] = 1
    result = Simulation(config).run()
    first, second = result.invocations
    # 30 bytes: store transfer is .31 s and memory write is .32 s. These overlap
    # the .05 s tool wait, so the inter-turn gap is .63 s rather than .68 s.
    assert second.ready_time_s - first.finish_time_s == pytest.approx(0.63)
    assert result.summary["average_kv_wait_agents"] > 0.0


def test_far_memory_selects_far_tier(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["invocation_model"]["kv"]["retention"] = "far_memory"
    result = Simulation(config).run()
    assert resource_by_id(result, "memory:2").bytes_processed == 0
    assert resource_by_id(result, "memory:3").bytes_processed > 0
    assert resource_by_id(result, "link:1->3").bytes_processed > 0
    assert resource_by_id(result, "link:3->0").bytes_processed > 0


def test_discard_uses_no_memory_or_kv_wait(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["invocation_model"]["kv"]["retention"] = "discard"
    result = Simulation(config).run()
    assert result.summary["average_kv_wait_agents"] == 0.0
    assert all(
        resource.bytes_processed == 0
        for resource in result.resources
        if resource.resource_type == "memory"
    )


def test_simulation_is_deterministic(minimal_config: dict) -> None:
    first = Simulation(deepcopy(minimal_config)).run()
    second = Simulation(deepcopy(minimal_config)).run()
    assert first.summary == second.summary
    assert first.agents == second.agents
    assert first.resources == second.resources


def test_end_time_leaves_agents_incomplete(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["simulation_end_s"] = 0.01
    result = Simulation(config).run()
    assert result.summary["simulation_time_s"] == 0.01
    assert result.summary["completed_agents"] == 0


def test_high_load_creates_compute_queue(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["workload"]["num_agents"] = 8
    config["workload"]["arrival"]["rate_per_second"] = 1_000_000.0
    result = Simulation(config).run()
    assert result.summary["max_compute_queue"] > 0
    assert result.summary["mean_queueing_ms"] > 0.0


def test_missing_required_pd_link_is_rejected(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["hardware"]["links"] = [
        link
        for link in config["hardware"]["links"]
        if (link["src"], link["dst"]) != (0, 1)
    ]
    with pytest.raises(ConfigurationError, match="prefill-to-decode"):
        Simulation(config).run()
