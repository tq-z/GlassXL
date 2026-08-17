from copy import deepcopy

from agentic_sim.core.task import ComputePhase, TaskType
from agentic_sim.core.simulation import Simulation
from agentic_sim.runtime.task_builder import TaskBuilder
from agentic_sim.workload.invocation import Invocation


def test_same_device_retains_kv_without_store_or_restore_tasks() -> None:
    builder = TaskBuilder({"kv": {"bytes_per_token": 10, "retention": "same_device"}})
    invocation = Invocation(0, 0, 1, 2, 1, 3, 0.0)

    tasks = builder.build(invocation)

    assert builder.retains_kv
    assert not builder.uses_memory_tier
    assert [task.task_type for task in tasks] == [
        TaskType.COMPUTE,
        TaskType.TRANSFER,
        TaskType.COMPUTE,
    ]
    assert tasks[0].metadata == {"phase": ComputePhase.PREFILL, "tokens": 2}
    assert tasks[1].metadata == {"bytes": 50, "direction": "prefill_to_decode"}
    assert builder.build_restore(invocation) == []
    assert builder.build_store(0, 0, 6) == []


def test_same_device_runs_without_memory_resources_or_retention_links(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["invocation_model"]["kv"]["retention"] = "same_device"
    config["policy"].pop("memory_resource")
    config["policy"].pop("far_memory_resource")
    config["hardware"]["memories"] = []
    config["hardware"]["links"] = [
        link
        for link in config["hardware"]["links"]
        if (link["src"], link["dst"]) == (0, 1)
    ]

    result = Simulation(config).run()

    assert result.summary["completed_agents"] == 2
    assert result.summary["average_kv_wait_agents"] == 0.0
    assert all(agent.current_context_tokens == 6 for agent in result.all_agents)
    assert all(resource.resource_type != "memory" for resource in result.resources)
    assert sum(flow.byte_count for flow in result.ingress_flows) > 0
