from copy import deepcopy

import pytest

from agentic_sim.core.simulation import Simulation
from agentic_sim.workload.workload_generator import WorkloadGenerator


def context_workload(minimal_config: dict) -> dict:
    config = deepcopy(minimal_config)
    config["workload"]["num_agents"] = 1
    config["workload"]["turns"] = {"type": "constant", "value": 3}
    config["workload"]["context_tokens"] = {"type": "constant", "value": 5}
    config["workload"]["input_tokens"] = {"type": "constant", "value": 2}
    config["workload"]["output_tokens"] = {"type": "constant", "value": 1}
    return config


def test_context_tokens_are_first_prompt_and_inputs_begin_on_turn_two(minimal_config: dict) -> None:
    config = context_workload(minimal_config)
    agent = WorkloadGenerator(config["workload"], seed=1).generate()[0]

    assert agent.initial_context_tokens == 5
    assert agent.input_tokens_per_turn == [0, 2, 2]
    assert agent.current_input_tokens == 5
    agent.complete_current_turn()
    agent.current_turn = 1
    assert agent.current_input_tokens == 2
    agent.complete_current_turn()
    agent.current_turn = 2
    agent.complete_current_turn()
    assert agent.current_context_tokens == 12


@pytest.mark.parametrize(
    ("retention", "prefill_tokens"),
    [("memory", [5, 2, 2]), ("discard", [5, 8, 11])],
)
def test_initial_context_drives_prefill_and_context_growth(
    minimal_config: dict, retention: str, prefill_tokens: list[int]
) -> None:
    config = context_workload(minimal_config)
    config["invocation_model"]["kv"]["retention"] = retention
    result = Simulation(config).run()

    assert result.summary["completed_agents"] == 1
    assert result.all_agents[0].current_context_tokens == 12
    assert [batch.total_tokens for batch in result.prefill_batches] == prefill_tokens
    assert [flow.byte_count for flow in result.ingress_flows] == [50, 80, 110]
