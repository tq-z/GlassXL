from copy import deepcopy

from agentic_sim.workload.agent import AgentState
from agentic_sim.workload.workload_generator import WorkloadGenerator


def test_workload_is_seeded_and_has_per_turn_values(minimal_config: dict) -> None:
    config = deepcopy(minimal_config["workload"])
    first = WorkloadGenerator(config, seed=9).generate()
    second = WorkloadGenerator(config, seed=9).generate()
    assert first == second
    assert all(agent.state is AgentState.NOT_ARRIVED for agent in first)
    assert all(agent.num_turns == 2 for agent in first)
    assert all(len(agent.input_tokens_per_turn) == 2 for agent in first)


def test_agent_context_grows_by_input_and_output(minimal_config: dict) -> None:
    agent = WorkloadGenerator(minimal_config["workload"], seed=1).generate()[0]
    agent.complete_current_turn()
    assert agent.current_context_tokens == 3
