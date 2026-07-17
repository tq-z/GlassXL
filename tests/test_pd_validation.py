from copy import deepcopy

import pytest

from agentic_sim.config import ConfigurationError, validate_config
from agentic_sim.core.simulation import Simulation


def test_negative_kv_bytes_per_token_is_rejected(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["invocation_model"]["kv"]["bytes_per_token"] = -1
    with pytest.raises(ConfigurationError, match="bytes_per_token"):
        validate_config(config)


def test_unknown_selected_memory_resource_is_rejected(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["policy"]["memory_resource"] = 99
    with pytest.raises(ConfigurationError, match="unknown selected"):
        Simulation(config).run()


def test_missing_retention_link_is_rejected(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["hardware"]["links"] = [
        link
        for link in config["hardware"]["links"]
        if (link["src"], link["dst"]) != (1, 2)
    ]
    with pytest.raises(ConfigurationError, match="decode-to-memory"):
        Simulation(config).run()
