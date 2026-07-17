from copy import deepcopy

import pytest

from agentic_sim.config import ConfigurationError, validate_config


def test_invalid_retention_is_rejected(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["invocation_model"]["kv"]["retention"] = "unknown"
    with pytest.raises(ConfigurationError, match="retention"):
        validate_config(config)


def test_nonpositive_compute_speed_is_rejected(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["hardware"]["devices"][0]["prefill_tokens_per_second"] = 0
    with pytest.raises(ConfigurationError, match="prefill_tokens_per_second"):
        validate_config(config)


def test_same_prefill_and_decode_device_is_rejected(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    config["policy"]["decode_device"] = 0
    with pytest.raises(ConfigurationError, match="distinct"):
        validate_config(config)


def test_selected_retention_resource_is_required(minimal_config: dict) -> None:
    config = deepcopy(minimal_config)
    del config["policy"]["memory_resource"]
    with pytest.raises(ConfigurationError, match="memory_resource"):
        validate_config(config)
