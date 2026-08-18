from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from agentic_sim.config import (
    ConfigurationError,
    SimulationConfig,
    load_config,
    load_hardware_config,
    load_mapping_config,
    load_workload_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIRECTORY = REPOSITORY_ROOT / "configs" / "example"
EXAMPLE_PATHS = (
    EXAMPLE_DIRECTORY / "hardware.yaml",
    EXAMPLE_DIRECTORY / "workload.yaml",
    EXAMPLE_DIRECTORY / "mapping.yaml",
)


def example_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    documents = []
    for path in EXAMPLE_PATHS:
        with path.open(encoding="utf-8") as stream:
            documents.append(yaml.safe_load(stream))
    return documents[0], documents[1], documents[2]


def write_documents(
    tmp_path: Path,
    hardware: dict[str, Any],
    workload: dict[str, Any],
    mapping: dict[str, Any],
) -> tuple[Path, Path, Path]:
    paths = tuple(tmp_path / name for name in ("hardware.yaml", "workload.yaml", "mapping.yaml"))
    for path, document in zip(paths, (hardware, workload, mapping), strict=True):
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return paths


def load_documents(
    tmp_path: Path,
    hardware: dict[str, Any],
    workload: dict[str, Any],
    mapping: dict[str, Any],
) -> SimulationConfig:
    return load_config(*write_documents(tmp_path, hardware, workload, mapping))


def test_three_examples_load_into_one_immutable_config() -> None:
    config = load_config(*EXAMPLE_PATHS)

    assert isinstance(config, SimulationConfig)
    assert config.hardware.devices[0].id == 0
    assert config.workload.source.num_sessions == 8
    assert config.mapping.model.num_experts == 8
    assert isinstance(config.hardware.devices, tuple)
    assert isinstance(config.mapping.decode_ffn.experts_by_device, tuple)
    with pytest.raises(FrozenInstanceError):
        config.workload.seed = 8  # type: ignore[misc]


def test_files_load_independently() -> None:
    hardware = load_hardware_config(EXAMPLE_PATHS[0])
    workload = load_workload_config(EXAMPLE_PATHS[1])
    mapping = load_mapping_config(EXAMPLE_PATHS[2])

    assert len(hardware.devices) == 5
    assert workload.seed == 7
    assert mapping.granularity == "decode_step"


def test_old_one_file_loader_shape_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="requires three configuration files"):
        load_config(REPOSITORY_ROOT / "configs" / "example.yaml")


@pytest.mark.parametrize(
    ("document_index", "mutation", "message"),
    [
        (
            0,
            lambda document: document["devices"][0].update({"prefill_tokens_per_second": 1}),
            r"hardware\.devices\[0\]: unknown field.*prefill_tokens_per_second",
        ),
        (
            1,
            lambda document: document["source"].update({"input_tokens": {"type": "constant", "value": 1}}),
            r"workload\.source: unknown field.*input_tokens",
        ),
        (
            2,
            lambda document: document.update({"policy": {"prefill_device": 0}}),
            r"mapping: unknown field.*policy",
        ),
    ],
)
def test_unknown_fields_and_legacy_aliases_are_rejected(
    tmp_path: Path,
    document_index: int,
    mutation,
    message: str,
) -> None:
    documents = list(example_documents())
    mutation(documents[document_index])
    paths = write_documents(tmp_path, *documents)

    loader = (load_hardware_config, load_workload_config, load_mapping_config)[document_index]
    with pytest.raises(ConfigurationError, match=message):
        loader(paths[document_index])


def test_duplicate_device_ids_are_rejected(tmp_path: Path) -> None:
    hardware, workload, mapping = example_documents()
    hardware["devices"][1]["id"] = 0

    with pytest.raises(ConfigurationError, match=r"hardware\.devices\[1\]\.id: duplicate device ID 0"):
        load_documents(tmp_path, hardware, workload, mapping)


def test_missing_required_interconnect_is_path_specific(tmp_path: Path) -> None:
    hardware, _, _ = example_documents()
    del hardware["interconnects"]["kv_p2p"]
    path = write_documents(tmp_path, hardware, {}, {})[0]

    with pytest.raises(
        ConfigurationError,
        match=r"hardware\.interconnects: missing required field.*kv_p2p",
    ):
        load_hardware_config(path)


@pytest.mark.parametrize(
    ("device_ids", "message"),
    [
        ([99], r"mapping\.decode_attn\.device_ids\[0\]: references unknown hardware device 99"),
        ([0], r"hardware device 0 does not provide decode_attn"),
    ],
)
def test_attention_references_require_existing_capable_devices(
    tmp_path: Path,
    device_ids: list[int],
    message: str,
) -> None:
    hardware, workload, mapping = example_documents()
    mapping["decode_attn"]["device_ids"] = device_ids

    with pytest.raises(ConfigurationError, match=message):
        load_documents(tmp_path, hardware, workload, mapping)


def test_prefill_and_attention_mapping_cannot_overlap_in_v1(tmp_path: Path) -> None:
    hardware, workload, mapping = example_documents()
    hardware["devices"][1]["capabilities"]["prefill"] = {
        "model": "aggregate_throughput",
        "tokens_per_second": 40000,
    }
    mapping["prefill"]["device_ids"] = [1]

    with pytest.raises(ConfigurationError, match="does not allow prefill and decode_attn"):
        load_documents(tmp_path, hardware, workload, mapping)


def test_attention_and_ffn_may_share_one_physical_device(tmp_path: Path) -> None:
    hardware, workload, mapping = example_documents()
    hardware["devices"][1]["capabilities"]["decode_ffn"] = {
        "model": "linear_expert_batch",
        "tokens_per_second": 100000,
    }
    mapping["decode_ffn"]["device_ids"] = [1]
    mapping["decode_ffn"]["experts_by_device"] = {1: list(range(8))}

    config = load_documents(tmp_path, hardware, workload, mapping)

    assert config.mapping.decode_attn.device_ids == (1, 2)
    assert config.mapping.decode_ffn.device_ids == (1,)


def test_each_ffn_device_requires_an_expert_assignment(tmp_path: Path) -> None:
    hardware, workload, mapping = example_documents()
    del mapping["decode_ffn"]["experts_by_device"][4]

    with pytest.raises(ConfigurationError, match="missing expert assignment for FFN device 4"):
        load_documents(tmp_path, hardware, workload, mapping)


def test_expert_assignment_cannot_reference_unlisted_device(tmp_path: Path) -> None:
    hardware, workload, mapping = example_documents()
    mapping["decode_ffn"]["experts_by_device"][99] = mapping["decode_ffn"]["experts_by_device"].pop(4)

    with pytest.raises(ConfigurationError, match=r"experts_by_device\[99\].*not listed"):
        load_documents(tmp_path, hardware, workload, mapping)


def test_uncovered_experts_are_rejected(tmp_path: Path) -> None:
    hardware, workload, mapping = example_documents()
    mapping["decode_ffn"]["experts_by_device"][4] = [4, 5, 6]

    with pytest.raises(ConfigurationError, match="uncovered expert IDs: 7"):
        load_documents(tmp_path, hardware, workload, mapping)


def test_duplicate_expert_ownership_is_rejected(tmp_path: Path) -> None:
    hardware, workload, mapping = example_documents()
    mapping["decode_ffn"]["experts_by_device"][4] = [3, 4, 5, 6, 7]

    with pytest.raises(ConfigurationError, match="expert 3 is already owned by device 3"):
        load_documents(tmp_path, hardware, workload, mapping)


def test_out_of_range_expert_is_rejected(tmp_path: Path) -> None:
    hardware, workload, mapping = example_documents()
    mapping["decode_ffn"]["experts_by_device"][4] = [4, 5, 6, 8]

    with pytest.raises(ConfigurationError, match=r"expert 8 is outside \[0, 8\)"):
        load_documents(tmp_path, hardware, workload, mapping)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("source", "context_tokens", "min"), -1, r"context_tokens\.min: must be at least 0"),
        (("source", "turns", "value"), 0, r"turns\.value: must be at least 1"),
        (("source", "arrival", "rate_per_second"), 0, r"arrival\.rate_per_second: must be greater than 0"),
        (("simulation_end_s",), float("inf"), r"simulation_end_s: must be a finite number"),
    ],
)
def test_workload_numeric_validation_is_strict(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    _, workload, _ = example_documents()
    target: dict[str, Any] = workload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    workload_path = write_documents(tmp_path, {}, workload, {})[1]

    with pytest.raises(ConfigurationError, match=message):
        load_workload_config(workload_path)


def test_discrete_probabilities_must_sum_to_one(tmp_path: Path) -> None:
    _, workload, _ = example_documents()
    workload["source"]["turns"] = {
        "type": "discrete",
        "values": [1, 2],
        "probabilities": [0.2, 0.2],
    }
    workload_path = write_documents(tmp_path, {}, workload, {})[1]

    with pytest.raises(ConfigurationError, match=r"turns\.probabilities: must sum to 1"):
        load_workload_config(workload_path)


def test_integer_token_distributions_reject_continuous_variants(tmp_path: Path) -> None:
    _, workload, _ = example_documents()
    workload["source"]["output_tokens"] = {"type": "uniform", "min": 1, "max": 2}
    workload_path = write_documents(tmp_path, {}, workload, {})[1]

    with pytest.raises(ConfigurationError, match="integer quantities support"):
        load_workload_config(workload_path)


def test_pool_is_unbounded_and_uses_one_symmetric_rate(tmp_path: Path) -> None:
    hardware, _, _ = example_documents()
    hardware["kv_pool"]["capacity_tokens"] = 100
    hardware_path = write_documents(tmp_path, hardware, {}, {})[0]
    with pytest.raises(ConfigurationError, match="finite capacity is not supported"):
        load_hardware_config(hardware_path)

    hardware, _, _ = example_documents()
    hardware["kv_pool"]["offload_tokens_per_second"] = 2_000_000
    hardware_path = write_documents(tmp_path, hardware, {}, {})[0]
    with pytest.raises(ConfigurationError, match="must be equal in V1"):
        load_hardware_config(hardware_path)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "hardware.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="duplicate YAML key 'schema_version'"):
        load_hardware_config(path)


def test_prefill_state_and_placement_must_match(tmp_path: Path) -> None:
    _, _, mapping = example_documents()
    mapping["prefill"]["state"] = "stateless"
    mapping_path = write_documents(tmp_path, {}, {}, mapping)[2]

    with pytest.raises(ConfigurationError, match="stateless prefill requires round_robin"):
        load_mapping_config(mapping_path)


def test_examples_comment_every_implemented_discriminator_value() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in EXAMPLE_PATHS)
    expected_values = {
        "aggregate_throughput",
        "linear_context",
        "linear_expert_batch",
        "startup_min_package",
        "startup_throughput",
        "synthetic",
        "poisson",
        "constant",
        "bounded_integer",
        "discrete",
        "uniform",
        "exponential",
        "lognormal",
        "decode_step",
        "stateful",
        "stateless",
        "sticky_round_robin",
        "round_robin",
        "least_active_context",
        "active_only",
        "lru_cache",
        "oracle_output",
    }
    for value in expected_values:
        assert value in text
