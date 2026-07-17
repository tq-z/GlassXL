"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentic_sim.workload.invocation import KVRetentionMode


class ConfigurationError(ValueError):
    """Raised when a simulator configuration is malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    if not isinstance(loaded, dict):
        raise ConfigurationError("configuration root must be a mapping")
    validate_config(loaded)
    return loaded


def validate_config(config: dict[str, Any]) -> None:
    """Validate fields required by the coarse P-D simulator."""

    required = ("seed", "hardware", "workload", "invocation_model", "policy")
    missing = [field for field in required if field not in config]
    if missing:
        raise ConfigurationError(f"missing required configuration fields: {missing}")
    if not isinstance(config["seed"], int):
        raise ConfigurationError("seed must be an integer")
    if float(config.get("simulation_end_s", 0.0)) < 0.0:
        raise ConfigurationError("simulation_end_s must be non-negative")

    workload = config["workload"]
    if int(workload.get("num_agents", 0)) < 0:
        raise ConfigurationError("workload.num_agents must be non-negative")
    for name in ("arrival", "turns", "input_tokens", "output_tokens", "tool_wait_ms"):
        if name not in workload:
            raise ConfigurationError(f"workload.{name} is required")

    hardware = config["hardware"]
    devices = hardware.get("devices", [])
    memories = hardware.get("memories", [])
    if not isinstance(devices, list) or not isinstance(memories, list):
        raise ConfigurationError("hardware devices and memories must be lists")
    compute_devices = [item for item in devices if str(item.get("class", "compute")) != "memory"]
    if len(compute_devices) < 2:
        raise ConfigurationError("at least two compute devices are required for P-D decoupling")
    for device in compute_devices:
        for field in ("prefill_tokens_per_second", "decode_tokens_per_second"):
            if field not in device or float(device[field]) <= 0.0:
                raise ConfigurationError(f"compute device {device.get('id')} requires positive {field}")

    invocation_model = config["invocation_model"]
    if not isinstance(invocation_model, dict) or not isinstance(invocation_model.get("kv"), dict):
        raise ConfigurationError("invocation_model.kv is required")
    kv_config = invocation_model["kv"]
    try:
        KVRetentionMode(str(kv_config["retention"]))
    except (KeyError, ValueError) as error:
        raise ConfigurationError("kv.retention must be memory, far_memory, or discard") from error
    if "bytes_per_token" not in kv_config or int(kv_config["bytes_per_token"]) < 0:
        raise ConfigurationError("kv.bytes_per_token must be non-negative")

    policy = config["policy"]
    if str(policy.get("type", "static_fcfs")) != "static_fcfs":
        raise ConfigurationError("only the static_fcfs policy is supported")
    for field in ("prefill_device", "decode_device"):
        if field not in policy:
            raise ConfigurationError(f"policy.{field} is required")
    if int(policy["prefill_device"]) == int(policy["decode_device"]):
        raise ConfigurationError("prefill and decode devices must be distinct")
    retention = KVRetentionMode(str(kv_config["retention"]))
    required_resource = {
        KVRetentionMode.MEMORY: "memory_resource",
        KVRetentionMode.FAR_MEMORY: "far_memory_resource",
    }.get(retention)
    if required_resource is not None and required_resource not in policy:
        raise ConfigurationError(f"policy.{required_resource} is required for {retention.value}")
