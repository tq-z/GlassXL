"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentic_sim.workload.invocation import KVRetentionMode


class ConfigurationError(ValueError):
    """Raised when a simulator configuration is malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML mapping from *path*."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    if not isinstance(loaded, dict):
        raise ConfigurationError("configuration root must be a mapping")
    validate_config(loaded)
    return loaded


def _positive(mapping: dict[str, Any], field: str, location: str) -> None:
    if field not in mapping or float(mapping[field]) <= 0.0:
        raise ConfigurationError(f"{location} requires positive {field}")


def _validate_compute_device(device: dict[str, Any]) -> None:
    location = f"compute device {device.get('id')}"
    prefill = device.get("prefill")
    decode = device.get("decode")
    has_legacy = "prefill_tokens_per_second" in device or "decode_tokens_per_second" in device
    if has_legacy:
        for field in ("prefill_tokens_per_second", "decode_tokens_per_second"):
            _positive(device, field, location)
    if prefill is not None:
        if not isinstance(prefill, dict):
            raise ConfigurationError(f"{location}.prefill must be a mapping")
        for field in ("single_request_tokens_per_second", "max_system_tokens_per_second"):
            _positive(prefill, field, f"{location}.prefill")
        for field in ("max_batch_requests", "max_batch_tokens"):
            if field not in prefill or int(prefill[field]) <= 0:
                raise ConfigurationError(f"{location}.prefill requires positive {field}")
        if str(prefill.get("layout", "packed")) != "packed":
            raise ConfigurationError("Phase 1.5 supports only prefill.layout=packed")
        if float(prefill.get("batch_wait_s", 0.0)) != 0.0 or float(prefill.get("batch_wait_us", 0.0)) != 0.0:
            raise ConfigurationError("Phase 1.5 requires prefill.batch_wait_us=0")
    if decode is not None:
        if not isinstance(decode, dict):
            raise ConfigurationError(f"{location}.decode must be a mapping")
        for field in ("single_session_tokens_per_second", "max_system_tokens_per_second"):
            _positive(decode, field, f"{location}.decode")
        if "max_active_sessions" not in decode or int(decode["max_active_sessions"]) <= 0:
            raise ConfigurationError(f"{location}.decode requires positive max_active_sessions")
    if not has_legacy and prefill is None and decode is None:
        raise ConfigurationError(f"{location} requires prefill and/or decode settings")
    if float(device.get("fixed_latency_s", 0.0)) < 0.0 or float(device.get("fixed_latency_us", 0.0)) < 0.0:
        raise ConfigurationError(f"{location} fixed latency must be non-negative")


def validate_config(config: dict[str, Any]) -> None:
    """Validate Phase 1 and Phase 1.5 coarse P-D configurations."""

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
    if "context_tokens" in workload and not isinstance(workload["context_tokens"], dict):
        raise ConfigurationError("workload.context_tokens must be a distribution mapping")

    hardware = config["hardware"]
    devices = hardware.get("devices", [])
    memories = hardware.get("memories", [])
    if not isinstance(devices, list) or not isinstance(memories, list):
        raise ConfigurationError("hardware devices and memories must be lists")
    compute_devices = [item for item in devices if str(item.get("class", "compute")) != "memory"]
    if len(compute_devices) < 2:
        raise ConfigurationError("at least two compute devices are required for P-D decoupling")
    for device in compute_devices:
        _validate_compute_device(device)

    ingress = hardware.get("kv_ingress")
    if ingress is not None:
        if not isinstance(ingress, dict):
            raise ConfigurationError("hardware.kv_ingress must be a mapping")
        for field in (
            "per_session_max_bandwidth_bytes_per_sec",
            "max_system_bandwidth_bytes_per_sec",
        ):
            _positive(ingress, field, "hardware.kv_ingress")
        if float(ingress.get("fixed_latency_s", 0.0)) < 0.0 or float(ingress.get("fixed_latency_us", 0.0)) < 0.0:
            raise ConfigurationError("hardware.kv_ingress fixed latency must be non-negative")

    invocation_model = config["invocation_model"]
    if not isinstance(invocation_model, dict) or not isinstance(invocation_model.get("kv"), dict):
        raise ConfigurationError("invocation_model.kv is required")
    kv_config = invocation_model["kv"]
    try:
        retention = KVRetentionMode(str(kv_config["retention"]))
    except (KeyError, ValueError) as error:
        raise ConfigurationError("kv.retention must be same_device, memory, far_memory, or discard") from error
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
    required_resource = {
        KVRetentionMode.MEMORY: "memory_resource",
        KVRetentionMode.FAR_MEMORY: "far_memory_resource",
    }.get(retention)
    if required_resource is not None and required_resource not in policy:
        raise ConfigurationError(f"policy.{required_resource} is required for {retention.value}")
