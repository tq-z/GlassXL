"""Cross-file validation for the V1 PD/AFD configuration."""

from __future__ import annotations

from .models import ConfigurationError, SimulationConfig


def validate_simulation_config(config: SimulationConfig) -> None:
    """Validate capability placement and expert ownership across all files."""

    devices = {device.id: device for device in config.hardware.devices}

    def require_capability(device_ids: tuple[int, ...], capability: str, path: str) -> None:
        for index, device_id in enumerate(device_ids):
            device = devices.get(device_id)
            item_path = f"{path}[{index}]"
            if device is None:
                raise ConfigurationError(f"{item_path}: references unknown hardware device {device_id}")
            if getattr(device.capabilities, capability) is None:
                raise ConfigurationError(
                    f"{item_path}: hardware device {device_id} does not provide {capability}"
                )

    require_capability(config.mapping.prefill.device_ids, "prefill", "mapping.prefill.device_ids")
    require_capability(config.mapping.decode_attn.device_ids, "decode_attn", "mapping.decode_attn.device_ids")
    require_capability(config.mapping.decode_ffn.device_ids, "decode_ffn", "mapping.decode_ffn.device_ids")

    prefill_decode_overlap = set(config.mapping.prefill.device_ids) & set(config.mapping.decode_attn.device_ids)
    if prefill_decode_overlap:
        device_id = min(prefill_decode_overlap)
        raise ConfigurationError(
            f"mapping.prefill.device_ids: V1 does not allow prefill and decode_attn placement on device {device_id}"
        )

    listed_ffn = set(config.mapping.decode_ffn.device_ids)
    placement_devices = [placement.device_id for placement in config.mapping.decode_ffn.experts_by_device]
    if len(set(placement_devices)) != len(placement_devices):
        raise ConfigurationError("mapping.decode_ffn.experts_by_device: duplicate device entry")
    extra = set(placement_devices) - listed_ffn
    if extra:
        device_id = min(extra)
        raise ConfigurationError(
            f"mapping.decode_ffn.experts_by_device[{device_id}]: "
            "device is not listed in mapping.decode_ffn.device_ids"
        )
    missing_devices = listed_ffn - set(placement_devices)
    if missing_devices:
        device_id = min(missing_devices)
        raise ConfigurationError(
            f"mapping.decode_ffn.experts_by_device: missing expert assignment for FFN device {device_id}"
        )

    owner_by_expert: dict[int, int] = {}
    for placement in config.mapping.decode_ffn.experts_by_device:
        for expert_id in placement.expert_ids:
            if expert_id >= config.mapping.model.num_experts:
                raise ConfigurationError(
                    f"mapping.decode_ffn.experts_by_device[{placement.device_id}]: "
                    f"expert {expert_id} is outside [0, {config.mapping.model.num_experts})"
                )
            previous = owner_by_expert.get(expert_id)
            if previous is not None:
                raise ConfigurationError(
                    f"mapping.decode_ffn.experts_by_device[{placement.device_id}]: expert {expert_id} "
                    f"is already owned by device {previous}; replication is not supported in V1"
                )
            owner_by_expert[expert_id] = placement.device_id

    uncovered = sorted(set(range(config.mapping.model.num_experts)) - set(owner_by_expert))
    if uncovered:
        raise ConfigurationError(
            f"mapping.decode_ffn.experts_by_device: uncovered expert IDs: {', '.join(map(str, uncovered))}"
        )
