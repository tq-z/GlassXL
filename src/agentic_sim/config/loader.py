"""Strict YAML loaders for hardware, workload, and mapping configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .models import (
    AllToAllModel,
    AttentionModel,
    AttentionPlacement,
    AttentionResidency,
    BatchLatencyPointConfig,
    BoundedIntegerDistributionConfig,
    CapacityReservationConfig,
    ConfigurationError,
    ConstantDistributionConfig,
    DecodeAttentionCapabilityConfig,
    DecodeAttentionMappingConfig,
    DecodeFFNCapabilityConfig,
    DecodeFFNMappingConfig,
    DeviceCapabilitiesConfig,
    DeviceConfig,
    DiscreteDistributionConfig,
    DistributionConfig,
    DistributionType,
    ExponentialDistributionConfig,
    ExpertModel,
    ExpertPlacementConfig,
    FFNAllToAllConfig,
    HardwareConfig,
    InterconnectsConfig,
    KVPoolConfig,
    KVP2PConfig,
    LognormalDistributionConfig,
    MappingConfig,
    ModelShapeConfig,
    P2PModel,
    PoissonArrivalConfig,
    PrefillCapabilityConfig,
    PrefillMappingConfig,
    PrefillModel,
    PrefillPlacement,
    PrefillState,
    SimulationConfig,
    SyntheticSourceConfig,
    UniformDistributionConfig,
    WorkloadConfig,
)
from .validation import validate_simulation_config


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that does not silently overwrite duplicate keys."""


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConfigurationError("YAML mapping keys must be scalar values") from error
        if duplicate:
            raise ConfigurationError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)

E = TypeVar("E")


def _read_yaml(path: str | Path, root_name: str) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            loaded = yaml.load(stream, Loader=_UniqueKeyLoader)
    except ConfigurationError:
        raise
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"{root_name}: cannot load {config_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"{root_name}: configuration root must be a mapping")
    if any(not isinstance(key, str) for key in loaded):
        raise ConfigurationError(f"{root_name}: root keys must be strings")
    return config_path, loaded


def _mapping(value: object, path: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path}: must be a mapping")
    return value


def _sequence(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{path}: must be a list")
    return value


def _fields(
    value: Mapping[Any, Any],
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    non_string = [key for key in value if not isinstance(key, str)]
    if non_string:
        raise ConfigurationError(f"{path}: field names must be strings, got {non_string[0]!r}")
    unknown = sorted(set(value) - required - optional)
    if unknown:
        raise ConfigurationError(f"{path}: unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise ConfigurationError(f"{path}: missing required field(s): {', '.join(missing)}")


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path}: must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{path}: must be at least {minimum}")
    return value


def _number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path}: must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{path}: must be a finite number")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        relation = "greater than" if strict_minimum else "at least"
        if invalid:
            raise ConfigurationError(f"{path}: must be {relation} {minimum:g}")
    return result


def _literal(value: object, path: str, allowed: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ConfigurationError(f"{path}: must be one of {', '.join(allowed)}")
    return value


def _enum(value: object, path: str, enum_type: type[E]) -> E:
    allowed = tuple(item.value for item in enum_type)  # type: ignore[attr-defined]
    literal = _literal(value, path, allowed)
    return enum_type(literal)  # type: ignore[call-arg]


def _schema_version(root: Mapping[str, Any], path: str) -> int:
    version = _integer(root["schema_version"], f"{path}.schema_version")
    if version != 1:
        raise ConfigurationError(f"{path}.schema_version: only version 1 is supported")
    return version


def _device_ids(value: object, path: str) -> tuple[int, ...]:
    raw = _sequence(value, path)
    if not raw:
        raise ConfigurationError(f"{path}: must contain at least one device ID")
    ids = tuple(_integer(item, f"{path}[{index}]", minimum=0) for index, item in enumerate(raw))
    if len(set(ids)) != len(ids):
        raise ConfigurationError(f"{path}: duplicate device ID")
    return ids


def _parse_prefill_capability(value: object, path: str) -> PrefillCapabilityConfig:
    raw = _mapping(value, path)
    _fields(raw, path, {"model", "tokens_per_second"})
    return PrefillCapabilityConfig(
        model=_enum(raw["model"], f"{path}.model", PrefillModel),
        tokens_per_second=_number(
            raw["tokens_per_second"],
            f"{path}.tokens_per_second",
            minimum=0,
            strict_minimum=True,
        ),
    )


def _parse_attention_capability(value: object, path: str) -> DecodeAttentionCapabilityConfig:
    raw = _mapping(value, path)
    _fields(raw, path, {"model", "context_tokens_per_second", "kv_capacity_tokens"})
    return DecodeAttentionCapabilityConfig(
        model=_enum(raw["model"], f"{path}.model", AttentionModel),
        context_tokens_per_second=_number(
            raw["context_tokens_per_second"],
            f"{path}.context_tokens_per_second",
            minimum=0,
            strict_minimum=True,
        ),
        kv_capacity_tokens=_integer(
            raw["kv_capacity_tokens"],
            f"{path}.kv_capacity_tokens",
            minimum=0,
        ),
    )


def _parse_ffn_capability(value: object, path: str) -> DecodeFFNCapabilityConfig:
    raw = _mapping(value, path)
    if "model" not in raw:
        raise ConfigurationError(f"{path}: missing required field(s): model")
    model = _enum(raw["model"], f"{path}.model", ExpertModel)
    if model is ExpertModel.LINEAR_EXPERT_BATCH:
        _fields(raw, path, {"model", "tokens_per_second"})
        return DecodeFFNCapabilityConfig(
            model=model,
            tokens_per_second=_number(
                raw["tokens_per_second"],
                f"{path}.tokens_per_second",
                minimum=0,
                strict_minimum=True,
            ),
        )

    _fields(raw, path, {"model", "points"})
    points_raw = _sequence(raw["points"], f"{path}.points")
    if len(points_raw) < 2:
        raise ConfigurationError(f"{path}.points: must contain at least two points")
    points: list[BatchLatencyPointConfig] = []
    for index, value in enumerate(points_raw):
        point_path = f"{path}.points[{index}]"
        point_raw = _mapping(value, point_path)
        _fields(point_raw, point_path, {"activation_tokens", "latency_us"})
        points.append(
            BatchLatencyPointConfig(
                activation_tokens=_integer(
                    point_raw["activation_tokens"],
                    f"{point_path}.activation_tokens",
                    minimum=0,
                ),
                latency_us=_number(
                    point_raw["latency_us"],
                    f"{point_path}.latency_us",
                    minimum=0,
                ),
            )
        )
    if points[0].activation_tokens != 0 or points[0].latency_us != 0.0:
        raise ConfigurationError(
            f"{path}.points: first point must be activation_tokens=0, latency_us=0"
        )
    for previous, current in zip(points, points[1:]):
        if current.activation_tokens <= previous.activation_tokens:
            raise ConfigurationError(
                f"{path}.points: activation_tokens must be strictly increasing"
            )
        if current.latency_us < previous.latency_us:
            raise ConfigurationError(
                f"{path}.points: latency_us must be nondecreasing"
            )
    if points[-1].latency_us <= points[-2].latency_us:
        raise ConfigurationError(
            f"{path}.points: final segment must have positive slope"
        )
    return DecodeFFNCapabilityConfig(model=model, points=tuple(points))


def load_hardware_config(path: str | Path) -> HardwareConfig:
    """Load one strict hardware.yaml file without resolving mapping references."""

    source_path, root = _read_yaml(path, "hardware")
    _fields(root, "hardware", {"schema_version", "devices", "kv_pool", "interconnects"})
    devices_raw = _sequence(root["devices"], "hardware.devices")
    if not devices_raw:
        raise ConfigurationError("hardware.devices: must contain at least one device")

    devices: list[DeviceConfig] = []
    seen_ids: set[int] = set()
    for index, value in enumerate(devices_raw):
        device_path = f"hardware.devices[{index}]"
        raw = _mapping(value, device_path)
        _fields(raw, device_path, {"id", "capabilities"})
        device_id = _integer(raw["id"], f"{device_path}.id", minimum=0)
        if device_id in seen_ids:
            raise ConfigurationError(f"{device_path}.id: duplicate device ID {device_id}")
        seen_ids.add(device_id)

        capabilities_path = f"{device_path}.capabilities"
        capabilities_raw = _mapping(raw["capabilities"], capabilities_path)
        _fields(
            capabilities_raw,
            capabilities_path,
            set(),
            {"prefill", "decode_attn", "decode_ffn"},
        )
        if not capabilities_raw:
            raise ConfigurationError(f"{capabilities_path}: must not be empty")
        devices.append(
            DeviceConfig(
                id=device_id,
                capabilities=DeviceCapabilitiesConfig(
                    prefill=_parse_prefill_capability(
                        capabilities_raw["prefill"],
                        f"{capabilities_path}.prefill",
                    )
                    if "prefill" in capabilities_raw
                    else None,
                    decode_attn=_parse_attention_capability(
                        capabilities_raw["decode_attn"],
                        f"{capabilities_path}.decode_attn",
                    )
                    if "decode_attn" in capabilities_raw
                    else None,
                    decode_ffn=_parse_ffn_capability(
                        capabilities_raw["decode_ffn"],
                        f"{capabilities_path}.decode_ffn",
                    )
                    if "decode_ffn" in capabilities_raw
                    else None,
                ),
            )
        )

    pool_raw = _mapping(root["kv_pool"], "hardware.kv_pool")
    _fields(
        pool_raw,
        "hardware.kv_pool",
        {"load_tokens_per_second", "offload_tokens_per_second", "start_latency_us"},
        {"capacity_tokens"},
    )
    capacity = pool_raw.get("capacity_tokens")
    if capacity is not None:
        raise ConfigurationError(
            "hardware.kv_pool.capacity_tokens: finite capacity is not supported in V1; use null"
        )
    load_rate = _number(
        pool_raw["load_tokens_per_second"],
        "hardware.kv_pool.load_tokens_per_second",
        minimum=0,
        strict_minimum=True,
    )
    offload_rate = _number(
        pool_raw["offload_tokens_per_second"],
        "hardware.kv_pool.offload_tokens_per_second",
        minimum=0,
        strict_minimum=True,
    )
    if load_rate != offload_rate:
        raise ConfigurationError(
            "hardware.kv_pool: load_tokens_per_second and offload_tokens_per_second "
            "must be equal in V1"
        )
    pool = KVPoolConfig(
        capacity_tokens=None,
        load_tokens_per_second=load_rate,
        offload_tokens_per_second=offload_rate,
        start_latency_us=_number(
            pool_raw["start_latency_us"],
            "hardware.kv_pool.start_latency_us",
            minimum=0,
        ),
    )

    interconnects_raw = _mapping(root["interconnects"], "hardware.interconnects")
    _fields(
        interconnects_raw,
        "hardware.interconnects",
        {"ffn_all_to_all", "kv_p2p"},
    )
    all_to_all_path = "hardware.interconnects.ffn_all_to_all"
    all_to_all_raw = _mapping(interconnects_raw["ffn_all_to_all"], all_to_all_path)
    _fields(
        all_to_all_raw,
        all_to_all_path,
        {"model", "start_latency_us", "min_package_tokens", "throughput_tokens_per_second"},
    )
    all_to_all = FFNAllToAllConfig(
        model=_enum(all_to_all_raw["model"], f"{all_to_all_path}.model", AllToAllModel),
        start_latency_us=_number(
            all_to_all_raw["start_latency_us"],
            f"{all_to_all_path}.start_latency_us",
            minimum=0,
        ),
        min_package_tokens=_integer(
            all_to_all_raw["min_package_tokens"],
            f"{all_to_all_path}.min_package_tokens",
            minimum=0,
        ),
        throughput_tokens_per_second=_number(
            all_to_all_raw["throughput_tokens_per_second"],
            f"{all_to_all_path}.throughput_tokens_per_second",
            minimum=0,
            strict_minimum=True,
        ),
    )

    p2p_path = "hardware.interconnects.kv_p2p"
    p2p_raw = _mapping(interconnects_raw["kv_p2p"], p2p_path)
    _fields(
        p2p_raw,
        p2p_path,
        {"model", "start_latency_us", "throughput_tokens_per_second"},
    )
    p2p = KVP2PConfig(
        model=_enum(p2p_raw["model"], f"{p2p_path}.model", P2PModel),
        start_latency_us=_number(
            p2p_raw["start_latency_us"],
            f"{p2p_path}.start_latency_us",
            minimum=0,
        ),
        throughput_tokens_per_second=_number(
            p2p_raw["throughput_tokens_per_second"],
            f"{p2p_path}.throughput_tokens_per_second",
            minimum=0,
            strict_minimum=True,
        ),
    )
    return HardwareConfig(
        schema_version=_schema_version(root, "hardware"),
        devices=tuple(devices),
        kv_pool=pool,
        interconnects=InterconnectsConfig(all_to_all, p2p),
        source_path=source_path,
    )


def _parse_distribution(
    value: object,
    path: str,
    *,
    integer_only: bool,
    minimum: float,
) -> DistributionConfig:
    raw = _mapping(value, path)
    if "type" not in raw:
        raise ConfigurationError(f"{path}: missing required field(s): type")
    distribution_type = _enum(raw["type"], f"{path}.type", DistributionType)
    integer_types = {
        DistributionType.CONSTANT,
        DistributionType.BOUNDED_INTEGER,
        DistributionType.DISCRETE,
    }
    if integer_only and distribution_type not in integer_types:
        raise ConfigurationError(
            f"{path}.type: integer quantities support constant, bounded_integer, or discrete"
        )

    if distribution_type is DistributionType.CONSTANT:
        _fields(raw, path, {"type", "value"})
        parsed: int | float
        if integer_only:
            parsed = _integer(raw["value"], f"{path}.value", minimum=math.ceil(minimum))
        else:
            parsed = _number(raw["value"], f"{path}.value", minimum=minimum)
        return ConstantDistributionConfig(distribution_type, parsed)

    if distribution_type is DistributionType.BOUNDED_INTEGER:
        _fields(raw, path, {"type", "min", "max"})
        lower = _integer(raw["min"], f"{path}.min", minimum=math.ceil(minimum))
        upper = _integer(raw["max"], f"{path}.max", minimum=math.ceil(minimum))
        if lower > upper:
            raise ConfigurationError(f"{path}: min must not exceed max")
        return BoundedIntegerDistributionConfig(distribution_type, lower, upper)

    if distribution_type is DistributionType.DISCRETE:
        _fields(raw, path, {"type", "values", "probabilities"})
        values_raw = _sequence(raw["values"], f"{path}.values")
        probabilities_raw = _sequence(raw["probabilities"], f"{path}.probabilities")
        if not values_raw:
            raise ConfigurationError(f"{path}.values: must not be empty")
        if len(values_raw) != len(probabilities_raw):
            raise ConfigurationError(
                f"{path}: values and probabilities must have the same length"
            )
        if integer_only:
            values: tuple[int | float, ...] = tuple(
                _integer(
                    item,
                    f"{path}.values[{index}]",
                    minimum=math.ceil(minimum),
                )
                for index, item in enumerate(values_raw)
            )
        else:
            values = tuple(
                _number(item, f"{path}.values[{index}]", minimum=minimum)
                for index, item in enumerate(values_raw)
            )
        probabilities = tuple(
            _number(item, f"{path}.probabilities[{index}]", minimum=0)
            for index, item in enumerate(probabilities_raw)
        )
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ConfigurationError(f"{path}.probabilities: must sum to 1")
        return DiscreteDistributionConfig(distribution_type, values, probabilities)

    if distribution_type is DistributionType.UNIFORM:
        _fields(raw, path, {"type", "min", "max"})
        lower = _number(raw["min"], f"{path}.min", minimum=minimum)
        upper = _number(raw["max"], f"{path}.max", minimum=minimum)
        if lower > upper:
            raise ConfigurationError(f"{path}: min must not exceed max")
        return UniformDistributionConfig(distribution_type, lower, upper)

    if distribution_type is DistributionType.EXPONENTIAL:
        _fields(raw, path, {"type", "rate"})
        if minimum > 0:
            raise ConfigurationError(
                f"{path}.type: exponential cannot guarantee values of at least {minimum:g}"
            )
        return ExponentialDistributionConfig(
            distribution_type,
            _number(
                raw["rate"],
                f"{path}.rate",
                minimum=0,
                strict_minimum=True,
            ),
        )

    if distribution_type is DistributionType.LOGNORMAL:
        _fields(raw, path, {"type", "mean", "sigma"}, {"min", "max"})
        lower = (
            _number(raw["min"], f"{path}.min", minimum=minimum)
            if "min" in raw
            else None
        )
        upper = (
            _number(raw["max"], f"{path}.max", minimum=minimum)
            if "max" in raw
            else None
        )
        if lower is not None and upper is not None and lower > upper:
            raise ConfigurationError(f"{path}: min must not exceed max")
        return LognormalDistributionConfig(
            distribution_type,
            _number(raw["mean"], f"{path}.mean"),
            _number(raw["sigma"], f"{path}.sigma", minimum=0),
            lower,
            upper,
        )

    raise AssertionError("all distribution variants are handled")


def load_workload_config(path: str | Path) -> WorkloadConfig:
    """Load one strict workload.yaml file."""

    source_path, root = _read_yaml(path, "workload")
    _fields(
        root,
        "workload",
        {"schema_version", "seed", "source"},
        {"simulation_end_s"},
    )
    source_raw = _mapping(root["source"], "workload.source")
    _fields(
        source_raw,
        "workload.source",
        {
            "type",
            "num_sessions",
            "arrival",
            "turns",
            "context_tokens",
            "prefill_tokens",
            "output_tokens",
            "tool_wait_ms",
        },
    )
    source_type = _literal(
        source_raw["type"],
        "workload.source.type",
        ("synthetic",),
    )
    arrival_raw = _mapping(source_raw["arrival"], "workload.source.arrival")
    _fields(
        arrival_raw,
        "workload.source.arrival",
        {"type", "rate_per_second"},
    )
    arrival_type = _literal(
        arrival_raw["type"],
        "workload.source.arrival.type",
        ("poisson",),
    )
    source = SyntheticSourceConfig(
        type=source_type,
        num_sessions=_integer(
            source_raw["num_sessions"],
            "workload.source.num_sessions",
            minimum=0,
        ),
        arrival=PoissonArrivalConfig(
            arrival_type,
            _number(
                arrival_raw["rate_per_second"],
                "workload.source.arrival.rate_per_second",
                minimum=0,
                strict_minimum=True,
            ),
        ),
        turns=_parse_distribution(
            source_raw["turns"],
            "workload.source.turns",
            integer_only=True,
            minimum=1,
        ),
        context_tokens=_parse_distribution(
            source_raw["context_tokens"],
            "workload.source.context_tokens",
            integer_only=True,
            minimum=0,
        ),
        prefill_tokens=_parse_distribution(
            source_raw["prefill_tokens"],
            "workload.source.prefill_tokens",
            integer_only=True,
            minimum=0,
        ),
        output_tokens=_parse_distribution(
            source_raw["output_tokens"],
            "workload.source.output_tokens",
            integer_only=True,
            minimum=0,
        ),
        tool_wait_ms=_parse_distribution(
            source_raw["tool_wait_ms"],
            "workload.source.tool_wait_ms",
            integer_only=False,
            minimum=0,
        ),
    )
    simulation_end = root.get("simulation_end_s")
    return WorkloadConfig(
        schema_version=_schema_version(root, "workload"),
        seed=_integer(root["seed"], "workload.seed"),
        simulation_end_s=(
            _number(
                simulation_end,
                "workload.simulation_end_s",
                minimum=0,
            )
            if simulation_end is not None
            else None
        ),
        source=source,
        source_path=source_path,
    )


def load_mapping_config(path: str | Path) -> MappingConfig:
    """Load one strict mapping.yaml file without resolving hardware references."""

    source_path, root = _read_yaml(path, "mapping")
    _fields(
        root,
        "mapping",
        {"schema_version", "granularity", "model", "prefill", "decode_attn", "decode_ffn"},
    )
    granularity = _literal(
        root["granularity"],
        "mapping.granularity",
        ("decode_step",),
    )

    model_raw = _mapping(root["model"], "mapping.model")
    _fields(model_raw, "mapping.model", {"num_layers", "num_experts", "top_k"})
    num_experts = _integer(
        model_raw["num_experts"],
        "mapping.model.num_experts",
        minimum=1,
    )
    top_k = _integer(model_raw["top_k"], "mapping.model.top_k", minimum=1)
    if top_k > num_experts:
        raise ConfigurationError(
            "mapping.model.top_k: must not exceed mapping.model.num_experts"
        )
    model = ModelShapeConfig(
        num_layers=_integer(
            model_raw["num_layers"],
            "mapping.model.num_layers",
            minimum=1,
        ),
        num_experts=num_experts,
        top_k=top_k,
    )

    prefill_raw = _mapping(root["prefill"], "mapping.prefill")
    _fields(
        prefill_raw,
        "mapping.prefill",
        {"device_ids", "state", "placement"},
    )
    state = _enum(prefill_raw["state"], "mapping.prefill.state", PrefillState)
    placement = _enum(
        prefill_raw["placement"],
        "mapping.prefill.placement",
        PrefillPlacement,
    )
    required_placement = (
        PrefillPlacement.STICKY_ROUND_ROBIN
        if state is PrefillState.STATEFUL
        else PrefillPlacement.ROUND_ROBIN
    )
    if placement is not required_placement:
        raise ConfigurationError(
            f"mapping.prefill.placement: {state.value} prefill requires "
            f"{required_placement.value}"
        )
    prefill = PrefillMappingConfig(
        _device_ids(prefill_raw["device_ids"], "mapping.prefill.device_ids"),
        state,
        placement,
    )

    attention_raw = _mapping(root["decode_attn"], "mapping.decode_attn")
    _fields(
        attention_raw,
        "mapping.decode_attn",
        {"device_ids", "placement", "residency", "capacity_reservation"},
    )
    capacity_path = "mapping.decode_attn.capacity_reservation"
    capacity_raw = _mapping(attention_raw["capacity_reservation"], capacity_path)
    _fields(capacity_raw, capacity_path, {"model"})
    capacity_model = _literal(
        capacity_raw["model"],
        f"{capacity_path}.model",
        ("oracle_output",),
    )
    attention = DecodeAttentionMappingConfig(
        device_ids=_device_ids(
            attention_raw["device_ids"],
            "mapping.decode_attn.device_ids",
        ),
        placement=_enum(
            attention_raw["placement"],
            "mapping.decode_attn.placement",
            AttentionPlacement,
        ),
        residency=_enum(
            attention_raw["residency"],
            "mapping.decode_attn.residency",
            AttentionResidency,
        ),
        capacity_reservation=CapacityReservationConfig(capacity_model),
    )

    ffn_raw = _mapping(root["decode_ffn"], "mapping.decode_ffn")
    _fields(
        ffn_raw,
        "mapping.decode_ffn",
        {"device_ids", "experts_by_device"},
    )
    ffn_device_ids = _device_ids(
        ffn_raw["device_ids"],
        "mapping.decode_ffn.device_ids",
    )
    experts_path = "mapping.decode_ffn.experts_by_device"
    experts_raw = _mapping(ffn_raw["experts_by_device"], experts_path)
    placements: list[ExpertPlacementConfig] = []
    for device_id, expert_values in experts_raw.items():
        if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
            raise ConfigurationError(
                f"{experts_path}: device keys must be non-negative integers"
            )
        expert_path = f"{experts_path}[{device_id}]"
        expert_list = _sequence(expert_values, expert_path)
        if not expert_list:
            raise ConfigurationError(f"{expert_path}: must not be empty")
        expert_ids = tuple(
            _integer(item, f"{expert_path}[{index}]", minimum=0)
            for index, item in enumerate(expert_list)
        )
        placements.append(ExpertPlacementConfig(device_id, expert_ids))

    return MappingConfig(
        schema_version=_schema_version(root, "mapping"),
        granularity=granularity,
        model=model,
        prefill=prefill,
        decode_attn=attention,
        decode_ffn=DecodeFFNMappingConfig(ffn_device_ids, tuple(placements)),
        source_path=source_path,
    )


def load_config(
    hardware_path: str | Path,
    workload_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> SimulationConfig:
    """Load, combine, and cross-validate the three required V1 files."""

    if workload_path is None or mapping_path is None:
        raise ConfigurationError(
            "V1 requires three configuration files: "
            "hardware.yaml, workload.yaml, and mapping.yaml"
        )
    config = SimulationConfig(
        hardware=load_hardware_config(hardware_path),
        workload=load_workload_config(workload_path),
        mapping=load_mapping_config(mapping_path),
    )
    validate_simulation_config(config)
    return config
