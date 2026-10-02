"""Immutable configuration models for the breaking PD/AFD schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias


class ConfigurationError(ValueError):
    """Raised when a configuration file does not satisfy the V1 contract."""


class PrefillModel(StrEnum):
    AGGREGATE_THROUGHPUT = "aggregate_throughput"
    ROOFLINE_CONTEXT = "roofline_context"


class AttentionModel(StrEnum):
    LINEAR_CONTEXT = "linear_context"


class ExpertModel(StrEnum):
    LINEAR_EXPERT_BATCH = "linear_expert_batch"
    BATCH_LATENCY_CURVE = "batch_latency_curve"


class AllToAllModel(StrEnum):
    STARTUP_MIN_PACKAGE = "startup_min_package"


class P2PModel(StrEnum):
    STARTUP_THROUGHPUT = "startup_throughput"


class DistributionType(StrEnum):
    CONSTANT = "constant"
    BOUNDED_INTEGER = "bounded_integer"
    DISCRETE = "discrete"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    LOGNORMAL = "lognormal"


class PrefillState(StrEnum):
    STATEFUL = "stateful"
    STATELESS = "stateless"


class PrefillPlacement(StrEnum):
    STICKY_ROUND_ROBIN = "sticky_round_robin"
    ROUND_ROBIN = "round_robin"


class AttentionPlacement(StrEnum):
    ROUND_ROBIN = "round_robin"
    LEAST_ACTIVE_CONTEXT = "least_active_context"


class AttentionResidency(StrEnum):
    ACTIVE_ONLY = "active_only"
    LRU_CACHE = "lru_cache"


@dataclass(frozen=True)
class RooflinePrefillConfig:
    flops_per_second: float
    memory_bandwidth_bytes_per_second: float
    flops_per_token: float
    flops_per_attention_pair: float
    fixed_weight_bytes: float
    routed_weight_bytes: float
    routed_expert_fraction: float
    context_bytes_per_token: float
    bytes_per_prefill_token: float


@dataclass(frozen=True)
class PrefillCapabilityConfig:
    model: PrefillModel
    tokens_per_second: float | None = None
    roofline: RooflinePrefillConfig | None = None


@dataclass(frozen=True)
class DecodeAttentionCapabilityConfig:
    model: AttentionModel
    context_tokens_per_second: float
    kv_capacity_tokens: int


@dataclass(frozen=True)
class BatchLatencyPointConfig:
    activation_tokens: int
    latency_us: float


@dataclass(frozen=True)
class DecodeFFNCapabilityConfig:
    model: ExpertModel
    tokens_per_second: float | None = None
    points: tuple[BatchLatencyPointConfig, ...] = ()


@dataclass(frozen=True)
class DeviceCapabilitiesConfig:
    prefill: PrefillCapabilityConfig | None = None
    decode_attn: DecodeAttentionCapabilityConfig | None = None
    decode_ffn: DecodeFFNCapabilityConfig | None = None


@dataclass(frozen=True)
class DeviceConfig:
    id: int
    capabilities: DeviceCapabilitiesConfig


@dataclass(frozen=True)
class KVPoolConfig:
    capacity_tokens: None
    load_tokens_per_second: float
    offload_tokens_per_second: float
    start_latency_us: float

    @property
    def tokens_per_second(self) -> float:
        """Return the V1 symmetric pool transfer rate."""

        return self.load_tokens_per_second


@dataclass(frozen=True)
class FFNAllToAllConfig:
    model: AllToAllModel
    start_latency_us: float
    min_package_tokens: int
    throughput_tokens_per_second: float


@dataclass(frozen=True)
class KVP2PConfig:
    model: P2PModel
    start_latency_us: float
    throughput_tokens_per_second: float


@dataclass(frozen=True)
class InterconnectsConfig:
    ffn_all_to_all: FFNAllToAllConfig
    kv_p2p: KVP2PConfig


@dataclass(frozen=True)
class HardwareConfig:
    schema_version: int
    devices: tuple[DeviceConfig, ...]
    kv_pool: KVPoolConfig
    interconnects: InterconnectsConfig
    source_path: Path


@dataclass(frozen=True)
class ConstantDistributionConfig:
    type: DistributionType
    value: int | float


@dataclass(frozen=True)
class BoundedIntegerDistributionConfig:
    type: DistributionType
    minimum: int
    maximum: int


@dataclass(frozen=True)
class DiscreteDistributionConfig:
    type: DistributionType
    values: tuple[int | float, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class UniformDistributionConfig:
    type: DistributionType
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ExponentialDistributionConfig:
    type: DistributionType
    rate: float


@dataclass(frozen=True)
class LognormalDistributionConfig:
    type: DistributionType
    mean: float
    sigma: float
    minimum: float | None
    maximum: float | None


DistributionConfig: TypeAlias = (
    ConstantDistributionConfig
    | BoundedIntegerDistributionConfig
    | DiscreteDistributionConfig
    | UniformDistributionConfig
    | ExponentialDistributionConfig
    | LognormalDistributionConfig
)


@dataclass(frozen=True)
class PoissonArrivalConfig:
    type: str
    rate_per_second: float


@dataclass(frozen=True)
class SyntheticSourceConfig:
    type: str
    num_sessions: int
    arrival: PoissonArrivalConfig
    turns: DistributionConfig
    context_tokens: DistributionConfig
    prefill_tokens: DistributionConfig
    output_tokens: DistributionConfig
    tool_wait_ms: DistributionConfig


@dataclass(frozen=True)
class WorkloadConfig:
    schema_version: int
    seed: int
    simulation_end_s: float | None
    source: SyntheticSourceConfig
    source_path: Path


@dataclass(frozen=True)
class ModelShapeConfig:
    num_layers: int
    num_experts: int
    top_k: int


@dataclass(frozen=True)
class PrefillMappingConfig:
    device_ids: tuple[int, ...]
    state: PrefillState
    placement: PrefillPlacement


@dataclass(frozen=True)
class CapacityReservationConfig:
    model: str


@dataclass(frozen=True)
class DecodeAttentionMappingConfig:
    device_ids: tuple[int, ...]
    placement: AttentionPlacement
    residency: AttentionResidency
    capacity_reservation: CapacityReservationConfig


@dataclass(frozen=True)
class ExpertPlacementConfig:
    device_id: int
    expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class DecodeFFNMappingConfig:
    device_ids: tuple[int, ...]
    experts_by_device: tuple[ExpertPlacementConfig, ...]


@dataclass(frozen=True)
class MappingConfig:
    schema_version: int
    granularity: str
    model: ModelShapeConfig
    prefill: PrefillMappingConfig
    decode_attn: DecodeAttentionMappingConfig
    decode_ffn: DecodeFFNMappingConfig
    source_path: Path


@dataclass(frozen=True)
class SimulationConfig:
    hardware: HardwareConfig
    workload: WorkloadConfig
    mapping: MappingConfig
