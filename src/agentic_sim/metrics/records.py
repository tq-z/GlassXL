"""Immutable token-unit metrics for the reconstructed PD/AFD simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass


MetricScalar = int | float | str | bool | None


@dataclass(frozen=True)
class SessionMetric:
    session_id: int
    arrival_time_s: float
    finish_time_s: float | None
    completion_latency_s: float | None
    completed_turns: int
    total_turns: int
    initial_context_tokens: int
    final_context_tokens: int
    generated_tokens: int
    tool_wait_s: float

    def to_dict(self) -> dict[str, MetricScalar]:
        return asdict(self)


@dataclass(frozen=True)
class TurnMetric:
    session_id: int
    turn_index: int
    arrival_time_s: float
    ready_time_s: float
    finish_time_s: float | None
    attention_capacity_wait_s: float
    prefill_queue_s: float
    prefill_service_s: float
    kv_pool_load_s: float
    kv_p2p_s: float
    decode_admission_s: float | None
    ttft_s: float | None
    inter_token_latencies_s: tuple[float, ...]
    decode_episode_latency_s: float | None
    tool_wait_s: float
    completion_latency_s: float | None
    context_tokens: int
    prefill_tokens: int
    output_tokens: int
    generated_tokens: int
    pool_write_tokens: int | None
    kv_pool_offload_s: float
    attention_device_id: int | None
    attention_cache_hit: bool | None

    def to_dict(self) -> dict[str, MetricScalar | tuple[float, ...]]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionAdmissionMetric:
    session_id: int
    turn_index: int
    time_s: float
    device_id: int
    required_final_kv_tokens: int
    transfer_tokens: int
    capacity_wait_s: float
    cache_hit: bool
    retry: bool
    evicted_session_ids: tuple[int, ...]


@dataclass(frozen=True)
class AttentionDeviceMetric:
    time_s: float
    device_id: int
    active_sessions: int
    resident_sessions: int
    active_context_tokens: int
    active_commitment_tokens: int
    resident_used_tokens: int
    capacity_tokens: int
    capacity_utilization: float
    dirty_sessions: int
    clean_inactive_sessions: int


@dataclass(frozen=True)
class PrefillBatchMetric:
    batch_id: int
    device_id: int
    state: str
    ready_time_s: float
    start_time_s: float
    finish_time_s: float
    request_count: int
    context_tokens: int
    prefill_tokens: int
    load_tokens: int
    transfer_tokens: int
    queue_latency_s: float
    service_latency_s: float


@dataclass(frozen=True)
class PrefillSessionMetric:
    session_id: int
    turn_index: int
    batch_id: int
    device_id: int
    state: str
    load_kind: str
    context_tokens: int
    prefill_tokens: int
    load_tokens: int
    compute_tokens: int
    transfer_tokens: int
    queue_latency_s: float
    service_latency_s: float
    stateful_affinity_hit: bool


@dataclass(frozen=True)
class FFNDeviceMetric:
    device_id: int
    expert_ids: tuple[int, ...]
    activation_tokens: int
    service_latency_s: float
    barrier_wait_s: float


@dataclass(frozen=True)
class DecodeStepMetric:
    decode_step: int
    ready_time_s: float
    session_ids: tuple[int, ...]
    finish_time_s: float
    active_sessions: int
    attention_time_s: float
    ffn_time_s: float
    ffn_all_to_all_time_s: float
    attention_barrier_s: float
    attention_device_wait_s: tuple[tuple[int, float], ...]
    routed_activation_tokens: int
    send_tokens: int
    dispatch_latency_s: float
    combine_latency_s: float
    expert_loads: tuple[tuple[int, int, int], ...]
    ffn_devices: tuple[FFNDeviceMetric, ...]
    ffn_load_imbalance_tokens: int
    emitted_tokens: int


@dataclass(frozen=True)
class ResourceMetric:
    resource_id: str
    resource_type: str
    busy_time_s: float
    queue_time_s: float
    utilization: float
    max_queue_length: int
    operation_busy_time_s: tuple[tuple[str, float], ...]
    transferred_tokens: int


@dataclass(frozen=True)
class MetricsSnapshot:
    summary: dict[str, int | float]
    sessions: tuple[SessionMetric, ...]
    turns: tuple[TurnMetric, ...]
    attention_admissions: tuple[AttentionAdmissionMetric, ...]
    attention_devices: tuple[AttentionDeviceMetric, ...]
    prefill_batches: tuple[PrefillBatchMetric, ...]
    prefill_sessions: tuple[PrefillSessionMetric, ...]
    decode_steps: tuple[DecodeStepMetric, ...]
    resources: tuple[ResourceMetric, ...]
