"""Structured Phase 1.5 batching and bandwidth metric records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillBatchMetric:
    batch_id: int
    start_time_s: float
    finish_time_s: float
    request_count: int
    total_tokens: int
    longest_tokens: int
    queue_length_at_start: int


@dataclass(frozen=True)
class IngressFlowMetric:
    flow_id: int
    session_id: int
    submitted_time_s: float
    data_start_time_s: float
    finish_time_s: float
    byte_count: int
    achieved_bandwidth_bytes_per_s: float


@dataclass(frozen=True)
class DecodeIterationMetric:
    iteration_id: int
    start_time_s: float
    finish_time_s: float
    active_sessions: int
    waiting_sessions: int
    emitted_tokens: int

