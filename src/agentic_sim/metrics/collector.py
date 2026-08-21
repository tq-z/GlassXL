"""Metrics collection from typed reconstructed controller results."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from agentic_sim.hardware.topology import Topology
from agentic_sim.runtime.attention_admission import AttentionAdmissionController
from agentic_sim.runtime.decode_controller import DecodeStepResult, PendingPoolWrite
from agentic_sim.runtime.prefill_controller import PrefillBatchResult, PrefillTransferResult
from agentic_sim.runtime.session import AttentionKVResidency, DecodeEpisodeState, SessionState

from .records import (
    AttentionAdmissionMetric,
    AttentionDeviceMetric,
    DecodeStepMetric,
    FFNDeviceMetric,
    MetricsSnapshot,
    PrefillBatchMetric,
    PrefillSessionMetric,
    ResourceMetric,
    SessionMetric,
    TurnMetric,
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass
class _PrefillData:
    session_id: int
    turn_index: int
    batch_id: int
    device_id: int
    state: str
    context_tokens: int
    prefill_tokens: int
    load_tokens: int
    queue_latency_s: float
    service_latency_s: float
    kv_pool_load_s: float
    stateful_affinity_hit: bool
    transfer_tokens: int = 0
    kv_p2p_s: float = 0.0


@dataclass
class _PrefillBatchData:
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
    queue_latency_s: float
    service_latency_s: float
    transfer_tokens: int = 0


class MetricsCollector:
    """Accumulate Section 18 records without legacy byte or task semantics."""

    def __init__(self) -> None:
        self._prefill: dict[tuple[int, int], _PrefillData] = {}
        self._prefill_batches: dict[int, _PrefillBatchData] = {}
        self._stateful_seen: set[int] = set()
        self.attention_admissions: list[AttentionAdmissionMetric] = []
        self.attention_devices: list[AttentionDeviceMetric] = []
        self.decode_steps: list[DecodeStepMetric] = []
        self._pool_completion: dict[tuple[int, int], tuple[float, float]] = {}
        self._pool_loads: list[tuple[float, int]] = []
        self._pool_offloads: list[tuple[float, int]] = []
        self._p2p_transfers: list[tuple[float, int]] = []

    def record_prefill(self, result: PrefillBatchResult, state: str) -> None:
        requests = {item.session.session_id: item for item in result.batch.requests}
        first_ready = min(item.ready_time_s for item in result.batch.requests)
        queue_latency = result.pool_load.start_time - first_ready
        batch_data = _PrefillBatchData(
            batch_id=result.batch.batch_id,
            device_id=result.batch.device_id,
            state=state,
            ready_time_s=first_ready,
            start_time_s=result.pool_load.start_time,
            finish_time_s=result.compute.finish_time,
            request_count=len(result.sessions),
            context_tokens=sum(item.context_tokens for item in result.sessions),
            prefill_tokens=sum(item.prefill_tokens for item in result.sessions),
            load_tokens=sum(item.load_tokens for item in result.sessions),
            queue_latency_s=queue_latency,
            service_latency_s=result.compute.service_time,
        )
        self._prefill_batches[batch_data.batch_id] = batch_data
        self._pool_loads.append((result.pool_load.finish_time, batch_data.load_tokens))
        for item in result.sessions:
            request = requests[item.session_id]
            episode = request.session.current_episode
            if episode is None:
                raise RuntimeError("prefill metrics require an active episode")
            affinity_hit = state == "stateful" and item.session_id in self._stateful_seen
            self._prefill[(item.session_id, episode.turn.turn_index)] = _PrefillData(
                session_id=item.session_id,
                turn_index=episode.turn.turn_index,
                batch_id=result.batch.batch_id,
                device_id=result.batch.device_id,
                state=state,
                context_tokens=item.context_tokens,
                prefill_tokens=item.prefill_tokens,
                load_tokens=item.load_tokens,
                queue_latency_s=result.pool_load.start_time - request.ready_time_s,
                service_latency_s=result.compute.service_time,
                kv_pool_load_s=result.pool_load.service_time,
                stateful_affinity_hit=affinity_hit,
            )
            if state == "stateful":
                self._stateful_seen.add(item.session_id)
        self.record_transfers(result.transfers)

    def record_transfers(self, transfers: tuple[PrefillTransferResult, ...]) -> None:
        for transfer in transfers:
            self._p2p_transfers.append((transfer.reservation.finish_time, transfer.total_tokens))
            for admission in transfer.admissions:
                key = self._active_key(admission.session_id)
                data = self._prefill[key]
                data.transfer_tokens += admission.transfer_tokens
                data.kv_p2p_s += transfer.reservation.service_time
                self._prefill_batches[data.batch_id].transfer_tokens += admission.transfer_tokens
                self.attention_admissions.append(
                    AttentionAdmissionMetric(
                        session_id=admission.session_id,
                        turn_index=key[1],
                        time_s=admission.admission_time_s,
                        device_id=admission.device_id,
                        required_final_kv_tokens=admission.required_final_kv_tokens,
                        transfer_tokens=admission.transfer_tokens,
                        capacity_wait_s=admission.capacity_wait_s,
                        cache_hit=admission.cache_hit,
                        retry=admission.capacity_wait_s > 0.0,
                        evicted_session_ids=admission.evicted_session_ids,
                    )
                )

    def record_decode(self, result: DecodeStepResult) -> None:
        device_metrics = tuple(
            FFNDeviceMetric(
                device_id=batch.device_id,
                expert_ids=tuple(work.expert_id for work in batch.experts),
                activation_tokens=batch.activation_tokens,
                service_latency_s=reservation.service_time,
                barrier_wait_s=result.ffn_barrier_s - reservation.finish_time,
            )
            for batch, reservation in zip(result.expert_batches, result.ffn, strict=True)
        )
        loads = tuple(item.activation_tokens for item in device_metrics)
        attention_time_s = result.attention_barrier_s - result.batch.ready_time_s
        ffn_time_s = result.ffn_barrier_s - result.dispatch.finish_time
        ffn_all_to_all_time_s = (
            result.dispatch.finish_time - result.attention_barrier_s
            + result.combine.finish_time - result.ffn_barrier_s
        )
        self.decode_steps.append(
            DecodeStepMetric(
                decode_step=result.decode_step,
                ready_time_s=result.batch.ready_time_s,
                session_ids=tuple(
                    session.session_id for session in result.batch.sessions
                ),
                finish_time_s=result.token_completion_time_s,
                active_sessions=result.batch.size,
                attention_time_s=attention_time_s,
                ffn_time_s=ffn_time_s,
                ffn_all_to_all_time_s=ffn_all_to_all_time_s,
                attention_barrier_s=result.attention_barrier_s,
                attention_device_wait_s=tuple(
                    (reservation.device_id, result.attention_barrier_s - reservation.finish_time)
                    for reservation in result.attention
                ),
                routed_activation_tokens=result.activation_tokens,
                send_tokens=result.send_tokens,
                dispatch_latency_s=result.dispatch.service_time,
                combine_latency_s=result.combine.service_time,
                expert_loads=tuple(
                    (work.expert_id, work.activation_tokens, work.device_id)
                    for work in result.expert_loads
                ),
                ffn_devices=device_metrics,
                ffn_load_imbalance_tokens=(max(loads) - min(loads)) if loads else 0,
                emitted_tokens=result.batch.size,
            )
        )
        for pending in result.pool_writes:
            self.record_pool_write(pending)

    def record_pool_write(self, pending: PendingPoolWrite) -> None:
        episode = pending.session.current_episode
        if episode is None:
            raise RuntimeError("pool-write metrics require an active episode")
        key = (pending.session_id, episode.turn.turn_index)
        self._pool_completion[key] = (
            pending.reservation.finish_time,
            pending.reservation.service_time,
        )
        self._pool_offloads.append((pending.reservation.finish_time, pending.payload_tokens))

    def capture_attention(
        self,
        time_s: float,
        attention: AttentionAdmissionController,
    ) -> None:
        for device_id in attention.device_ids:
            device = attention.devices[device_id]
            residents = tuple(device.resident_sessions.values())
            dirty = sum(
                item.kv.attention.state is AttentionKVResidency.DRAINING_DIRTY
                for item in residents
            )
            clean = sum(
                item.kv.attention.state is AttentionKVResidency.INACTIVE_CLEAN
                for item in residents
            )
            reserved = device.reserved_or_used_tokens
            utilization = reserved / device.capacity_tokens if device.capacity_tokens else 0.0
            self.attention_devices.append(
                AttentionDeviceMetric(
                    time_s=time_s,
                    device_id=device_id,
                    active_sessions=sum(
                        item.kv.attention.state is AttentionKVResidency.ACTIVE
                        for item in residents
                    ),
                    resident_sessions=len(residents),
                    active_context_tokens=device.active_context_load,
                    active_commitment_tokens=device.active_commitment_tokens,
                    resident_used_tokens=device.resident_used_tokens,
                    capacity_tokens=device.capacity_tokens,
                    capacity_utilization=utilization,
                    dirty_sessions=dirty,
                    clean_inactive_sessions=clean,
                )
            )

    def snapshot(
        self,
        sessions: tuple[SessionState, ...],
        topology: Topology,
        horizon_s: float,
    ) -> MetricsSnapshot:
        turns = tuple(
            metric
            for session in sorted(sessions, key=lambda item: item.session_id)
            for metric in self._turn_metrics(session, horizon_s)
        )
        session_metrics = tuple(self._session_metric(session, horizon_s) for session in sessions)
        completed_sessions = [item for item in session_metrics if item.finish_time_s is not None]
        completed_turns = [item for item in turns if item.finish_time_s is not None]
        visible_steps = [item for item in self.decode_steps if item.finish_time_s <= horizon_s]
        ttfts = [item.ttft_s for item in completed_turns if item.ttft_s is not None]
        inter_token = [value for item in completed_turns for value in item.inter_token_latencies_s]
        latencies = [item.completion_latency_s for item in completed_sessions if item.completion_latency_s is not None]
        resources = self._resource_metrics(topology, horizon_s)
        admissions = tuple(
            item for item in self.attention_admissions if item.time_s <= horizon_s
        )
        prefill_sessions = tuple(
            PrefillSessionMetric(
                session_id=item.session_id,
                turn_index=item.turn_index,
                batch_id=item.batch_id,
                device_id=item.device_id,
                state=item.state,
                load_kind=("full_context" if item.load_tokens == item.context_tokens else "delta"),
                context_tokens=item.context_tokens,
                prefill_tokens=item.prefill_tokens,
                load_tokens=item.load_tokens,
                compute_tokens=item.prefill_tokens,
                transfer_tokens=item.transfer_tokens,
                queue_latency_s=item.queue_latency_s,
                service_latency_s=item.service_latency_s,
                stateful_affinity_hit=item.stateful_affinity_hit,
            )
            for item in sorted(self._prefill.values(), key=lambda value: (value.session_id, value.turn_index))
        )
        prefill_batches = tuple(
            PrefillBatchMetric(
                batch_id=item.batch_id,
                device_id=item.device_id,
                state=item.state,
                ready_time_s=item.ready_time_s,
                start_time_s=item.start_time_s,
                finish_time_s=item.finish_time_s,
                request_count=item.request_count,
                context_tokens=item.context_tokens,
                prefill_tokens=item.prefill_tokens,
                load_tokens=item.load_tokens,
                transfer_tokens=item.transfer_tokens,
                queue_latency_s=item.queue_latency_s,
                service_latency_s=item.service_latency_s,
            )
            for item in sorted(self._prefill_batches.values(), key=lambda value: value.batch_id)
            if item.start_time_s <= horizon_s
        )
        emitted_tokens = sum(item.emitted_tokens for item in visible_steps)
        summary: dict[str, int | float] = {
            "simulation_time_s": horizon_s,
            "offered_sessions": sum(item.spec.arrival_time_s <= horizon_s for item in sessions),
            "completed_sessions": len(completed_sessions),
            "completed_turns": len(completed_turns),
            "decode_steps": len(visible_steps),
            "emitted_tokens": emitted_tokens,
            "decode_token_throughput_per_s": emitted_tokens / horizon_s if horizon_s else 0.0,
            "mean_session_latency_ms": (fmean(latencies) * 1_000.0 if latencies else 0.0),
            "p95_session_latency_ms": _percentile(latencies, 0.95) * 1_000.0,
            "mean_ttft_ms": (fmean(ttfts) * 1_000.0 if ttfts else 0.0),
            "p95_ttft_ms": _percentile(ttfts, 0.95) * 1_000.0,
            "mean_inter_token_ms": (fmean(inter_token) * 1_000.0 if inter_token else 0.0),
            "prefill_compute_tokens": sum(item.compute_tokens for item in prefill_sessions),
            "kv_pool_load_tokens": sum(tokens for finish, tokens in self._pool_loads if finish <= horizon_s),
            "kv_pool_offload_tokens": sum(tokens for finish, tokens in self._pool_offloads if finish <= horizon_s),
            "kv_p2p_tokens": sum(tokens for finish, tokens in self._p2p_transfers if finish <= horizon_s),
            "routed_activation_tokens": sum(item.routed_activation_tokens for item in visible_steps),
            "all_to_all_send_tokens": sum(2 * item.send_tokens for item in visible_steps),
            "attention_admissions": len(admissions),
            "attention_retries": sum(item.retry for item in admissions),
            "attention_cache_hits": sum(item.cache_hit for item in admissions),
            "attention_cache_misses": sum(not item.cache_hit for item in admissions),
            "lru_evictions": sum(len(item.evicted_session_ids) for item in admissions),
        }
        return MetricsSnapshot(
            summary=summary,
            sessions=session_metrics,
            turns=turns,
            attention_admissions=tuple(
                sorted(
                    admissions,
                    key=lambda item: (item.time_s, item.session_id, item.turn_index),
                )
            ),
            attention_devices=tuple(
                sorted(
                    (
                        item
                        for item in self.attention_devices
                        if item.time_s <= horizon_s
                    ),
                    key=lambda item: (item.time_s, item.device_id),
                )
            ),
            prefill_batches=prefill_batches,
            prefill_sessions=prefill_sessions,
            decode_steps=tuple(visible_steps),
            resources=resources,
        )

    def _active_key(self, session_id: int) -> tuple[int, int]:
        matches = [key for key in self._prefill if key[0] == session_id]
        if not matches:
            raise RuntimeError(f"session {session_id} has no recorded prefill")
        return max(matches, key=lambda key: key[1])

    def _turn_metrics(self, session: SessionState, horizon_s: float) -> tuple[TurnMetric, ...]:
        episodes: list[DecodeEpisodeState] = list(session.completed_episodes)
        if session.current_episode is not None:
            episodes.append(session.current_episode)
        admissions = {
            (item.session_id, item.turn_index): item for item in self.attention_admissions
        }
        result: list[TurnMetric] = []
        for episode in episodes:
            key = (session.session_id, episode.turn.turn_index)
            prefill = self._prefill.get(key)
            admission = admissions.get(key)
            pool = self._pool_completion.get(key)
            finish = pool[0] if pool is not None and pool[0] <= horizon_s else None
            token_times = tuple(time for time in episode.token_completion_times_s if time <= horizon_s)
            ttft = token_times[0] - episode.turn_ready_time_s if token_times else None
            inter_token = tuple(later - earlier for earlier, later in zip(token_times, token_times[1:]))
            decode_latency = (
                token_times[-1] - episode.attention_ready_time_s
                if token_times and episode.attention_ready_time_s is not None
                else (0.0 if episode.turn.output_tokens == 0 and finish is not None else None)
            )
            result.append(
                TurnMetric(
                    session_id=session.session_id,
                    turn_index=episode.turn.turn_index,
                    arrival_time_s=session.spec.arrival_time_s,
                    ready_time_s=episode.turn_ready_time_s,
                    finish_time_s=finish,
                    attention_capacity_wait_s=episode.attention_capacity_wait_s,
                    prefill_queue_s=prefill.queue_latency_s if prefill else 0.0,
                    prefill_service_s=prefill.service_latency_s if prefill else 0.0,
                    kv_pool_load_s=prefill.kv_pool_load_s if prefill else 0.0,
                    kv_p2p_s=prefill.kv_p2p_s if prefill else 0.0,
                    decode_admission_s=(
                        admission.time_s - episode.turn_ready_time_s if admission else None
                    ),
                    ttft_s=ttft,
                    inter_token_latencies_s=inter_token,
                    decode_episode_latency_s=decode_latency,
                    tool_wait_s=(
                        episode.turn.tool_wait_s
                        if episode.turn.turn_index < len(session.spec.turns) - 1
                        else 0.0
                    ),
                    completion_latency_s=(finish - episode.turn_ready_time_s if finish is not None else None),
                    context_tokens=episode.context_before_prefill,
                    prefill_tokens=episode.turn.prefill_tokens,
                    output_tokens=episode.turn.output_tokens,
                    generated_tokens=len(token_times),
                    pool_write_tokens=episode.pool_write_tokens if finish is not None else None,
                    kv_pool_offload_s=pool[1] if finish is not None and pool is not None else 0.0,
                    attention_device_id=admission.device_id if admission else None,
                    attention_cache_hit=admission.cache_hit if admission else None,
                )
            )
        return tuple(result)

    def _session_metric(self, session: SessionState, horizon_s: float) -> SessionMetric:
        last_key = (session.session_id, len(session.spec.turns) - 1)
        pool = self._pool_completion.get(last_key)
        finish = pool[0] if session.is_finished and pool is not None and pool[0] <= horizon_s else None
        return SessionMetric(
            session_id=session.session_id,
            arrival_time_s=session.spec.arrival_time_s,
            finish_time_s=finish,
            completion_latency_s=(finish - session.spec.arrival_time_s if finish is not None else None),
            completed_turns=len(session.completed_episodes),
            total_turns=len(session.spec.turns),
            initial_context_tokens=session.spec.initial_context_tokens,
            final_context_tokens=session.current_context_tokens,
            generated_tokens=sum(item.generated_tokens for item in session.completed_episodes),
            tool_wait_s=sum(item.tool_wait_s for item in session.spec.turns[:-1]),
        )

    def _resource_metrics(self, topology: Topology, horizon_s: float) -> tuple[ResourceMetric, ...]:
        result: list[ResourceMetric] = []
        for device_id in sorted(topology.devices):
            device = topology.devices[device_id]
            operations: dict[str, float] = {}
            for item in device.reservations:
                busy = max(0.0, min(item.finish_time, horizon_s) - item.start_time)
                operations[item.operation.value] = operations.get(item.operation.value, 0.0) + busy
            result.append(
                ResourceMetric(
                    resource_id=device.queue.resource_id,
                    resource_type="physical_device",
                    busy_time_s=device.queue.busy_time(horizon_s),
                    queue_time_s=sum(
                        item.queue_delay for item in device.reservations if item.start_time <= horizon_s
                    ),
                    utilization=device.queue.utilization(horizon_s),
                    max_queue_length=device.queue.max_queue_length,
                    operation_busy_time_s=tuple(sorted(operations.items())),
                    transferred_tokens=0,
                )
            )
        logical = (
            ("ffn_all_to_all", topology.ffn_all_to_all.queue, sum(2 * item.send_tokens for item in self.decode_steps if item.finish_time_s <= horizon_s)),
            ("kv_p2p", topology.kv_p2p.queue, sum(tokens for finish, tokens in self._p2p_transfers if finish <= horizon_s)),
            ("kv_pool", topology.kv_pool.queue, sum(tokens for finish, tokens in self._pool_loads + self._pool_offloads if finish <= horizon_s)),
        )
        for resource_type, queue, tokens in logical:
            result.append(
                ResourceMetric(
                    resource_id=queue.resource_id,
                    resource_type=resource_type,
                    busy_time_s=queue.busy_time(horizon_s),
                    queue_time_s=sum(
                        item.queue_delay for item in queue.reservations if item.start_time <= horizon_s
                    ),
                    utilization=queue.utilization(horizon_s),
                    max_queue_length=queue.max_queue_length,
                    operation_busy_time_s=((resource_type, queue.busy_time(horizon_s)),),
                    transferred_tokens=tokens,
                )
            )
        return tuple(result)
