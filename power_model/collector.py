"""MetricsCollector subclass that bins per-device activity for energy pricing.

Swap it into ``AgenticRuntime.metrics`` after construction (then re-call
``capture_attention(0.0, ...)``). Every override calls ``super()`` first, so
the simulator's own metrics are unchanged. The base ``record_prefill`` calls
``record_transfers`` and the base ``record_decode`` calls
``record_pool_write``; P2P and pool offloads are therefore priced only in those
two hooks.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from agentic_sim.config.models import SimulationConfig
from agentic_sim.metrics import MetricsCollector
from roofline_model import DeploymentSpec, HardwareSpec, ModelSpec, PrecisionSpec, kv_token_size
from roofline_model.activity import (
    Activity,
    DecodeAttentionCoefficients,
    PrefillContextCoefficients,
    ffn_layer_activity,
    ffn_layer_time,
    kv_append_bytes_per_token,
    moe_layer_count,
)

from .binner import ACTIVITY_KINDS, ActivityBinner
from .layout import DeviceLayout

_IO_TX = ACTIVITY_KINDS.index("io_tx")
_IO_RX = ACTIVITY_KINDS.index("io_rx")
_MEM_RD = ACTIVITY_KINDS.index("mem_rd")
_MEM_WR = ACTIVITY_KINDS.index("mem_wr")


def activity_vector(activity: Activity, io_tx: float = 0.0, io_rx: float = 0.0) -> np.ndarray:
    return np.array(
        [
            activity.flops_gemm,
            activity.flops_attn,
            activity.weight_rd + activity.kv_rd,
            activity.kv_wr,
            activity.act,
            io_tx,
            io_rx,
        ]
    )


def _hardware(document: dict[str, Any] | None) -> HardwareSpec | None:
    if not document:
        return None
    return HardwareSpec(
        float(document["flops_per_second"]),
        float(document["memory_bandwidth_bytes_per_second"]),
    )


class PowerCollector(MetricsCollector):
    def __init__(
        self,
        sim_config: SimulationConfig,
        layout: DeviceLayout,
        binner: ActivityBinner,
        model: ModelSpec,
        precision: PrecisionSpec,
        *,
        report: dict[str, Any] | None = None,
        routing: Any | None = None,
        track_reservations: bool = False,
        timing_samples: int = 200_000,
    ) -> None:
        super().__init__()
        shape = sim_config.mapping.model
        if (shape.num_layers, shape.num_experts, shape.top_k) != (
            model.num_layers,
            model.num_experts,
            model.top_k,
        ):
            raise ValueError(
                f"mapping model shape {shape} does not match power.yaml model {model.display_name}"
            )
        self.layout = layout
        self.binner = binner
        self.model = model
        self.precision = precision
        self.routing = routing
        mapping = sim_config.mapping
        self.deployment = DeploymentSpec(
            dp=len(mapping.decode_attn.device_ids),
            ep=len(mapping.decode_ffn.device_ids),
        )
        self.attention = DecodeAttentionCoefficients.from_model(model, precision)
        self.append_bytes = kv_append_bytes_per_token(model, precision)
        self.prefill = PrefillContextCoefficients.from_model(model, precision)
        self.moe_layers = moe_layer_count(model)
        self.copy_bytes = model.routed_activation_width * precision.activation_bytes
        inputs = (report or {}).get("inputs") or {}
        reference_context = int(
            (inputs.get("operating_point") or {}).get("decode_context_tokens", 32768)
        )
        self.kv_token_bytes = kv_token_size(model, precision, reference_context)[2]
        self.tp_bytes_per_token = 2.0 * model.num_layers * model.hidden_size * precision.activation_bytes
        hardware = inputs.get("hardware") or {}
        deployment = (report or {}).get("deployment") or {}
        self.attention_hardware = _hardware(hardware.get("decode_attention"))
        self.ffn_hardware = _hardware(hardware.get("decode_ffn"))
        self.prefill_hardware = _hardware(deployment.get("prefill_device_hardware"))
        self._ffn_cache: dict[tuple[int, int], tuple[Activity, float]] = {}
        self._timing_limit = timing_samples
        self.timing_ratios: dict[str, list[float]] = {
            "prefill": [],
            "attn": [],
            "ffn": [],
            "ffn_routed": [],
        }
        self.track_reservations = track_reservations
        self.seen_reservations: list[tuple[int, str, float, float]] = []

    # ------------------------------------------------------------------ helpers
    def _track(self, reservations: Sequence[Any]) -> None:
        if self.track_reservations:
            self.seen_reservations.extend(
                (item.device_id, item.operation.value, item.start_time, item.finish_time)
                for item in reservations
            )

    def _timing(self, kind: str, simulated_s: float, activity_s: float) -> None:
        samples = self.timing_ratios[kind]
        if activity_s > 0.0 and len(samples) < self._timing_limit:
            samples.append(simulated_s / activity_s)

    def _ffn_equal(self, tokens: int, experts: int) -> tuple[Activity, float]:
        """Whole-model FFN activity/time when every layer sees the same local load."""

        key = (tokens, experts)
        cached = self._ffn_cache.get(key)
        if cached is not None:
            return cached
        activity = Activity()
        seconds = 0.0
        for group in self.model.layer_groups:
            activity = activity + ffn_layer_activity(
                self.model, group, self.precision, self.deployment, tokens, experts
            ).scaled(group.count)
            if self.ffn_hardware is not None:
                seconds += group.count * ffn_layer_time(
                    self.model, group, self.ffn_hardware, self.precision, self.deployment, tokens, experts
                )
        self._ffn_cache[key] = (activity, seconds)
        return activity, seconds

    # ------------------------------------------------------------------- decode
    def record_decode(self, result) -> None:
        super().record_decode(result)
        batch = result.batch
        # Real hardware interleaves attention, dispatch, FFN and combine layer by
        # layer behind a per-layer barrier, so every device's step work runs from
        # the last device's attention start to the combine finish. (A device that
        # starts early only waits at the first barrier.)
        started = [
            reservation.start_time
            for local, reservation in zip(batch.local_batches, result.attention, strict=True)
            if local.work
        ]
        t0 = max(started) if started else min(reservation.start_time for reservation in result.attention)
        t1 = result.combine.finish_time
        self._track(result.attention)
        self._track(result.dispatch.endpoint_reservations)
        self._track(result.ffn)
        self._track(result.combine.endpoint_reservations)

        attention_ids: list[int] = []
        attention_vectors: list[np.ndarray] = []
        for local, reservation in zip(batch.local_batches, result.attention, strict=True):
            if not local.work:
                continue
            activity = self.attention.activity([work.context_tokens for work in local.work])
            attention_ids.append(local.device_id)
            attention_vectors.append(activity_vector(activity))
            if self.attention_hardware is not None:
                modelled = max(
                    activity.flops / self.attention_hardware.flops_per_second,
                    (activity.bytes_moved - len(local.work) * self.append_bytes)
                    / self.attention_hardware.memory_bandwidth_bytes_per_second,
                )
                self._timing("attn", reservation.service_time, modelled)
        if attention_ids:
            self.binner.add_many(t0, t1, attention_ids, np.vstack(attention_vectors), "attn")

        if batch.size == 0:
            return
        if self.routing is None:
            ffn_ids, ffn_vectors, io = self._equal_ffn_and_all_to_all(result)
        else:
            ffn_ids, ffn_vectors, io = self.routing.step_activity(self, result)
        if ffn_ids:
            self.binner.add_many(t0, t1, ffn_ids, np.vstack(ffn_vectors), "ffn")
        if io:
            ids = sorted(io)
            vectors = np.zeros((len(ids), len(ACTIVITY_KINDS)))
            for row, device_id in enumerate(ids):
                vectors[row, _IO_TX], vectors[row, _IO_RX] = io[device_id]
            self.binner.add_many(t0, t1, ids, vectors, "a2a")

    def _equal_ffn_and_all_to_all(self, result):
        ffn_ids: list[int] = []
        ffn_vectors: list[np.ndarray] = []
        for device_batch, reservation in zip(result.expert_batches, result.ffn, strict=True):
            tokens = device_batch.activation_tokens
            if tokens == 0:
                continue
            experts = sum(1 for work in device_batch.experts if work.activation_tokens > 0)
            activity, seconds = self._ffn_equal(tokens, experts)
            ffn_ids.append(device_batch.device_id)
            ffn_vectors.append(activity_vector(activity))
            self._timing("ffn", reservation.service_time, seconds)
        return ffn_ids, ffn_vectors, self.equal_all_to_all(result)

    def equal_all_to_all(self, result) -> dict[int, tuple[float, float]]:
        """Remote dispatch + combine bytes under proportional (equal) routing.

        copies(a -> d) = L_moe * B_a * x_d / B per step; copies are remote when
        the session's attention device differs from the FFN device.
        """

        batch = result.batch
        sessions = np.array([len(local.sessions) for local in batch.local_batches], dtype=float)
        tokens = np.array([item.activation_tokens for item in result.expert_batches], dtype=float)
        attention_ids = [local.device_id for local in batch.local_batches]
        ffn_ids = [item.device_id for item in result.expert_batches]
        copies = np.outer(sessions, tokens) / batch.size
        copies[np.equal.outer(attention_ids, ffn_ids)] = 0.0
        scale = self.moe_layers * self.copy_bytes
        dispatch_tx = copies.sum(axis=1) * scale
        dispatch_rx = copies.sum(axis=0) * scale
        io: dict[int, list[float]] = {}
        for device_id, sent in zip(attention_ids, dispatch_tx):
            entry = io.setdefault(device_id, [0.0, 0.0])
            entry[0] += sent  # dispatch out
            entry[1] += sent  # combine back in
        for device_id, received in zip(ffn_ids, dispatch_rx):
            entry = io.setdefault(device_id, [0.0, 0.0])
            entry[1] += received  # dispatch in
            entry[0] += received  # combine out
        return {device_id: (tx, rx) for device_id, (tx, rx) in io.items() if tx or rx}

    # ------------------------------------------------------------------ prefill
    def record_prefill(self, result, state: str) -> None:
        super().record_prefill(result, state)
        self._bin_prefill(result)

    def record_unfinished_prefill(self, scheduled) -> None:
        """A batch still computing when the run stopped: bin its pool load and compute.

        With transfers booked at completion, such a batch never reaches
        ``record_prefill``; its work before the horizon still draws power.
        """

        self._bin_prefill(scheduled)

    def _bin_prefill(self, result) -> None:
        compute = result.compute
        device_id = compute.device_id
        self._track((compute, result.pool_load.endpoint_reservation))
        works = [(item.prefill_tokens, item.context_tokens) for item in result.sessions]
        activity = self.prefill.activity(works)
        self.binner.add(compute.start_time, compute.finish_time, device_id, activity_vector(activity), "prefill")
        if self.prefill_hardware is not None:
            self._timing("prefill", compute.service_time, self.prefill.latency_s(works, self.prefill_hardware))

        chips = self.layout.chips.get(device_id, 1)
        tokens = sum(prefill for prefill, _ in works)
        if chips > 1 and tokens > 0:
            # Two ring all-reduces of activations per layer; every chip sends and receives.
            per_chip = self.tp_bytes_per_token * tokens * 2.0 * (chips - 1) / chips
            vector = np.zeros(len(ACTIVITY_KINDS))
            vector[_IO_TX] = vector[_IO_RX] = chips * per_chip
            self.binner.add(compute.start_time, compute.finish_time, device_id, vector, "tp")

        loaded = sum(item.load_tokens for item in result.sessions) * self.kv_token_bytes
        if loaded > 0.0:
            vector = np.zeros(len(ACTIVITY_KINDS))
            vector[_MEM_WR] = vector[_IO_RX] = loaded
            pool = result.pool_load
            self.binner.add(pool.start_time, pool.finish_time, device_id, vector, "pool_load")
            self.binner.add_pool(pool.start_time, pool.finish_time, loaded, "pool_load")

    def record_transfers(self, transfers) -> None:
        super().record_transfers(transfers)
        for transfer in transfers:
            reservation = transfer.reservation
            self._track(reservation.endpoint_reservations)
            vectors: dict[int, np.ndarray] = {}
            source = vectors.setdefault(transfer.source_device_id, np.zeros(len(ACTIVITY_KINDS)))
            sent = transfer.total_tokens * self.kv_token_bytes
            source[_MEM_RD] += sent
            source[_IO_TX] += sent
            for admission in transfer.admissions:
                received = admission.transfer_tokens * self.kv_token_bytes
                vector = vectors.setdefault(admission.device_id, np.zeros(len(ACTIVITY_KINDS)))
                vector[_MEM_WR] += received
                vector[_IO_RX] += received
            ids = sorted(vectors)
            self.binner.add_many(
                reservation.start_time,
                reservation.finish_time,
                ids,
                np.vstack([vectors[device_id] for device_id in ids]),
                "kv_p2p",
            )

    def record_pool_write(self, pending) -> None:
        super().record_pool_write(pending)
        reservation = pending.reservation
        endpoint = reservation.endpoint_reservation
        self._track((endpoint,))
        written = pending.payload_tokens * self.kv_token_bytes
        if written <= 0.0:
            return
        vector = np.zeros(len(ACTIVITY_KINDS))
        vector[_MEM_RD] = vector[_IO_TX] = written
        self.binner.add(reservation.start_time, reservation.finish_time, endpoint.device_id, vector, "pool_offload")
        self.binner.add_pool(reservation.start_time, reservation.finish_time, written, "pool_offload")

    # ------------------------------------------------------------- diagnostics
    def timing_summary(self) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for kind, ratios in self.timing_ratios.items():
            if not ratios:
                continue
            values = np.asarray(ratios)
            summary[kind] = {
                "samples": int(values.size),
                "median": float(np.median(values)),
                "p5": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
        return summary

    def seen_counter(self) -> Counter:
        return Counter(self.seen_reservations)
