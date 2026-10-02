"""Real MoE routing from captured traces, used to attribute FFN and all-to-all energy.

Traces: ``moe_routing/routing/<tag>_<instance>__segNN__<call>.npy`` hold uint8
top-k expert ids with shape (tokens, layers, top_k), captured by vLLM while a
coding agent ran SWE-bench tasks; ``qwen_moe_trace.jsonl`` maps each LLM call
to its rows. For a round, rows [input_tokens_total, row_end) are the tokens the
model generated (decode); earlier rows are prefill.

Each simulated session replays a contiguous stream of one trace session's
decode rows from a seeded offset, so expert popularity and token-to-token reuse
survive; batching comes from the simulator. The simulator's timing is not
changed: routing only decides which chip spends expert FFN and all-to-all
energy. Expert ownership is read from ``mapping.yaml`` so energy and the
simulator's FFN curves agree on placement.
"""

from __future__ import annotations

import fnmatch
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agentic_sim.config.models import MappingConfig
from roofline_model import DeploymentSpec, HardwareSpec, ModelSpec, PrecisionSpec
from roofline_model.activity import ffn_layer_activity, moe_layer_count

from .binner import ACTIVITY_KINDS


@dataclass(frozen=True)
class DecodeRowPool:
    """Per trace session, its decode rows in round order: uint8 [rows, layers, top_k]."""

    streams: tuple[np.ndarray, ...]
    names: tuple[str, ...]
    num_layers: int
    num_experts: int
    top_k: int
    rounds_used: int
    excluded_rounds: dict[str, int]

    @property
    def total_rows(self) -> int:
        return int(sum(len(stream) for stream in self.streams))

    def all_rows(self) -> np.ndarray:
        return np.concatenate(self.streams, axis=0)

    @classmethod
    def load(
        cls,
        routing_dir: Path,
        *,
        num_layers: int,
        num_experts: int,
        top_k: int,
        file_glob: str = "235b_*.npy",
        trace_jsonl: str = "qwen_moe_trace.jsonl",
    ) -> DecodeRowPool:
        routing_dir = Path(routing_dir)
        # Rounds without routing (failed calls) have no file name; keep those whose
        # session carries the same run tag so they are counted as excluded.
        tag = file_glob.split("*")[0]
        rounds: list[dict[str, Any]] = []
        with (routing_dir / trace_jsonl).open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                name = (record.get("routing") or {}).get("file")
                instance = str(record.get("session_id", "")).split(":")[-1]
                if fnmatch.fnmatch(name, file_glob) if name else instance.startswith(tag):
                    rounds.append(record)
        rounds.sort(key=lambda item: (item["session_id"], item["round_index"]))
        excluded: Counter = Counter()
        arrays: dict[str, np.ndarray] = {}
        per_session: dict[str, list[np.ndarray]] = {}
        used = 0
        for record in rounds:
            routing = record.get("routing")
            if not routing:
                excluded["no_routing"] += 1
                continue
            if routing.get("layers") != num_layers or routing.get("top_k") != top_k:
                excluded["shape_mismatch"] += 1
                continue
            path = routing_dir / "routing" / routing["file"]
            if routing["file"] not in arrays:
                array = np.load(path, mmap_mode="r")
                if array.ndim != 3 or array.shape[1:] != (num_layers, top_k):
                    raise ValueError(f"{path}: expected (tokens, {num_layers}, {top_k}), got {array.shape}")
                arrays[routing["file"]] = array
            array = arrays[routing["file"]]
            start = int(record["input_tokens_total"])
            end = int(routing["row_end"])
            if routing["row_start"] >= routing["row_end"] or end > len(array) or start < routing["row_start"]:
                excluded["bad_row_pointer"] += 1
                continue
            if end <= start:
                excluded["no_decode_rows"] += 1
                continue
            per_session.setdefault(record["session_id"], []).append(np.asarray(array[start:end]))
            used += 1
        if not per_session:
            raise ValueError(f"no decode rows matched {file_glob} in {routing_dir}")
        names = tuple(sorted(per_session))
        streams = tuple(np.ascontiguousarray(np.concatenate(per_session[name], axis=0)) for name in names)
        largest = max(int(stream.max()) for stream in streams)
        if largest >= num_experts:
            raise ValueError(f"expert id {largest} out of range for {num_experts} experts")
        return cls(streams, names, num_layers, num_experts, top_k, used, dict(excluded))


def ownership(mapping: MappingConfig, num_experts: int) -> tuple[tuple[int, ...], np.ndarray]:
    """(FFN device ids, expert -> column index into those ids)."""

    device_ids = tuple(mapping.decode_ffn.device_ids)
    column = {device_id: index for index, device_id in enumerate(device_ids)}
    owner = np.full(num_experts, -1, dtype=np.int64)
    for placement in mapping.decode_ffn.experts_by_device:
        owner[list(placement.expert_ids)] = column[placement.device_id]
    if (owner < 0).any():
        raise ValueError("every expert needs an owner in mapping.yaml")
    return device_ids, owner


def layer_device_loads(rows: np.ndarray, owner: np.ndarray, devices: int) -> tuple[np.ndarray, np.ndarray]:
    """Per layer and FFN device: routed tokens x[L, D] and distinct experts hit k[L, D]."""

    batch, layers, _ = rows.shape
    experts = owner.size
    flat = (np.arange(layers, dtype=np.int64)[None, :, None] * experts + rows).reshape(-1)
    counts = np.bincount(flat, minlength=layers * experts).reshape(layers, experts)
    one_hot = np.zeros((experts, devices))
    one_hot[np.arange(experts), owner] = 1.0
    return counts @ one_hot, (counts > 0) @ one_hot


class RoutingSource:
    """Seeded per-session cursors over real decode rows, plus per-step aggregation."""

    def __init__(self, pool: DecodeRowPool, mapping: MappingConfig, *, seed: int = 0) -> None:
        if (mapping.model.num_experts, mapping.model.top_k) != (pool.num_experts, pool.top_k):
            raise ValueError("routing traces and mapping.yaml disagree on experts or top_k")
        self.pool = pool
        self.ffn_ids, self.owner = ownership(mapping, pool.num_experts)
        self.column = {device_id: index for index, device_id in enumerate(self.ffn_ids)}
        self.rng = np.random.default_rng(seed)
        self.cursors: dict[int, list[int]] = {}
        self._routed: np.ndarray | None = None
        self.steps = 0
        self.copies = 0.0
        self.remote_copies = 0.0
        self.whole_step_imbalance: list[float] = []
        self.layer_imbalance: list[float] = []
        self.tokens_by_device = np.zeros(len(self.ffn_ids))

    # ---------------------------------------------------------------- streams
    def rows(self, session_ids) -> np.ndarray:
        out = np.empty((len(session_ids), self.pool.num_layers, self.pool.top_k), dtype=np.uint8)
        for index, session_id in enumerate(session_ids):
            cursor = self.cursors.get(session_id)
            if cursor is None:
                stream = int(self.rng.integers(len(self.pool.streams)))
                cursor = [stream, int(self.rng.integers(len(self.pool.streams[stream])))]
                self.cursors[session_id] = cursor
            stream = self.pool.streams[cursor[0]]
            out[index] = stream[cursor[1]]
            cursor[1] = (cursor[1] + 1) % len(stream)
        return out

    # -------------------------------------------------------------- collector
    def _coefficients(self, collector) -> dict[str, Any]:
        """Routed activity per token and per active expert, per MoE layer."""

        if self._routed is None:
            model: ModelSpec = collector.model
            moe = next(group for group in model.layer_groups if group.ffn_kind == "moe")
            one = ffn_layer_activity(model, moe, collector.precision, collector.deployment, 1, 1, 0)
            two = ffn_layer_activity(model, moe, collector.precision, collector.deployment, 2, 1, 0)
            per_token = _vector(two) - _vector(one)
            self._routed = np.vstack([per_token, _vector(one) - per_token])
            if moe_layer_count(model) != self.pool.num_layers:
                raise ValueError(
                    f"traces have {self.pool.num_layers} routed layers, model has {moe_layer_count(model)}"
                )
        return {"per_token": self._routed[0], "per_expert": self._routed[1]}

    def step_activity(self, collector, result):
        batch = result.batch
        session_ids = [session.session_id for session in batch.sessions]
        device_of = {
            session.session_id: local.device_id
            for local in batch.local_batches
            for session in local.sessions
        }
        rows = self.rows(session_ids)
        devices = len(self.ffn_ids)
        x, k = layer_device_loads(rows, self.owner, devices)
        coefficients = self._coefficients(collector)

        # Non-routed work (shared/common experts, dense layers) stays evenly EP-sharded.
        share = len(session_ids) * collector.model.top_k / devices
        nonrouted = np.zeros(len(ACTIVITY_KINDS))
        for group in collector.model.layer_groups:
            nonrouted += group.count * _vector(
                ffn_layer_activity(
                    collector.model, group, collector.precision, collector.deployment, 0, 0, share
                )
            )
        tokens = x.sum(axis=0)
        experts = k.sum(axis=0)
        vectors = (
            tokens[:, None] * coefficients["per_token"][None, :]
            + experts[:, None] * coefficients["per_expert"][None, :]
            + nonrouted[None, :]
        )
        ffn_ids = list(self.ffn_ids)

        # All-to-all: a (token, expert) copy is remote when its session's
        # attention device is not the expert's owner. Dispatch there, combine back.
        session_column = np.array([self.column.get(device_of[sid], -1) for sid in session_ids])
        owner_column = self.owner[rows]
        remote = owner_column != session_column[:, None, None]
        per_session = remote.sum(axis=(1, 2)).astype(float)
        received = np.bincount(owner_column[remote], minlength=devices).astype(float)
        scale = collector.copy_bytes
        io: dict[int, list[float]] = {}
        for session_id, sent in zip(session_ids, per_session):
            entry = io.setdefault(device_of[session_id], [0.0, 0.0])
            entry[0] += sent * scale
            entry[1] += sent * scale
        for column, count in enumerate(received):
            if count:
                entry = io.setdefault(self.ffn_ids[column], [0.0, 0.0])
                entry[1] += count * scale
                entry[0] += count * scale

        self._record(x, tokens, remote)
        if collector.ffn_hardware is not None:
            # Simulator FFN phase (skew-aware curve) vs this step's real straggler time.
            collector._timing(
                "ffn_routed",
                result.ffn_barrier_s - result.dispatch.finish_time,
                self.step_time(x, k, share, collector),
            )
        keep = [index for index, value in enumerate(tokens) if value > 0 or nonrouted.any()]
        return (
            [ffn_ids[index] for index in keep],
            [vectors[index] for index in keep],
            {device_id: (tx, rx) for device_id, (tx, rx) in io.items() if tx or rx},
        )

    def step_time(self, x: np.ndarray, k: np.ndarray, share: float, collector) -> float:
        """Real straggler FFN time: sum over layers of the slowest device (roofline)."""

        return layer_folded_time(
            x, k, share, collector.model, collector.ffn_hardware, collector.precision, collector.deployment
        )

    # ------------------------------------------------------------ statistics
    def _record(self, x: np.ndarray, tokens: np.ndarray, remote: np.ndarray) -> None:
        self.steps += 1
        self.copies += remote.size
        self.remote_copies += float(remote.sum())
        self.tokens_by_device += tokens
        mean = tokens.mean()
        if mean > 0 and len(self.whole_step_imbalance) < 100_000:
            self.whole_step_imbalance.append(float(tokens.max() / mean))
            layer_mean = x.mean(axis=1)
            valid = layer_mean > 0
            self.layer_imbalance.append(float((x.max(axis=1)[valid] / layer_mean[valid]).mean()))

    def summary(self) -> dict[str, Any]:
        cumulative = self.tokens_by_device
        mean = cumulative.mean() if cumulative.size else 0.0
        order = np.argsort(cumulative)[::-1]
        return {
            "ok": True,
            "trace_sessions": len(self.pool.streams),
            "decode_rows": self.pool.total_rows,
            "rounds_used": self.pool.rounds_used,
            "excluded_rounds": self.pool.excluded_rounds,
            "steps": self.steps,
            "simulated_sessions_routed": len(self.cursors),
            "remote_copy_fraction": self.remote_copies / self.copies if self.copies else 0.0,
            "whole_step_device_max_over_mean_median": (
                float(np.median(self.whole_step_imbalance)) if self.whole_step_imbalance else None
            ),
            "per_layer_device_max_over_mean_median": (
                float(np.median(self.layer_imbalance)) if self.layer_imbalance else None
            ),
            "cumulative_device_max_over_mean": float(cumulative.max() / mean) if mean > 0 else None,
            "cumulative_device_min_over_mean": float(cumulative.min() / mean) if mean > 0 else None,
            "hottest_devices": [int(self.ffn_ids[index]) for index in order[:8]],
        }


def _vector(activity) -> np.ndarray:
    return np.array(
        [activity.flops_gemm, activity.flops_attn, activity.weight_rd + activity.kv_rd,
         activity.kv_wr, activity.act, 0.0, 0.0]
    )


def layer_folded_time(
    x: np.ndarray,
    k: np.ndarray,
    share: float,
    model: ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    deployment: DeploymentSpec,
) -> float:
    """sum_l max_d of the roofline FFN time of device d at MoE layer l.

    Routed GEMMs use each device's own (tokens, distinct experts); shared,
    common and dense work uses the even EP share, as in ``_ffn_layer_work``.
    """

    matrix = model.routed_width * model.expert_intermediate_size
    flops = 2.0 * x * matrix
    memory = np.minimum(k, x) * matrix * precision.weight_bytes + x * (
        model.routed_width + model.expert_intermediate_size
    ) * precision.activation_bytes
    routed = 3.0 * np.maximum(flops / hardware.flops_per_second, memory / hardware.memory_bandwidth_bytes_per_second)
    original = share / model.top_k
    shared = 0.0
    dense = 0.0
    if share > 0:
        if model.shared_active_params:
            shared += max(
                2.0 * original * model.shared_active_params / hardware.flops_per_second,
                (model.shared_active_params * precision.weight_bytes / deployment.ep
                 + original * model.shared_experts * 3.0
                 * (model.hidden_size + model.shared_intermediate_size) * precision.activation_bytes)
                / hardware.memory_bandwidth_bytes_per_second,
            )
        if model.moe_common_active_params:
            shared += max(
                2.0 * original * model.moe_common_active_params / hardware.flops_per_second,
                (model.moe_common_active_params * precision.weight_bytes / deployment.ep
                 + original * 4.0 * model.hidden_size * precision.activation_bytes)
                / hardware.memory_bandwidth_bytes_per_second,
            )
        for group in model.layer_groups:
            if group.ffn_kind == "dense":
                dense += group.count * max(
                    2.0 * original * model.dense_active_params / hardware.flops_per_second,
                    (model.dense_active_params * precision.weight_bytes / deployment.ep
                     + original * 3.0 * (model.hidden_size + model.dense_intermediate_size)
                     * precision.activation_bytes)
                    / hardware.memory_bandwidth_bytes_per_second,
                )
    per_layer = routed.max(axis=1) + np.where(x.sum(axis=1) > 0, shared, 0.0)
    return float(per_layer.sum() + dense)
