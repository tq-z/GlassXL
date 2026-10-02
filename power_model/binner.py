"""Time-binned activity counters per simulator device.

The collector bins raw activity (FLOPs and bytes by kind), not energy. Energy
is linear in the coefficients, so a run can be re-priced with new coefficients
without re-running the simulator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ACTIVITY_KINDS = ("flop_gemm", "flop_attn", "mem_rd", "mem_wr", "act", "io_tx", "io_rx")
CATEGORIES = ("prefill", "attn", "ffn", "a2a", "kv_p2p", "pool_load", "pool_offload", "tp")
KIND_INDEX = {name: index for index, name in enumerate(ACTIVITY_KINDS)}
CATEGORY_INDEX = {name: index for index, name in enumerate(CATEGORIES)}


@dataclass(frozen=True)
class BinnedActivity:
    """Finalized counters: ``activity[T, n_dev, 7]`` over ``t_edges_s``."""

    t_edges_s: np.ndarray
    device_ids: tuple[int, ...]
    activity: np.ndarray
    pool_bytes: np.ndarray
    by_category: np.ndarray
    pool_by_category: np.ndarray
    beyond_horizon: np.ndarray
    pool_beyond_horizon: float
    events: int

    @property
    def horizon_s(self) -> float:
        return float(self.t_edges_s[-1])

    @property
    def bin_widths_s(self) -> np.ndarray:
        return np.diff(self.t_edges_s)

    def save(self, path, meta_json: str = "{}") -> None:
        np.savez_compressed(
            path,
            t_edges_s=self.t_edges_s,
            device_ids=np.asarray(self.device_ids, dtype=np.int64),
            activity=self.activity,
            pool_bytes=self.pool_bytes,
            by_category=self.by_category,
            pool_by_category=self.pool_by_category,
            beyond_horizon=self.beyond_horizon,
            pool_beyond_horizon=np.asarray(self.pool_beyond_horizon),
            events=np.asarray(self.events),
            activity_kinds=np.asarray(ACTIVITY_KINDS),
            categories=np.asarray(CATEGORIES),
            meta_json=np.asarray(meta_json),
        )

    @classmethod
    def load(cls, path) -> tuple[BinnedActivity, str]:
        with np.load(path, allow_pickle=False) as data:
            if tuple(data["activity_kinds"]) != ACTIVITY_KINDS or tuple(data["categories"]) != CATEGORIES:
                raise ValueError(f"{path}: activity kinds or categories differ from this version")
            binned = cls(
                t_edges_s=data["t_edges_s"],
                device_ids=tuple(int(item) for item in data["device_ids"]),
                activity=data["activity"],
                pool_bytes=data["pool_bytes"],
                by_category=data["by_category"],
                pool_by_category=data["pool_by_category"],
                beyond_horizon=data["beyond_horizon"],
                pool_beyond_horizon=float(data["pool_beyond_horizon"]),
                events=int(data["events"]),
            )
            return binned, str(data["meta_json"])


def _bin_count(horizon_s: float, dt_s: float) -> int:
    """Bins of width dt covering [0, horizon]; a float quotient just above an
    integer (16.01 / 0.01 = 1601.0000000000002) must not add a zero-width bin."""

    return max(1, math.ceil(horizon_s / dt_s - 1e-9))


class ActivityBinner:
    """Spread each event's activity uniformly over its [t0, t1] window.

    Bins are ``dt_s`` wide from t=0. With a horizon, bins stop there and any part
    of an event past it is tallied in ``beyond_horizon``; without one, bins grow
    until ``finalize``. A zero-length event lands in the bin containing t0.
    """

    def __init__(
        self,
        dt_s: float,
        device_ids: Sequence[int],
        horizon_s: float | None = None,
    ) -> None:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if horizon_s is not None and (not math.isfinite(horizon_s) or horizon_s <= 0.0):
            raise ValueError("horizon_s must be finite and positive")
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("device ids must be unique")
        self.dt_s = float(dt_s)
        self.horizon_s = None if horizon_s is None else float(horizon_s)
        self.device_ids = tuple(device_ids)
        self.index = {device_id: position for position, device_id in enumerate(self.device_ids)}
        capacity = 1024 if horizon_s is None else _bin_count(horizon_s, dt_s)
        self._fixed_bins = None if horizon_s is None else capacity
        n_dev = len(self.device_ids)
        self.activity = np.zeros((capacity, n_dev, len(ACTIVITY_KINDS)))
        self.pool_bytes = np.zeros(capacity)
        self.by_category = np.zeros((n_dev, len(CATEGORIES), len(ACTIVITY_KINDS)))
        self.pool_by_category = np.zeros(len(CATEGORIES))
        self.beyond_horizon = np.zeros((n_dev, len(ACTIVITY_KINDS)))
        self.pool_beyond_horizon = 0.0
        self.max_time_s = 0.0
        self.events = 0

    def _ensure_capacity(self, bins: int) -> None:
        current = self.activity.shape[0]
        if bins <= current:
            return
        grown = max(bins, 2 * current)
        activity = np.zeros((grown, *self.activity.shape[1:]))
        activity[:current] = self.activity
        pool = np.zeros(grown)
        pool[:current] = self.pool_bytes
        self.activity = activity
        self.pool_bytes = pool

    def _weights(self, t0: float, t1: float) -> tuple[int, np.ndarray, float]:
        """Return (first bin, per-bin fractions, fraction past the horizon)."""

        if not (math.isfinite(t0) and math.isfinite(t1)) or t0 < 0.0 or t1 < t0:
            raise ValueError(f"invalid event window [{t0}, {t1}]")
        self.max_time_s = max(self.max_time_s, t1)
        horizon = math.inf if self.horizon_s is None else self.horizon_s
        last_bin = None if self._fixed_bins is None else self._fixed_bins - 1
        if t1 == t0:
            if t0 > horizon:
                return 0, np.zeros(0), 1.0
            first = int(t0 // self.dt_s)
            if last_bin is not None:
                first = min(first, last_bin)
            else:
                self._ensure_capacity(first + 1)
            return first, np.ones(1), 0.0
        duration = t1 - t0
        end = min(t1, horizon)
        if end <= t0:
            return 0, np.zeros(0), 1.0
        first = int(t0 // self.dt_s)
        last = max(first, math.ceil(end / self.dt_s) - 1)
        if last_bin is not None:
            first = min(first, last_bin)
            last = min(last, last_bin)
        else:
            self._ensure_capacity(last + 1)
        lower = np.arange(first, last + 1) * self.dt_s
        upper = lower + self.dt_s
        overlaps = np.minimum(end, upper) - np.maximum(t0, lower)
        weights = np.clip(overlaps, 0.0, None) / duration
        inside = (end - t0) / duration
        # Renormalize the in-horizon part so floating-point overlap sums conserve it exactly.
        total = float(weights.sum())
        if total > 0.0:
            weights *= inside / total
        return first, weights, 1.0 - inside

    def add(
        self,
        t0: float,
        t1: float,
        device_id: int,
        vector: Sequence[float] | np.ndarray,
        category: str,
    ) -> None:
        self.add_many(t0, t1, (device_id,), np.asarray(vector, dtype=float)[None, :], category)

    def add_many(
        self,
        t0: float,
        t1: float,
        device_ids: Sequence[int],
        vectors: np.ndarray,
        category: str,
    ) -> None:
        """Add one shared window for several devices (ids must be unique)."""

        if len(device_ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=float)
        if vectors.shape != (len(device_ids), len(ACTIVITY_KINDS)):
            raise ValueError("vectors must have shape (devices, activity kinds)")
        rows = np.fromiter((self.index[device_id] for device_id in device_ids), dtype=np.int64)
        if len(set(rows.tolist())) != len(rows):
            raise ValueError("add_many device ids must be unique")
        first, weights, beyond = self._weights(t0, t1)
        category_index = CATEGORY_INDEX[category]
        if weights.size:
            self.activity[first:first + weights.size, rows, :] += (
                weights[:, None, None] * vectors[None, :, :]
            )
            self.by_category[rows, category_index, :] += (1.0 - beyond) * vectors
        if beyond > 0.0:
            self.beyond_horizon[rows, :] += beyond * vectors
        self.events += 1

    def add_pool(self, t0: float, t1: float, nbytes: float, category: str) -> None:
        """Traffic at the KV-pool pseudo-die (off-panel backing store)."""

        first, weights, beyond = self._weights(t0, t1)
        if weights.size:
            self.pool_bytes[first:first + weights.size] += weights * nbytes
            self.pool_by_category[CATEGORY_INDEX[category]] += (1.0 - beyond) * nbytes
        self.pool_beyond_horizon += beyond * nbytes

    def finalize(self, end_time_s: float | None = None) -> BinnedActivity:
        if self.horizon_s is not None:
            horizon = self.horizon_s
            bins = self._fixed_bins
        else:
            # Every event ends by max_time_s, so these bins hold everything recorded,
            # except a zero-length event exactly on the final edge: fold it back in.
            horizon = max(end_time_s or 0.0, self.max_time_s, self.dt_s)
            bins = _bin_count(horizon, self.dt_s)
            self._ensure_capacity(bins)
            if self.activity.shape[0] > bins:
                self.activity[bins - 1] += self.activity[bins:].sum(axis=0)
                self.activity[bins:] = 0.0
                self.pool_bytes[bins - 1] += self.pool_bytes[bins:].sum()
                self.pool_bytes[bins:] = 0.0
        edges = np.arange(bins + 1, dtype=float) * self.dt_s
        edges[-1] = horizon
        return BinnedActivity(
            t_edges_s=edges,
            device_ids=self.device_ids,
            activity=self.activity[:bins].copy(),
            pool_bytes=self.pool_bytes[:bins].copy(),
            by_category=self.by_category.copy(),
            pool_by_category=self.pool_by_category.copy(),
            beyond_horizon=self.beyond_horizon.copy(),
            pool_beyond_horizon=self.pool_beyond_horizon,
            events=self.events,
        )
