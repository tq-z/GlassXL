from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from agentic_sim.config import load_config, load_hardware_config
from agentic_sim.hardware import Device
from power_model import load_power_config
from power_model.binner import CATEGORIES, KIND_INDEX
from power_model.checks import conservation, io_balance
from power_model.routing import DecodeRowPool, RoutingSource, layer_device_loads
from power_model.simulate import simulate
from roofline_model import PrecisionSpec, resolve_model
from scripts.generate_config import build_documents, build_parser, write_documents

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROUTING_DIR = Path(os.environ.get("MOE_ROUTING_DIR", REPOSITORY_ROOT.parent / "moe_routing"))
SMOKE_GENERATOR = REPOSITORY_ROOT / "configs" / "power" / "smoke_generator.yaml"
POWER_YAML = REPOSITORY_ROOT / "configs" / "power" / "panagretti_b200.yaml"
WORKLOAD_YAML = REPOSITORY_ROOT / "configs" / "glassml" / "workload" / "workload.yaml"

pytestmark = pytest.mark.skipif(
    not (ROUTING_DIR / "qwen_moe_trace.jsonl").exists(),
    reason="moe_routing traces not available (set MOE_ROUTING_DIR)",
)


@pytest.fixture(scope="module")
def pool() -> DecodeRowPool:
    return DecodeRowPool.load(ROUTING_DIR, num_layers=94, num_experts=128, top_k=8)


def test_pool_uses_decode_rows_of_the_235b_run_only(pool: DecodeRowPool) -> None:
    assert len(pool.streams) == 20
    assert pool.total_rows == 47_376
    assert pool.rounds_used == 350
    assert pool.excluded_rounds == {"no_routing": 10}
    assert all(stream.dtype == np.uint8 and stream.shape[1:] == (94, 8) for stream in pool.streams)


def test_routing_statistics_reproduce_measured_targets(pool: DecodeRowPool) -> None:
    rows = pool.all_rows()
    owner = np.repeat(np.arange(64), 2)  # EP64, contiguous, as the generator places experts
    x, _ = layer_device_loads(rows, owner, 64)
    per_die = x.sum(axis=0)
    mean = per_die.mean()
    assert per_die.max() / mean == pytest.approx(1.49, abs=0.03)
    assert per_die.min() / mean == pytest.approx(0.64, abs=0.03)
    assert per_die.std() / mean == pytest.approx(0.18, abs=0.02)
    reuse = np.concatenate(
        [
            (stream[:-1, :, :, None] == stream[1:, :, None, :]).any(-1).sum(-1).mean(-1) / 8.0
            for stream in pool.streams
        ]
    ).mean()
    assert reuse == pytest.approx(0.44, abs=0.01)
    sample = rows[np.random.default_rng(0).integers(0, len(rows), 1024)]
    x, k = layer_device_loads(sample, owner, 64)
    assert np.allclose(x.sum(axis=1), 1024 * 8)
    assert (k <= 2).all() and (k <= x).all()
    assert (x.max(axis=1) / x.mean(axis=1)).mean() == pytest.approx(4.9, abs=0.3)


def test_cursors_replay_contiguous_streams(tmp_path: Path, pool: DecodeRowPool) -> None:
    directory = tmp_path / "config"
    args = build_parser().parse_args(
        ["--input-yaml", str(SMOKE_GENERATOR), "--output-dir", str(directory), "--dp", "4", "--ep", "4",
         "--prefill-devices", "1", "--prefill-tp", "2"]
    )
    write_documents(directory, build_documents(args), force=False)
    config = load_config(directory / "hardware.yaml", WORKLOAD_YAML, directory / "mapping.yaml")
    source = RoutingSource(pool, config.mapping, seed=5)
    first = source.rows([7, 9])
    second = source.rows([7])
    stream, position = source.cursors[7]
    expected = pool.streams[stream][(position - 2) % len(pool.streams[stream])]
    assert np.array_equal(first[0], expected)
    assert np.array_equal(second[0], pool.streams[stream][(position - 1) % len(pool.streams[stream])])


def test_real_routing_changes_attribution_not_the_simulation(tmp_path: Path, pool: DecodeRowPool) -> None:
    from agentic_sim.workload import SessionSpec, TurnSpec

    directory = tmp_path / "config"
    args = build_parser().parse_args(
        ["--input-yaml", str(SMOKE_GENERATOR), "--output-dir", str(directory), "--dp", "4", "--ep", "4",
         "--prefill-devices", "1", "--prefill-tp", "2"]
    )
    documents = build_documents(args)
    write_documents(directory, documents, force=False)
    config = load_config(directory / "hardware.yaml", WORKLOAD_YAML, directory / "mapping.yaml")
    document = yaml.safe_load(POWER_YAML.read_text(encoding="utf-8"))
    document["nmp"]["overrides"] = {"e_io_pj_per_bit": 0.1}
    power_path = tmp_path / "power.yaml"
    power_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    power = load_power_config(power_path)
    sessions = tuple(
        SessionSpec(index, 0.03 * index, 3000 + 200 * index,
                    tuple(TurnSpec(turn_index=turn, prefill_tokens=30, output_tokens=16, tool_wait_s=0.01)
                          for turn in range(2)))
        for index in range(8)
    )
    model = resolve_model("qwen3-235b")
    runs = {
        name: simulate(
            config, power, model, PrecisionSpec(), report=documents[2], session_specs=sessions,
            routing=routing,
        )
        for name, routing in (("equal", None), ("traces", RoutingSource(pool, config.mapping, seed=1)))
    }
    equal, traces = runs["equal"], runs["traces"]
    assert equal.snapshot.summary == traces.snapshot.summary
    assert equal.snapshot.decode_steps == traces.snapshot.decode_steps
    ffn = CATEGORIES.index("ffn")
    flop = KIND_INDEX["flop_gemm"]
    assert math.isclose(
        equal.binned.by_category[:, ffn, flop].sum(), traces.binned.by_category[:, ffn, flop].sum(), rel_tol=1e-9
    )
    for run in runs.values():
        assert conservation(run.binned, run.trace)["ok"]
        assert io_balance(run.binned)["ok"]
    # Real routing reuses hot experts within a step, so fewer weights are streamed.
    weights = KIND_INDEX["mem_rd"]
    assert traces.binned.by_category[:, ffn, weights].sum() < equal.binned.by_category[:, ffn, weights].sum()
    summary = runs["traces"].collector.routing.summary()
    assert summary["steps"] == equal.snapshot.summary["decode_steps"]
    assert 0.6 < summary["remote_copy_fraction"] < 0.9  # about 3 of 4 EP ranks are remote


def test_skew_aware_curves_load_and_track_real_time(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--input-yaml", str(SMOKE_GENERATOR), "--output-dir", str(tmp_path), "--dp", "8", "--ep", "8",
         "--prefill-devices", "1", "--routing-traces", str(ROUTING_DIR),
         "--skew-samples", "8", "--skew-max-batch", "512"]
    )
    documents = build_documents(args)
    write_documents(tmp_path, documents, force=False)
    hardware = load_hardware_config(tmp_path / "hardware.yaml")
    for device in hardware.devices[1:]:
        Device.from_config(device)  # curves satisfy the simulator's validation
    skew = documents[2]["ffn_skew"]
    assert skew["decode_rows"] == 47_376
    # Per-device curves reproduce the (monotone) real straggler time at every batch size;
    # the remaining gap to raw sample means is sampling noise.
    assert skew["max_abs_error_vs_monotone_real"] <= 1e-9
    assert skew["max_abs_error_vs_measured_mean"] <= 0.10
    last = skew["table"][-1]
    assert last["balanced_over_real"] < 0.8  # balanced routing hides the hot-expert straggler
