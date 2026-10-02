from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from agentic_sim.config import load_config
from agentic_sim.metrics import MetricsCollector
from agentic_sim.runtime.runtime import AgenticRuntime
from agentic_sim.workload import SessionSpec, TurnSpec
from power_model import ActivityBinner, BinnedActivity, DeviceLayout, load_power_config, price
from power_model.binner import ACTIVITY_KINDS, KIND_INDEX
from power_model.checks import anchors, completeness, conservation, io_balance
from power_model.simulate import simulate
from roofline_model import PrecisionSpec, resolve_model
from scripts.generate_config import build_documents, build_parser, write_documents

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POWER_YAML = REPOSITORY_ROOT / "configs" / "power" / "panagretti_b200.yaml"
SMOKE_GENERATOR = REPOSITORY_ROOT / "configs" / "power" / "smoke_generator.yaml"
WORKLOAD_YAML = REPOSITORY_ROOT / "configs" / "glassml" / "workload" / "workload.yaml"

EXPLICIT = {
    "nmp": {
        "e_mac_gemm_pj": 0.6,
        "e_operand_fill_pj_per_bit": 0.02,
        "e_io_pj_per_bit": 0.1,
        "e_pic_pj_per_bit": 0.02,
        "p_dram_static_w": 4.0,
        "p_logic_static_w": 30.0,
        "p_io_static_w": 10.0,
    },
    "gpu": {
        "e_hbm_pj_per_bit": 4.0,
        "e_link_pj_per_bit": 1.0,
        "p_xpu_static_w": 150.0,
        "p_hbm_static_w": 20.0,
        "p_io_static_w": 10.0,
        "e_flop_pj": 0.5,
    },
    "kv_pool": {"e_pool_pj_per_bit": 2.0},
}


def write_power(tmp_path: Path, *, allow: bool = False, overrides: dict | None = None) -> Path:
    document = yaml.safe_load(POWER_YAML.read_text(encoding="utf-8"))
    document["allow_placeholders"] = allow
    for section, values in (overrides or {}).items():
        document[section]["overrides"] = dict(values)
    path = tmp_path / "power.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Qwen3-235B on 1 prefill device (TP2) + 4 co-located NMP nodes."""

    out = tmp_path_factory.mktemp("tiny_power")
    args = build_parser().parse_args(
        [
            "--input-yaml", str(SMOKE_GENERATOR),
            "--output-dir", str(out),
            "--dp", "4", "--ep", "4",
            "--prefill-devices", "1", "--prefill-tp", "2",
        ]
    )
    documents = build_documents(args)
    write_documents(out, documents, force=False)
    return out, documents[2]


def tiny_sessions(count: int = 6) -> tuple[SessionSpec, ...]:
    return tuple(
        SessionSpec(
            session_id=index,
            arrival_time_s=0.05 * index,
            initial_context_tokens=2000 + 500 * index,
            turns=tuple(
                TurnSpec(turn_index=turn, prefill_tokens=40 + turn, output_tokens=12, tool_wait_s=0.02)
                for turn in range(3)
            ),
        )
        for index in range(count)
    )


def run_tiny(tiny, tmp_path, sessions, *, horizon=None, track=False):
    directory, report = tiny
    config = load_config(directory / "hardware.yaml", WORKLOAD_YAML, directory / "mapping.yaml")
    if horizon is not None:
        config = replace(config, workload=replace(config.workload, simulation_end_s=horizon))
    power = load_power_config(write_power(tmp_path, overrides=EXPLICIT))
    run = simulate(
        config,
        power,
        resolve_model(power.model),
        PrecisionSpec(),
        report=report,
        session_specs=sessions,
        horizon_s=horizon,
        track_reservations=track,
    )
    return config, power, run


def test_binner_conserves_activity_and_clips_at_horizon() -> None:
    binner = ActivityBinner(0.1, (3, 7), horizon_s=1.0)
    vector = np.arange(1.0, 8.0)
    binner.add(0.05, 0.35, 3, vector, "attn")
    binner.add(0.9, 1.4, 7, vector, "ffn")  # 20% inside the horizon
    binner.add(0.35, 0.35, 7, vector, "prefill")  # zero length -> one bin
    binned = binner.finalize()
    assert binned.activity.shape == (10, 2, len(ACTIVITY_KINDS))
    assert np.allclose(binned.activity[:, 0, :].sum(axis=0), vector)
    assert np.allclose(binned.activity[:, 1, :].sum(axis=0), 1.2 * vector)
    assert np.allclose(binned.beyond_horizon[1], 0.8 * vector)
    assert np.allclose(binned.by_category.sum(axis=(0, 1)), binned.activity.sum(axis=(0, 1)))
    assert np.allclose(binned.activity[0, 0], vector / 6.0)
    assert np.allclose(binned.activity[3, 1], vector)
    assert binned.t_edges_s[-1] == 1.0


def test_binner_grows_without_a_horizon() -> None:
    binner = ActivityBinner(0.5, (0,))
    binner.add(0.0, 1500.0, 0, np.ones(len(ACTIVITY_KINDS)), "attn")
    binned = binner.finalize(end_time_s=1600.0)
    assert binned.activity.shape[0] == 3200
    assert math.isclose(binned.activity.sum(), len(ACTIVITY_KINDS))
    assert binned.t_edges_s[-1] == 1600.0


def test_power_config_requires_real_values_or_explicit_placeholders(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="p_logic_static_w"):
        load_power_config(write_power(tmp_path))
    placeholder = load_power_config(write_power(tmp_path, allow=True))
    assert "nmp.p_logic_static_w" in placeholder.placeholders
    assert placeholder.provenance["nmp.p_logic_static_w"].startswith("PLACEHOLDER")
    explicit = load_power_config(write_power(tmp_path, overrides=EXPLICIT))
    assert explicit.placeholders == ()
    assert explicit.nmp["p_logic_static_w"] == 30.0
    with pytest.raises(ValueError, match="unknown coefficient"):
        load_power_config(write_power(tmp_path, overrides={"nmp": {"not_a_coefficient": 1}}))


def test_pricing_matches_hand_computed_die_power(tiny, tmp_path: Path) -> None:
    directory, report = tiny
    config = load_config(directory / "hardware.yaml", WORKLOAD_YAML, directory / "mapping.yaml")
    power = load_power_config(write_power(tmp_path, overrides=EXPLICIT))
    layout = DeviceLayout.from_config(config, power, report)
    assert layout.kinds == {0: "gpu", 1: "nmp", 2: "nmp", 3: "nmp", 4: "nmp"}
    assert layout.chips[0] == 2
    binner = ActivityBinner(1.0, layout.device_ids, horizon_s=1.0)
    nmp = np.zeros(len(ACTIVITY_KINDS))
    nmp[KIND_INDEX["flop_gemm"]] = 2e12
    nmp[KIND_INDEX["mem_rd"]] = 1e12
    nmp[KIND_INDEX["io_tx"]] = 5e11
    binner.add(0.0, 1.0, 2, nmp, "attn")
    gpu = np.zeros(len(ACTIVITY_KINDS))
    gpu[KIND_INDEX["flop_attn"]] = 4e14
    gpu[KIND_INDEX["mem_wr"]] = 2e12
    gpu[KIND_INDEX["io_tx"]] = 1e12
    binner.add(0.0, 1.0, 0, gpu, "prefill")
    binner.add_pool(0.0, 1.0, 1e12, "pool_load")
    trace = price(binner.finalize(), layout, power, report)
    watts = dict(zip(trace.die_names, trace.power_w[0]))
    # DRAM: 1e12 B * 8 * 0.43 pJ / 4 dies + 4 W static.
    assert math.isclose(watts["nmp002.dram0"], 1e12 * 8 * 0.43e-12 / 4 + 4.0)
    # Logic: 1e12 MACs * 0.6 pJ + 1e12 B * 8 * 0.02 pJ operand fill + 5e11 B * 8 * 0.1 pJ I/O
    # + (30 + 10) W static.
    assert math.isclose(
        watts["nmp002.logic"], 1e12 * 0.6e-12 + 1e12 * 8 * 0.02e-12 + 5e11 * 8 * 0.1e-12 + 40.0
    )
    assert math.isclose(watts["nmp002.pic"], 5e11 * 8 * 0.02e-12 + 3.0)
    assert math.isclose(watts["nmp003.logic"], 40.0)
    # GPU device split over 2 chips: 4e14 FLOP * 0.5 pJ / 2 and 2e12 B * 8 * 4 pJ / 2.
    assert math.isclose(watts["gpu0_1.xpu"], 4e14 * 0.5e-12 / 2 + 150.0)
    assert math.isclose(watts["gpu0_0.hbm"], 2e12 * 8 * 4e-12 / 2 + 20.0)
    # Panel link on each chip's PIC die: 1e12 B * 8 * 1 pJ / 2 + 3 W heaters + 10 W drivers.
    assert math.isclose(watts["gpu0_1.pic"], 1e12 * 8 * 1e-12 / 2 + 13.0)
    assert math.isclose(watts["kv_pool"], 1e12 * 8 * 2e-12)


def test_tiny_run_satisfies_invariants(tiny, tmp_path: Path) -> None:
    config, power, run = run_tiny(tiny, tmp_path, tiny_sessions(), horizon=None, track=True)
    assert run.snapshot.summary["completed_sessions"] == 6
    assert conservation(run.binned, run.trace)["ok"]
    assert io_balance(run.binned)["ok"]
    report = completeness(run.collector, run.runtime)
    assert report["ok"], report
    assert run.collector.timing_summary()["prefill"]["median"] == pytest.approx(1.0)
    assert run.collector.timing_summary()["ffn"]["median"] == pytest.approx(1.0)
    categories = dict(zip(run.trace.meta["energy_columns"], run.trace.energy_j.sum(axis=0)))
    assert categories["attn"] > 0 and categories["ffn"] > 0 and categories["prefill"] > 0
    assert categories["kv_p2p"] > 0 and categories["a2a"] > 0

    reference = AgenticRuntime(config, session_specs=tiny_sessions(), prefill_transfers_at_completion=True)
    reference.metrics = MetricsCollector()
    reference.metrics.capture_attention(0.0, reference.attention)
    plain = reference.run()
    assert plain.summary == run.snapshot.summary
    assert plain.decode_steps == run.snapshot.decode_steps


def test_prefill_transfers_at_completion_do_not_stall_running_decode(tmp_path: Path) -> None:
    """A long prefill's KV transfer must not block decode steps that run before it is ready."""

    args = build_parser().parse_args(
        ["--input-yaml", str(SMOKE_GENERATOR), "--output-dir", str(tmp_path), "--dp", "1", "--ep", "1",
         "--prefill-devices", "1", "--prefill-tp", "2"]
    )
    write_documents(tmp_path, build_documents(args), force=False)
    config = load_config(tmp_path / "hardware.yaml", WORKLOAD_YAML, tmp_path / "mapping.yaml")
    sessions = (
        SessionSpec(0, 0.0, 1000, (TurnSpec(turn_index=0, prefill_tokens=16, output_tokens=200, tool_wait_s=0.0),)),
        # Arrives while session 0 decodes; its 300K-token context takes long to load and prefill.
        SessionSpec(1, 0.05, 300_000, (TurnSpec(turn_index=0, prefill_tokens=2000, output_tokens=4, tool_wait_s=0.0),)),
    )

    def worst_gap(at_completion: bool) -> float:
        runtime = AgenticRuntime(config, session_specs=sessions, prefill_transfers_at_completion=at_completion)
        steps = runtime.run().decode_steps
        early = [step for step in steps if step.session_ids == (0,)]
        return max(step.finish_time_s - step.ready_time_s for step in early)

    legacy, fixed = worst_gap(False), worst_gap(True)
    assert legacy > 0.1  # a session-0 step waits behind session 1's future transfer
    assert fixed < 0.02  # steps stay at their normal few-millisecond length


def test_empty_run_is_exactly_static_power(tiny, tmp_path: Path) -> None:
    _, power, run = run_tiny(tiny, tmp_path, (), horizon=2.0)
    assert np.array_equal(run.trace.power_w, np.broadcast_to(run.trace.static_w, run.trace.power_w.shape))
    assert run.trace.energy_j[:, :-1].sum() == 0.0


def test_reprice_round_trip(tiny, tmp_path: Path) -> None:
    _, power, run = run_tiny(tiny, tmp_path, tiny_sessions(3), horizon=5.0)
    path = tmp_path / "activity.npz"
    run.binned.save(path, '{"note": "test"}')
    loaded, meta = BinnedActivity.load(path)
    assert meta == '{"note": "test"}'
    repriced = price(loaded, run.layout, power, tiny[1])
    assert np.allclose(repriced.power_w, run.trace.power_w)
    doubled = replace(power, nmp={**power.nmp, "e_dram_pj_per_bit": 2 * power.nmp["e_dram_pj_per_bit"]})
    higher = price(loaded, run.layout, doubled, tiny[1])
    dram = [index for index, name in enumerate(run.trace.die_names) if ".dram" in name]
    assert (higher.power_w[:, dram] >= run.trace.power_w[:, dram]).all()


def test_perf_sim_h200_presets_and_anchor_semantics(tiny) -> None:
    directory, report = tiny
    config = load_config(directory / "hardware.yaml", WORKLOAD_YAML, directory / "mapping.yaml")
    power = load_power_config(REPOSITORY_ROOT / "configs" / "power" / "h200_perf_sim.yaml")
    assert (power.nmp_preset, power.gpu_preset) == ("perf_sim_default_nmp", "h200")
    # Wang et al. CICC 2024 Table 1 (w/ undercut): heaters per 1.024 Tb/s link x 63 links.
    heaters = (206 + 59.6 + 28) * 1e-15 * 1.024e12 * 63
    assert power.nmp["p_pic_static_w"] == pytest.approx(heaters, rel=1e-3)
    assert power.gpu["p_pic_static_w"] == pytest.approx(heaters, rel=1e-3)
    assert "nmp.p_pic_static_w" not in power.placeholders
    result = anchors(power, DeviceLayout.from_config(config, power, report), report)
    # No paper target applies and the derived-e_flop GPU check is an identity.
    assert result["ok"] is None
    assert "ok" not in result["nmp_pic_w"] and result["nmp_pic_w"]["paper_reference"] == 3.0
    assert "identity" in result["gpu_full_compute_and_hbm_w"]["note"]


def test_binner_has_no_zero_width_last_bin() -> None:
    for horizon in (16.01, 0.07, 10.13):
        edges = ActivityBinner(0.01, [0], horizon_s=horizon).finalize().t_edges_s
        assert np.diff(edges).min() > 0.0
    binner = ActivityBinner(0.01, [0])
    binner.add(0.0, 0.01, 0, np.ones(len(ACTIVITY_KINDS)), "attn")
    binner.add(0.02, 0.02, 0, np.ones(len(ACTIVITY_KINDS)), "attn")  # zero length on the final edge
    finished = binner.finalize()
    assert finished.activity.sum() == pytest.approx(finished.by_category.sum())


def test_anchors_match_paper_die_powers(tiny, tmp_path: Path) -> None:
    directory, report = tiny
    config = load_config(directory / "hardware.yaml", WORKLOAD_YAML, directory / "mapping.yaml")
    power = load_power_config(write_power(tmp_path, allow=True))
    result = anchors(power, DeviceLayout.from_config(config, power, report), report)
    assert result["ok"], result
    assert result["nmp_dram_die_w"]["value"] == pytest.approx(58.9, rel=1e-3)
    assert result["gpu_full_compute_and_hbm_w"]["value"] == pytest.approx(1000.0)
    assert result["gpu_compute_only_w"] < 1000.0
