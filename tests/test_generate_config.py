from __future__ import annotations

import math

import pytest
import yaml

from agentic_sim.config.loader import load_hardware_config, load_mapping_config
from roofline_model import resolve_model, supported_models
from scripts.generate_config import build_documents, build_parser, load_input_yaml, merge_input_yaml, write_documents


def arguments(tmp_path, model: str = "kimi-k2", *, dp: int = 2, ep: int = 4):
    return build_parser().parse_args(
        [
            "--model", model,
            "--device-flops-per-second", "1e15",
            "--device-memory-bandwidth-bytes-per-second", "3e12",
            "--dp", str(dp),
            "--ep", str(ep),
            "--prefill-tokens", "4096",
            "--decode-context-tokens", "32768",
            "--attn-kv-capacity-bytes", "8e10",
            "--ffn-all-to-all-bandwidth-bytes-per-second", "4e11",
            "--ffn-all-to-all-start-latency-us", "5",
            "--ffn-all-to-all-min-payload-bytes", "16384",
            "--kv-p2p-bandwidth-bytes-per-second", "2e11",
            "--kv-p2p-start-latency-us", "5",
            "--kv-pool-bandwidth-bytes-per-second", "1e11",
            "--kv-pool-start-latency-us", "10",
            "--output-dir", str(tmp_path),
        ]
    )


@pytest.mark.parametrize("model", supported_models())
def test_generator_outputs_load_for_every_model(tmp_path, model: str) -> None:
    args = arguments(tmp_path, model)
    documents = build_documents(args)
    write_documents(tmp_path, documents, force=False)
    hardware = load_hardware_config(tmp_path / "hardware.yaml")
    mapping = load_mapping_config(tmp_path / "mapping.yaml")
    spec = resolve_model(model)
    assert mapping.model.num_layers == spec.num_layers
    assert mapping.model.num_experts == spec.num_experts
    assert mapping.model.top_k == spec.top_k
    owned = [
        expert
        for placement in mapping.decode_ffn.experts_by_device
        for expert in placement.expert_ids
    ]
    assert sorted(owned) == list(range(spec.num_experts))
    assert mapping.prefill.device_ids == (0,)
    assert mapping.decode_attn.device_ids == (1, 2)
    assert mapping.decode_ffn.device_ids == (3, 4, 5, 6)
    assert all(
        device.capabilities.decode_ffn is not None
        for device in hardware.devices[3:]
    )


def test_token_byte_conversions_and_report_totals(tmp_path) -> None:
    args = arguments(tmp_path, "kimi-k3")
    hardware, mapping, report = build_documents(args)
    conversions = report["token_conversions"]
    assert conversions["routed_activation_bytes_per_token"] == 3584 * 2
    assert math.isclose(
        hardware["interconnects"]["ffn_all_to_all"]["throughput_tokens_per_second"],
        4e11 / (3584 * 2),
    )
    assert hardware["interconnects"]["ffn_all_to_all"]["min_package_tokens"] == math.ceil(
        16384 / (3584 * 2)
    )
    assert sum(group["layer_count"] for group in report["layer_groups"]) == 93
    assert mapping["model"]["num_layers"] == 93


def test_equal_width_colocation_and_unequal_rejection(tmp_path) -> None:
    equal = arguments(tmp_path, dp=4, ep=4)
    equal.colocate_attn_ffn = True
    hardware, mapping, _ = build_documents(equal)
    assert mapping["decode_attn"]["device_ids"] == mapping["decode_ffn"]["device_ids"]
    assert len(hardware["devices"]) == 5
    assert set(hardware["devices"][1]["capabilities"]) == {"decode_attn", "decode_ffn"}
    unequal = arguments(tmp_path, dp=2, ep=4)
    unequal.colocate_attn_ffn = True
    with pytest.raises(ValueError, match="requires --dp and --ep to be equal"):
        build_documents(unequal)


def test_ep_limit_determinism_and_overwrite_protection(tmp_path) -> None:
    args = arguments(tmp_path, "qwen3-235b")
    first = build_documents(args)
    assert first == build_documents(args)
    write_documents(tmp_path, first, force=False)
    with pytest.raises(FileExistsError, match="use --force"):
        write_documents(tmp_path, first, force=False)
    write_documents(tmp_path, first, force=True)
    too_wide = arguments(tmp_path, "qwen3-235b", ep=129)
    with pytest.raises(ValueError, match="exceeds routed expert count"):
        build_documents(too_wide)


def test_yaml_input_loads_and_cli_overrides(tmp_path) -> None:
    values = {
        "model": "Kimi-K2",
        "device_flops_per_second": 1e15,
        "device_memory_bandwidth_bytes_per_second": 3e12,
        "dp": 2,
        "ep": 4,
        "prefill_tokens": 4096,
        "decode_context_tokens": 32768,
        "attn_kv_capacity_bytes": 8e10,
        "ffn_all_to_all_bandwidth_bytes_per_second": 4e11,
        "ffn_all_to_all_start_latency_us": 5,
        "ffn_all_to_all_min_payload_bytes": 16384,
        "kv_p2p_bandwidth_bytes_per_second": 2e11,
        "kv_p2p_start_latency_us": 5,
        "kv_pool_bandwidth_bytes_per_second": 1e11,
        "kv_pool_start_latency_us": 10,
        "output_dir": str(tmp_path / "generated"),
        "colocate_attn_ffn": False,
    }
    input_path = tmp_path / "inputs.yaml"
    input_path.write_text(yaml.safe_dump(values), encoding="utf-8")

    parsed = build_parser().parse_args(["--input-yaml", str(input_path)])
    merged = merge_input_yaml(parsed)
    assert merged.model == "Kimi-K2"
    assert merged.weight_bytes == 2.0
    assert merged.output_dir == tmp_path / "generated"
    _, _, report = build_documents(merged)
    assert report["model"]["canonical_key"] == "kimi-k2"

    overridden = build_parser().parse_args(
        ["--input-yaml", str(input_path), "--model", "QWEN3-235B", "--dp", "1"]
    )
    overridden = merge_input_yaml(overridden)
    assert overridden.model == "QWEN3-235B"
    assert overridden.dp == 1


def test_yaml_input_rejects_unknown_keys_and_missing_values(tmp_path) -> None:
    unknown_path = tmp_path / "unknown.yaml"
    unknown_path.write_text("model: Kimi-K2\nnot_an_option: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_input_yaml(unknown_path)
    with pytest.raises(ValueError, match="missing required"):
        merge_input_yaml(build_parser().parse_args([]))
