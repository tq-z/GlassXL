#!/usr/bin/env python3
"""Generate simulator YAML from the lightweight MoE roofline estimator."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roofline_model import (  # noqa: E402
    DeploymentSpec,
    FFNCurve,
    HardwareSpec,
    ModelEstimate,
    OperatingPoint,
    PhaseEstimate,
    PrecisionSpec,
    estimate_model,
    resolve_model,
    supported_models,
)

OUTPUT_NAMES = ("hardware.yaml", "mapping.yaml", "roofline_report.yaml")

YAML_INPUT_KEYS = {
    "model",
    "device_flops_per_second",
    "device_memory_bandwidth_bytes_per_second",
    "dp",
    "ep",
    "prefill_tokens",
    "decode_context_tokens",
    "attn_kv_capacity_bytes",
    "ffn_all_to_all_bandwidth_bytes_per_second",
    "ffn_all_to_all_start_latency_us",
    "ffn_all_to_all_min_payload_bytes",
    "kv_p2p_bandwidth_bytes_per_second",
    "kv_p2p_start_latency_us",
    "kv_pool_bandwidth_bytes_per_second",
    "kv_pool_start_latency_us",
    "output_dir",
    "weight_bytes",
    "activation_bytes",
    "kv_bytes",
    "colocate_attn_ffn",
    "force",
}

def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate coarse roofline-calibrated hardware and mapping YAML."
    )
    parser.add_argument(
        "--model", help=f"Model alias: {', '.join(supported_models())}"
    )
    parser.add_argument("--device-flops-per-second", type=_positive_float)
    parser.add_argument(
        "--device-memory-bandwidth-bytes-per-second",
        type=_positive_float,
    )
    parser.add_argument("--dp", type=_positive_int)
    parser.add_argument("--ep", type=_positive_int)
    parser.add_argument("--prefill-tokens", type=_positive_int)
    parser.add_argument("--decode-context-tokens", type=_positive_int)
    parser.add_argument(
        "--attn-kv-capacity-bytes", type=_nonnegative_float
    )
    parser.add_argument(
        "--ffn-all-to-all-bandwidth-bytes-per-second",
        type=_positive_float,
    )
    parser.add_argument(
        "--ffn-all-to-all-start-latency-us",
        type=_nonnegative_float,
    )
    parser.add_argument(
        "--ffn-all-to-all-min-payload-bytes",
        type=_nonnegative_float,
    )
    parser.add_argument(
        "--kv-p2p-bandwidth-bytes-per-second",
        type=_positive_float,
    )
    parser.add_argument(
        "--kv-p2p-start-latency-us", type=_nonnegative_float
    )
    parser.add_argument(
        "--kv-pool-bandwidth-bytes-per-second",
        type=_positive_float,
    )
    parser.add_argument(
        "--kv-pool-start-latency-us", type=_nonnegative_float
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weight-bytes", type=_positive_float)
    parser.add_argument("--activation-bytes", type=_positive_float)
    parser.add_argument("--kv-bytes", type=_positive_float)
    parser.add_argument("--colocate-attn-ffn", action="store_true", default=None)
    parser.add_argument("--force", action="store_true", default=None)
    parser.add_argument(
        "--input-yaml",
        "--config-yaml",
        "--config",
        dest="input_yaml",
        type=Path,
        help="YAML file containing CLI destination keys; explicit flags override it",
    )
    return parser




def _yaml_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"YAML key {key!r} must be true or false")
    return value


def load_input_yaml(path: Path) -> dict[str, Any]:
    """Load and strictly validate the flat generator-input YAML document."""

    try:
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load input YAML {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("input YAML root must be a mapping")
    if any(not isinstance(key, str) for key in document):
        raise ValueError("input YAML keys must be strings")
    unknown = sorted(set(document) - YAML_INPUT_KEYS)
    if unknown:
        raise ValueError("input YAML has unknown key(s): " + ", ".join(unknown))
    return dict(document)


def merge_input_yaml(args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML values into parsed arguments; explicit CLI values win."""

    values = vars(args).copy()
    input_yaml = values.get("input_yaml")
    yaml_values = load_input_yaml(input_yaml) if input_yaml is not None else {}
    for key, value in yaml_values.items():
        if values.get(key) is None:
            values[key] = value

    required = (
        "model", "device_flops_per_second", "device_memory_bandwidth_bytes_per_second",
        "dp", "ep", "prefill_tokens", "decode_context_tokens",
        "attn_kv_capacity_bytes", "ffn_all_to_all_bandwidth_bytes_per_second",
        "ffn_all_to_all_start_latency_us", "ffn_all_to_all_min_payload_bytes",
        "kv_p2p_bandwidth_bytes_per_second", "kv_p2p_start_latency_us",
        "kv_pool_bandwidth_bytes_per_second", "kv_pool_start_latency_us",
        "output_dir",
    )
    missing = [key for key in required if values.get(key) is None]
    if missing:
        flags = ", ".join("--" + key.replace("_", "-") for key in missing)
        raise ValueError(
            f"missing required generator input(s): {flags}; provide them as flags or in --input-yaml"
        )

    for key in (
        "device_flops_per_second", "device_memory_bandwidth_bytes_per_second",
        "ffn_all_to_all_bandwidth_bytes_per_second", "kv_p2p_bandwidth_bytes_per_second",
        "kv_pool_bandwidth_bytes_per_second",
    ):
        values[key] = _positive_float(str(values[key]))
    for key in (
        "attn_kv_capacity_bytes", "ffn_all_to_all_start_latency_us",
        "ffn_all_to_all_min_payload_bytes", "kv_p2p_start_latency_us",
        "kv_pool_start_latency_us",
    ):
        values[key] = _nonnegative_float(str(values[key]))
    for key in ("dp", "ep", "prefill_tokens", "decode_context_tokens"):
        try:
            values[key] = _positive_int(str(values[key]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"YAML/CLI key {key!r} must be a positive integer") from error
    for key in ("weight_bytes", "activation_bytes", "kv_bytes"):
        values[key] = _positive_float(str(2.0 if values.get(key) is None else values[key]))
    for key in ("colocate_attn_ffn", "force"):
        value = values.get(key)
        values[key] = False if value is None else _yaml_bool(value, key)
    values["output_dir"] = Path(values["output_dir"])
    values["model"] = str(values["model"])
    return argparse.Namespace(**values)

def _expert_placements(
    num_experts: int, device_ids: Sequence[int]
) -> dict[int, list[int]]:
    quotient, remainder = divmod(num_experts, len(device_ids))
    placements: dict[int, list[int]] = {}
    start = 0
    for rank, device_id in enumerate(device_ids):
        count = quotient + (rank < remainder)
        placements[device_id] = list(range(start, start + count))
        start += count
    return placements


def _curve_document(curve: FFNCurve) -> dict[str, Any]:
    return {
        "model": "batch_latency_curve",
        "points": [
            {
                "activation_tokens": point.activation_tokens,
                "latency_us": point.latency_s * 1_000_000.0,
            }
            for point in curve.points
        ],
    }


def _curve_latency(curve: FFNCurve, tokens: int) -> float:
    if tokens <= 0:
        return 0.0
    points = curve.points
    for index in range(1, len(points)):
        right = points[index]
        if tokens <= right.activation_tokens:
            left = points[index - 1]
            fraction = (tokens - left.activation_tokens) / (
                right.activation_tokens - left.activation_tokens
            )
            return left.latency_s + fraction * (right.latency_s - left.latency_s)
    left, right = points[-2:]
    slope = (right.latency_s - left.latency_s) / (
        right.activation_tokens - left.activation_tokens
    )
    return right.latency_s + (tokens - right.activation_tokens) * slope


def _phase_document(phase: PhaseEstimate) -> dict[str, Any]:
    return {
        "flops": phase.flops,
        "bytes": phase.bytes_moved,
        "compute_latency_s": phase.compute_s,
        "memory_latency_s": phase.memory_s,
        "latency_s": phase.latency_s,
        "tokens_per_second": phase.tokens_per_second,
        "bottleneck": phase.bottleneck,
    }


def _scaled_phase_document(
    phase: PhaseEstimate, count: int, token_numerator: int
) -> dict[str, Any]:
    latency = phase.latency_s * count
    return {
        "flops": phase.flops * count,
        "bytes": phase.bytes_moved * count,
        "compute_latency_s": phase.compute_s * count,
        "memory_latency_s": phase.memory_s * count,
        "latency_s": latency,
        "tokens_per_second": token_numerator / latency,
        "bottleneck": phase.bottleneck,
    }


def _curve_report(curve: FFNCurve) -> dict[str, Any]:
    return {
        "local_routed_experts": curve.local_experts,
        "points": [
            {
                "activation_tokens": point.activation_tokens,
                "latency_s": point.latency_s,
                "tokens_per_second": point.tokens_per_second,
            }
            for point in curve.points
        ],
    }


def _ffn_barrier_summary(
    estimate: ModelEstimate,
    placements: dict[int, list[int]],
    curve_by_count: dict[int, FFNCurve],
) -> list[dict[str, Any]]:
    spec = estimate.model
    result: list[dict[str, Any]] = []
    for original_tokens in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        routed_total = original_tokens * spec.top_k
        base, remainder = divmod(routed_total, spec.num_experts)
        device_rows: list[dict[str, Any]] = []
        barrier = 0.0
        for device_id, expert_ids in placements.items():
            local_tokens = base * len(expert_ids)
            local_tokens += sum(expert_id < remainder for expert_id in expert_ids)
            latency = _curve_latency(curve_by_count[len(expert_ids)], local_tokens)
            barrier = max(barrier, latency)
            device_rows.append(
                {
                    "device_id": device_id,
                    "routed_activation_tokens": local_tokens,
                    "latency_s": latency,
                }
            )
        result.append(
            {
                "original_decode_tokens": original_tokens,
                "total_routed_activations": routed_total,
                "barrier_latency_s": barrier,
                "original_tokens_per_second": original_tokens / barrier,
                "devices": device_rows,
            }
        )
    return result


def build_documents(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    args = merge_input_yaml(args)
    model = resolve_model(args.model)
    if args.ep > model.num_experts:
        raise ValueError(
            f"ep={args.ep} exceeds routed expert count {model.num_experts}"
        )
    if args.colocate_attn_ffn and args.dp != args.ep:
        raise ValueError("--colocate-attn-ffn requires --dp and --ep to be equal")

    hardware_spec = HardwareSpec(
        args.device_flops_per_second,
        args.device_memory_bandwidth_bytes_per_second,
    )
    precision = PrecisionSpec(
        args.weight_bytes, args.activation_bytes, args.kv_bytes
    )
    deployment = DeploymentSpec(args.dp, args.ep)
    operating_point = OperatingPoint(
        args.prefill_tokens, args.decode_context_tokens
    )
    estimate = estimate_model(
        model, hardware_spec, precision, deployment, operating_point
    )

    attention_ids = list(range(1, 1 + args.dp))
    ffn_ids = (
        attention_ids.copy()
        if args.colocate_attn_ffn
        else list(range(1 + args.dp, 1 + args.dp + args.ep))
    )
    placements = _expert_placements(model.num_experts, ffn_ids)
    curve_by_count = {
        item.local_experts: item.all_layers for item in estimate.ffn_curve_sets
    }

    capacity_tokens = math.floor(
        args.attn_kv_capacity_bytes / estimate.effective_kv_bytes_per_token
    )
    all_to_all_rate = (
        args.ffn_all_to_all_bandwidth_bytes_per_second
        / estimate.routed_activation_bytes
    )
    min_package_tokens = math.ceil(
        args.ffn_all_to_all_min_payload_bytes
        / estimate.routed_activation_bytes
    )
    kv_p2p_rate = (
        args.kv_p2p_bandwidth_bytes_per_second
        / estimate.effective_kv_bytes_per_token
    )
    kv_pool_rate = (
        args.kv_pool_bandwidth_bytes_per_second
        / estimate.effective_kv_bytes_per_token
    )

    capabilities: dict[int, dict[str, Any]] = {
        0: {
            "prefill": {
                "model": "aggregate_throughput",
                "tokens_per_second": estimate.prefill_all_layers.tokens_per_second,
            }
        }
    }
    for device_id in attention_ids:
        capabilities.setdefault(device_id, {})["decode_attn"] = {
            "model": "linear_context",
            "context_tokens_per_second": (
                estimate.decode_attention_all_layers.tokens_per_second
            ),
            "kv_capacity_tokens": capacity_tokens,
        }
    for device_id in ffn_ids:
        local_count = len(placements[device_id])
        capabilities.setdefault(device_id, {})["decode_ffn"] = _curve_document(
            curve_by_count[local_count]
        )

    hardware = {
        "schema_version": 1,
        "devices": [
            {"id": device_id, "capabilities": capabilities[device_id]}
            for device_id in sorted(capabilities)
        ],
        "kv_pool": {
            "capacity_tokens": None,
            "load_tokens_per_second": kv_pool_rate,
            "offload_tokens_per_second": kv_pool_rate,
            "start_latency_us": args.kv_pool_start_latency_us,
        },
        "interconnects": {
            "ffn_all_to_all": {
                "model": "startup_min_package",
                "start_latency_us": args.ffn_all_to_all_start_latency_us,
                "min_package_tokens": min_package_tokens,
                "throughput_tokens_per_second": all_to_all_rate,
            },
            "kv_p2p": {
                "model": "startup_throughput",
                "start_latency_us": args.kv_p2p_start_latency_us,
                "throughput_tokens_per_second": kv_p2p_rate,
            },
        },
    }
    mapping = {
        "schema_version": 1,
        "granularity": "decode_step",
        "model": {
            "num_layers": model.num_layers,
            "num_experts": model.num_experts,
            "top_k": model.top_k,
        },
        "prefill": {
            "device_ids": [0],
            "state": "stateful",
            "placement": "sticky_round_robin",
        },
        "decode_attn": {
            "device_ids": attention_ids,
            "placement": "least_active_context",
            "residency": "lru_cache",
            "capacity_reservation": {"model": "oracle_output"},
        },
        "decode_ffn": {
            "device_ids": ffn_ids,
            "experts_by_device": placements,
        },
    }

    curve_sets = {
        item.local_experts: item for item in estimate.ffn_curve_sets
    }
    group_rows = []
    for group_estimate in estimate.layer_groups:
        group = group_estimate.group
        ffn_curves = {}
        for local_count, curve_set in curve_sets.items():
            group_curve = dict(curve_set.layer_groups)[group.name]
            ffn_curves[local_count] = _curve_report(group_curve)
        group_rows.append(
            {
                "name": group.name,
                "layer_count": group.count,
                "layer_indices": group.layer_indices,
                "attention_kind": group.attention.kind,
                "ffn_kind": group.ffn_kind,
                "prefill": {
                    "per_layer": _phase_document(group_estimate.prefill),
                    "all_group_layers": _scaled_phase_document(
                        group_estimate.prefill, group.count, args.prefill_tokens
                    ),
                },
                "decode_attention": {
                    "per_layer": _phase_document(
                        group_estimate.decode_attention
                    ),
                    "all_group_layers": _scaled_phase_document(
                        group_estimate.decode_attention,
                        group.count,
                        args.decode_context_tokens,
                    ),
                },
                "decode_ffn_all_group_layers": ffn_curves,
            }
        )

    report = {
        "schema_version": 1,
        "model": {
            "requested_alias": args.model,
            "canonical_key": model.key,
            "display_name": model.display_name,
            "official_sources": list(model.source_urls),
            "num_main_layers": model.num_layers,
            "num_routed_experts": model.num_experts,
            "top_k": model.top_k,
            "num_shared_experts": model.shared_experts,
            "hidden_size": model.hidden_size,
            "routed_expert_input_width": model.routed_activation_width,
            "excluded": ["vision towers", "MTP/speculative auxiliary layers"],
        },
        "inputs": {
            "hardware": asdict(hardware_spec),
            "precision": asdict(precision),
            "deployment": asdict(deployment),
            "operating_point": asdict(operating_point),
            "effective_rates": True,
            "si_units": "decimal",
        },
        "token_conversions": {
            "routed_activation_bytes_per_token": estimate.routed_activation_bytes,
            "growing_kv_elements_per_model_token": (
                estimate.growing_kv_elements_per_token
            ),
            "fixed_state_elements_per_session": (
                estimate.fixed_state_elements_per_session
            ),
            "effective_kv_bytes_per_model_token_at_reference_context": (
                estimate.effective_kv_bytes_per_token
            ),
            "attention_kv_capacity_tokens": capacity_tokens,
            "ffn_all_to_all_tokens_per_second": all_to_all_rate,
            "ffn_all_to_all_min_package_tokens": min_package_tokens,
            "kv_p2p_tokens_per_second": kv_p2p_rate,
            "kv_pool_tokens_per_second": kv_pool_rate,
        },
        "layer_groups": group_rows,
        "all_layers": {
            "prefill": _phase_document(estimate.prefill_all_layers),
            "decode_attention_per_device": _phase_document(
                estimate.decode_attention_all_layers
            ),
            "decode_attention_dp_aggregate_context_tokens_per_second": (
                args.dp
                * estimate.decode_attention_all_layers.tokens_per_second
            ),
            "decode_ffn_per_device_curves": {
                local_count: _curve_report(curve_set.all_layers)
                for local_count, curve_set in curve_sets.items()
            },
            "decode_ffn_ep_barrier_summary": _ffn_barrier_summary(
                estimate, placements, curve_by_count
            ),
        },
        "assumptions": [
            "Prefill uses compute latency and is calibrated at the supplied P and existing C.",
            "Supplied FLOP/s and byte/s rates are effective; no hidden efficiency is applied.",
            "Routing is perfectly balanced; routed activations already include top-k.",
            "Dense/shared FFN work is ideally EP-sharded; dispatch/combine stays separate.",
            "Compressed MLA and reference-context amortization of recurrent state are used.",
            "This is a coarse lower-bound calibration, not a kernel/capacity simulator.",
        ],
    }
    return hardware, mapping, report


def write_documents(
    output_dir: Path,
    documents: Sequence[dict[str, Any]],
    *,
    force: bool,
) -> tuple[Path, ...]:
    paths = tuple(output_dir / name for name in OUTPUT_NAMES)
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to replace existing output(s): {names}; use --force"
        )
    rendered = [
        yaml.safe_dump(document, sort_keys=False) for document in documents
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, text in zip(paths, rendered, strict=True):
        path.write_text(text, encoding="utf-8")
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = merge_input_yaml(parser.parse_args(argv))
        documents = build_documents(args)
        paths = write_documents(
            args.output_dir, documents, force=args.force
        )
    except (ValueError, FileExistsError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
