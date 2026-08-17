"""Configurable direct-link generic topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_sim.hardware.device import DecodeSettings, Device, MemoryResource, PrefillSettings
from agentic_sim.hardware.link import Link


def seconds(config: dict[str, Any]) -> float:
    if "fixed_latency_s" in config:
        return float(config["fixed_latency_s"])
    return float(config.get("fixed_latency_us", 0.0)) / 1_000_000.0


def bandwidth_bytes_per_second(config: dict[str, Any]) -> float:
    if "bandwidth_bytes_per_sec" in config:
        return float(config["bandwidth_bytes_per_sec"])
    if "bandwidth_gbps" in config:
        return float(config["bandwidth_gbps"]) * 1_000_000_000.0 / 8.0
    raise ValueError("bandwidth_bytes_per_sec or bandwidth_gbps is required")


def _prefill_settings(item: dict[str, Any]) -> PrefillSettings | None:
    nested = item.get("prefill")
    if isinstance(nested, dict):
        return PrefillSettings(
            single_request_tokens_per_second=float(nested["single_request_tokens_per_second"]),
            max_system_tokens_per_second=float(nested["max_system_tokens_per_second"]),
            max_batch_requests=int(nested["max_batch_requests"]),
            max_batch_tokens=int(nested["max_batch_tokens"]),
            layout=str(nested.get("layout", "packed")),
            batch_wait=float(nested.get("batch_wait_s", float(nested.get("batch_wait_us", 0.0)) / 1_000_000.0)),
        )
    if "prefill_tokens_per_second" in item:
        rate = float(item["prefill_tokens_per_second"])
        return PrefillSettings(rate, rate, 1, 2**63 - 1)
    return None


def _decode_settings(item: dict[str, Any]) -> DecodeSettings | None:
    nested = item.get("decode")
    if isinstance(nested, dict):
        return DecodeSettings(
            single_session_tokens_per_second=float(nested["single_session_tokens_per_second"]),
            max_system_tokens_per_second=float(nested["max_system_tokens_per_second"]),
            max_active_sessions=int(nested["max_active_sessions"]),
        )
    if "decode_tokens_per_second" in item:
        rate = float(item["decode_tokens_per_second"])
        return DecodeSettings(rate, rate, 1)
    return None


@dataclass
class Topology:
    devices: dict[int, Device]
    memories: dict[int, MemoryResource]
    links: dict[tuple[int, int], Link]
    default_link_config: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Topology:
        devices: dict[int, Device] = {}
        memories: dict[int, MemoryResource] = {}
        for item in config.get("devices", []):
            item_id = int(item["id"])
            if str(item.get("class", "compute")) == "memory":
                if item_id in memories:
                    raise ValueError(f"duplicate memory id {item_id}")
                memories[item_id] = MemoryResource(
                    memory_id=item_id,
                    capacity_bytes=int(item.get("capacity_bytes", 0)),
                    bandwidth_bytes_per_sec=bandwidth_bytes_per_second(item),
                    fixed_latency=seconds(item),
                )
            else:
                if item_id in devices:
                    raise ValueError(f"duplicate compute device id {item_id}")
                devices[item_id] = Device(
                    device_id=item_id,
                    device_class=str(item.get("class", "compute")),
                    fixed_latency=seconds(item),
                    prefill=_prefill_settings(item),
                    decode=_decode_settings(item),
                )
        for item in config.get("memories", []):
            item_id = int(item["id"])
            if item_id in memories:
                raise ValueError(f"duplicate memory id {item_id}")
            memories[item_id] = MemoryResource(
                memory_id=item_id,
                capacity_bytes=int(item.get("capacity_bytes", 0)),
                bandwidth_bytes_per_sec=bandwidth_bytes_per_second(item),
                fixed_latency=seconds(item),
            )

        links: dict[tuple[int, int], Link] = {}
        for item in config.get("links", []):
            key = (int(item["src"]), int(item["dst"]))
            if key in links:
                raise ValueError(f"duplicate link {key}")
            links[key] = Link(
                src=key[0],
                dst=key[1],
                bandwidth_bytes_per_sec=bandwidth_bytes_per_second(item),
                fixed_latency=seconds(item),
            )
        default_link = config.get("default_all_to_all_link")
        return cls(devices, memories, links, dict(default_link) if default_link else None)

    def get_link(self, src: int, dst: int) -> Link:
        key = (src, dst)
        if key not in self.links and self.default_link_config is not None:
            self.links[key] = Link(
                src=src,
                dst=dst,
                bandwidth_bytes_per_sec=bandwidth_bytes_per_second(self.default_link_config),
                fixed_latency=seconds(self.default_link_config),
            )
        try:
            return self.links[key]
        except KeyError as error:
            raise KeyError(f"no direct link from {src} to {dst}") from error
