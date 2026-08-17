"""Typed assembly of V1 devices, interconnects, and KV pool."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_sim.config.models import HardwareConfig
from agentic_sim.hardware.device import Device
from agentic_sim.hardware.interconnect import (
    FFNAllToAllInterconnect,
    InterconnectReservation,
    KVP2PInterconnect,
)
from agentic_sim.hardware.kv_pool import KVPool, KVPoolReservation


@dataclass
class Topology:
    """All physical and logical hardware resources for one simulation."""

    devices: dict[int, Device]
    ffn_all_to_all: FFNAllToAllInterconnect
    kv_p2p: KVP2PInterconnect
    kv_pool: KVPool

    @classmethod
    def from_config(cls, config: HardwareConfig) -> Topology:
        devices = {item.id: Device.from_config(item) for item in config.devices}
        if len(devices) != len(config.devices):
            raise ValueError("hardware configuration contains duplicate device IDs")
        return cls(
            devices=devices,
            ffn_all_to_all=FFNAllToAllInterconnect.from_config(
                config.interconnects.ffn_all_to_all
            ),
            kv_p2p=KVP2PInterconnect.from_config(config.interconnects.kv_p2p),
            kv_pool=KVPool.from_config(config.kv_pool),
        )

    def get_device(self, device_id: int) -> Device:
        try:
            return self.devices[device_id]
        except KeyError as error:
            raise KeyError(f"unknown physical device {device_id}") from error

    def submit_all_to_all(
        self,
        send_tokens: int,
        participant_ids: tuple[int, ...],
        ready_time: float,
    ) -> InterconnectReservation:
        participants = tuple(self.get_device(device_id) for device_id in participant_ids)
        return self.ffn_all_to_all.submit(send_tokens, participants, ready_time)

    def submit_kv_p2p(
        self,
        tokens: int,
        src_id: int,
        dst_id: int,
        ready_time: float,
    ) -> InterconnectReservation:
        return self.kv_p2p.submit(
            tokens,
            self.get_device(src_id),
            self.get_device(dst_id),
            ready_time,
        )

    def submit_pool_load(
        self,
        tokens: int,
        device_id: int,
        ready_time: float,
    ) -> KVPoolReservation:
        return self.kv_pool.submit_load(tokens, self.get_device(device_id), ready_time)

    def submit_pool_offload(
        self,
        tokens: int,
        device_id: int,
        ready_time: float,
    ) -> KVPoolReservation:
        return self.kv_pool.submit_offload(tokens, self.get_device(device_id), ready_time)
