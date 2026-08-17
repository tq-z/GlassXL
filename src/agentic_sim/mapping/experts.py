"""Deterministic equal-route expert loads and configured ownership."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_sim.config.models import DecodeFFNMappingConfig
from agentic_sim.hardware.compute_models import ExpertWork


def _nonnegative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True)
class ExpertDeviceBatch:
    """All configured expert loads submitted atomically to one EP device."""

    device_id: int
    experts: tuple[ExpertWork, ...]

    @property
    def activation_tokens(self) -> int:
        return sum(work.activation_tokens for work in self.experts)


@dataclass(frozen=True)
class ExpertMapping:
    """One non-replicated expert ownership map shared by every V1 decode step."""

    device_ids: tuple[int, ...]
    owner_by_expert: tuple[int, ...]

    @classmethod
    def from_config(
        cls,
        config: DecodeFFNMappingConfig,
        num_experts: int,
    ) -> ExpertMapping:
        _nonnegative_integer(num_experts, "num_experts")
        if num_experts == 0:
            raise ValueError("num_experts must be positive")
        if not config.device_ids or len(set(config.device_ids)) != len(config.device_ids):
            raise ValueError("FFN mapping device IDs must be non-empty and unique")
        placements = {item.device_id: item.expert_ids for item in config.experts_by_device}
        if set(placements) != set(config.device_ids):
            raise ValueError("expert placements must exactly match FFN device IDs")

        owners: list[int | None] = [None] * num_experts
        for device_id in config.device_ids:
            expert_ids = placements[device_id]
            if not expert_ids:
                raise ValueError(f"FFN device {device_id} must own at least one expert")
            for expert_id in expert_ids:
                _nonnegative_integer(expert_id, "expert_id")
                if expert_id >= num_experts:
                    raise ValueError(f"expert {expert_id} is outside [0, {num_experts})")
                if owners[expert_id] is not None:
                    raise ValueError(f"expert {expert_id} has replicated ownership")
                owners[expert_id] = device_id
        missing = tuple(index for index, owner in enumerate(owners) if owner is None)
        if missing:
            raise ValueError(f"experts have no configured owner: {missing}")
        return cls(config.device_ids, tuple(owner for owner in owners if owner is not None))

    @property
    def num_experts(self) -> int:
        return len(self.owner_by_expert)

    def equal_loads(self, activation_tokens: int) -> tuple[ExpertWork, ...]:
        """Conserve A exactly using deterministic quotient/remainder assignment."""

        _nonnegative_integer(activation_tokens, "activation_tokens")
        quotient, remainder = divmod(activation_tokens, self.num_experts)
        loads = tuple(
            ExpertWork(
                expert_id=expert_id,
                activation_tokens=quotient + (expert_id < remainder),
                device_id=device_id,
            )
            for expert_id, device_id in enumerate(self.owner_by_expert)
        )
        if sum(work.activation_tokens for work in loads) != activation_tokens:
            raise RuntimeError("equal expert routing failed activation conservation")
        return loads

    def device_batches(self, activation_tokens: int) -> tuple[ExpertDeviceBatch, ...]:
        loads = self.equal_loads(activation_tokens)
        return tuple(
            ExpertDeviceBatch(
                device_id=device_id,
                experts=tuple(work for work in loads if work.device_id == device_id),
            )
            for device_id in self.device_ids
        )
