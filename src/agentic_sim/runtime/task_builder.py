"""Map invocations to hardware-independent coarse P-D task graphs."""

from __future__ import annotations

from typing import Any

from agentic_sim.core.task import ComputePhase, ComputeTask, MemoryTask, Task, TransferTask
from agentic_sim.workload.invocation import Invocation, KVRetentionMode


class TaskBuilder:
    def __init__(self, config: dict[str, Any]) -> None:
        kv_config = config["kv"]
        self.kv_bytes_per_token = int(kv_config["bytes_per_token"])
        if self.kv_bytes_per_token < 0:
            raise ValueError("kv.bytes_per_token must be non-negative")
        self.retention = KVRetentionMode(str(kv_config["retention"]))
        self._next_task_id = 0

    @property
    def retains_kv(self) -> bool:
        return self.retention is not KVRetentionMode.DISCARD

    def build(self, invocation: Invocation) -> list[Task]:
        tasks: list[Task] = []
        dependency: list[int] = []
        if self.retains_kv and invocation.context_tokens > 0:
            restore_bytes = invocation.context_tokens * self.kv_bytes_per_token
            memory_read = MemoryTask(
                task_id=self._take_task_id(),
                agent_id=invocation.agent_id,
                invocation_id=invocation.invocation_id,
                metadata={"bytes": restore_bytes, "operation": "read"},
            )
            restore = TransferTask(
                task_id=self._take_task_id(),
                agent_id=invocation.agent_id,
                invocation_id=invocation.invocation_id,
                dependencies=[memory_read.task_id],
                metadata={"bytes": restore_bytes, "direction": "restore"},
            )
            tasks.extend((memory_read, restore))
            dependency = [restore.task_id]

        prefill_tokens = (
            invocation.input_tokens
            if self.retains_kv
            else invocation.context_tokens + invocation.input_tokens
        )
        prefill = ComputeTask(
            task_id=self._take_task_id(),
            agent_id=invocation.agent_id,
            invocation_id=invocation.invocation_id,
            dependencies=dependency,
            metadata={"phase": ComputePhase.PREFILL, "tokens": prefill_tokens},
        )
        prompt_tokens = invocation.context_tokens + invocation.input_tokens
        pd_transfer = TransferTask(
            task_id=self._take_task_id(),
            agent_id=invocation.agent_id,
            invocation_id=invocation.invocation_id,
            dependencies=[prefill.task_id],
            metadata={
                "bytes": prompt_tokens * self.kv_bytes_per_token,
                "direction": "prefill_to_decode",
            },
        )
        decode = ComputeTask(
            task_id=self._take_task_id(),
            agent_id=invocation.agent_id,
            invocation_id=invocation.invocation_id,
            dependencies=[pd_transfer.task_id],
            metadata={"phase": ComputePhase.DECODE, "tokens": invocation.output_tokens},
        )
        tasks.extend((prefill, pd_transfer, decode))
        return tasks

    def build_store(
        self,
        agent_id: int,
        invocation_id: int,
        context_tokens: int,
    ) -> list[Task]:
        if not self.retains_kv:
            return []
        byte_count = context_tokens * self.kv_bytes_per_token
        transfer = TransferTask(
            task_id=self._take_task_id(),
            agent_id=agent_id,
            invocation_id=invocation_id,
            metadata={"bytes": byte_count, "direction": "store", "background": True},
        )
        write = MemoryTask(
            task_id=self._take_task_id(),
            agent_id=agent_id,
            invocation_id=invocation_id,
            dependencies=[transfer.task_id],
            metadata={"bytes": byte_count, "operation": "write", "background": True},
        )
        return [transfer, write]

    def _take_task_id(self) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id
