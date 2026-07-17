"""Hardware-independent task abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TaskType(Enum):
    COMPUTE = auto()
    MEMORY = auto()
    TRANSFER = auto()
    WAIT = auto()


class ComputePhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class Task:
    task_id: int
    task_type: TaskType
    agent_id: int
    invocation_id: int
    ready_time: float = 0.0
    dependencies: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ComputeTask(Task):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, task_type=TaskType.COMPUTE, **kwargs)


class MemoryTask(Task):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, task_type=TaskType.MEMORY, **kwargs)


class TransferTask(Task):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, task_type=TaskType.TRANSFER, **kwargs)


class WaitTask(Task):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, task_type=TaskType.WAIT, **kwargs)
