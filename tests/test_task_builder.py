from agentic_sim.core.task import ComputePhase, TaskType
from agentic_sim.runtime.task_builder import TaskBuilder
from agentic_sim.workload.invocation import Invocation


def invocation(context_tokens: int = 0) -> Invocation:
    return Invocation(
        invocation_id=0,
        agent_id=0,
        turn_index=int(context_tokens > 0),
        input_tokens=2,
        output_tokens=1,
        context_tokens=context_tokens,
        ready_time=0.0,
    )


def test_first_retained_turn_has_only_pd_core_tasks() -> None:
    tasks = TaskBuilder({"kv": {"bytes_per_token": 10, "retention": "memory"}}).build(
        invocation()
    )
    assert [task.task_type for task in tasks] == [
        TaskType.COMPUTE,
        TaskType.TRANSFER,
        TaskType.COMPUTE,
    ]
    assert tasks[0].metadata == {"phase": ComputePhase.PREFILL, "tokens": 2}
    assert tasks[1].metadata == {"bytes": 20, "direction": "prefill_to_decode"}
    assert tasks[2].metadata == {"phase": ComputePhase.DECODE, "tokens": 1}


def test_retained_later_turn_restores_kv_and_prefills_only_new_input() -> None:
    tasks = TaskBuilder({"kv": {"bytes_per_token": 10, "retention": "far_memory"}}).build(
        invocation(context_tokens=3)
    )
    assert [task.task_type for task in tasks] == [
        TaskType.MEMORY,
        TaskType.TRANSFER,
        TaskType.COMPUTE,
        TaskType.TRANSFER,
        TaskType.COMPUTE,
    ]
    assert tasks[0].metadata == {"bytes": 30, "operation": "read"}
    assert tasks[1].metadata == {"bytes": 30, "direction": "restore"}
    assert tasks[2].metadata == {"phase": ComputePhase.PREFILL, "tokens": 2}
    assert tasks[3].metadata == {"bytes": 50, "direction": "prefill_to_decode"}
    assert tasks[4].metadata == {"phase": ComputePhase.DECODE, "tokens": 1}


def test_discard_recomputes_context_without_memory_tasks() -> None:
    builder = TaskBuilder({"kv": {"bytes_per_token": 10, "retention": "discard"}})
    tasks = builder.build(invocation(context_tokens=3))
    assert [task.task_type for task in tasks] == [
        TaskType.COMPUTE,
        TaskType.TRANSFER,
        TaskType.COMPUTE,
    ]
    assert tasks[0].metadata == {"phase": ComputePhase.PREFILL, "tokens": 5}
    assert builder.build_store(0, 0, 6) == []


def test_store_contains_background_transfer_and_write() -> None:
    builder = TaskBuilder({"kv": {"bytes_per_token": 10, "retention": "memory"}})
    tasks = builder.build_store(agent_id=0, invocation_id=0, context_tokens=3)
    assert [task.task_type for task in tasks] == [TaskType.TRANSFER, TaskType.MEMORY]
    assert tasks[0].metadata == {
        "bytes": 30,
        "direction": "store",
        "background": True,
    }
    assert tasks[1].metadata == {"bytes": 30, "operation": "write", "background": True}
    assert tasks[1].dependencies == [tasks[0].task_id]
