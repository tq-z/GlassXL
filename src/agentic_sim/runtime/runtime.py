"""Event-driven agentic runtime with tool/KV fork-join barriers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agentic_sim.core.event import Event, EventType
from agentic_sim.core.event_engine import EventEngine
from agentic_sim.core.task import Task
from agentic_sim.hardware.hardware import HardwareState
from agentic_sim.hardware.simple_backend import SimpleHardwareBackend, TaskExecution
from agentic_sim.metrics.collector import MetricsCollector
from agentic_sim.runtime.reference_policy import ExecutionPolicy
from agentic_sim.runtime.task_builder import TaskBuilder
from agentic_sim.workload.agent import Agent, AgentState
from agentic_sim.workload.invocation import Invocation


@dataclass
class TurnBarrier:
    tool_complete: bool
    kv_complete: bool


class AgenticRuntime:
    def __init__(
        self,
        engine: EventEngine,
        agents: list[Agent],
        backend: SimpleHardwareBackend,
        policy: ExecutionPolicy,
        task_builder: TaskBuilder,
        metrics: MetricsCollector,
    ) -> None:
        self.engine = engine
        self.agents = {agent.agent_id: agent for agent in agents}
        self.backend = backend
        self.policy = policy
        self.task_builder = task_builder
        self.metrics = metrics
        self.hardware_state = HardwareState(backend.topology)
        self._next_invocation_id = 0
        self._next_graph_id = 0
        self._tasks: dict[int, Task] = {}
        self._children: dict[int, list[int]] = {}
        self._finished_tasks: set[int] = set()
        self._scheduled_tasks: set[int] = set()
        self._task_graph: dict[int, int] = {}
        self._remaining_by_graph: dict[int, int] = {}
        self._graph_callbacks: dict[int, Callable[[], None]] = {}
        self._turn_barriers: dict[int, TurnBarrier] = {}
        self._register_handlers()

    def schedule_arrivals(self) -> None:
        for agent in self.agents.values():
            self.engine.schedule(agent.arrival_time, EventType.AGENT_ARRIVAL, agent.agent_id)

    def _register_handlers(self) -> None:
        handlers = {
            EventType.AGENT_ARRIVAL: self._on_agent_arrival,
            EventType.INVOCATION_READY: self._on_invocation_ready,
            EventType.TASK_READY: self._on_task_ready,
            EventType.TASK_START: self._on_task_start,
            EventType.TASK_FINISH: self._on_task_finish,
            EventType.TOOL_WAIT_START: self._on_tool_wait_start,
            EventType.TOOL_RETURN: self._on_tool_return,
            EventType.AGENT_FINISH: self._on_agent_finish,
        }
        for event_type, handler in handlers.items():
            self.engine.register_handler(event_type, handler)

    def _transition(self, agent: Agent, new_state: AgentState) -> None:
        self.metrics.state_transition(self.engine.now, agent.state, new_state)
        agent.state = new_state

    def _on_agent_arrival(self, event: Event) -> None:
        agent = self.agents[int(event.payload)]
        if agent.state is not AgentState.NOT_ARRIVED:
            raise RuntimeError("agent arrived more than once")
        self._transition(agent, AgentState.READY)
        self.engine.schedule(self.engine.now, EventType.INVOCATION_READY, agent.agent_id)

    def _on_invocation_ready(self, event: Event) -> None:
        agent = self.agents[int(event.payload)]
        if agent.state is not AgentState.READY:
            raise RuntimeError("invocation became ready from an invalid agent state")
        self._transition(agent, AgentState.RUNNING)
        invocation = Invocation(
            invocation_id=self._next_invocation_id,
            agent_id=agent.agent_id,
            turn_index=agent.current_turn,
            input_tokens=agent.input_tokens_per_turn[agent.current_turn],
            output_tokens=agent.output_tokens_per_turn[agent.current_turn],
            context_tokens=agent.current_context_tokens,
            ready_time=self.engine.now,
        )
        self._next_invocation_id += 1
        self.metrics.invocation_ready(invocation)
        self._schedule_graph(
            self.task_builder.build(invocation),
            lambda agent_id=agent.agent_id, invocation_id=invocation.invocation_id: (
                self._complete_invocation(agent_id, invocation_id)
            ),
        )

    def _schedule_graph(self, tasks: list[Task], on_complete: Callable[[], None]) -> None:
        if not tasks:
            on_complete()
            return
        graph_id = self._next_graph_id
        self._next_graph_id += 1
        self._remaining_by_graph[graph_id] = len(tasks)
        self._graph_callbacks[graph_id] = on_complete
        for task in tasks:
            self._tasks[task.task_id] = task
            self._task_graph[task.task_id] = graph_id
            for dependency in task.dependencies:
                self._children.setdefault(dependency, []).append(task.task_id)
        for task in tasks:
            if not task.dependencies:
                self._schedule_task_ready(task.task_id)

    def _schedule_task_ready(self, task_id: int) -> None:
        if task_id not in self._scheduled_tasks:
            self._scheduled_tasks.add(task_id)
            self.engine.schedule(self.engine.now, EventType.TASK_READY, task_id)

    def _on_task_ready(self, event: Event) -> None:
        task = self._tasks[int(event.payload)]
        if not set(task.dependencies).issubset(self._finished_tasks):
            raise RuntimeError(f"task {task.task_id} became ready before its dependencies")
        task.ready_time = self.engine.now
        placement = self.policy.place_task(task, self.hardware_state)
        execution = self.backend.submit(task, placement, self.engine.now)
        agent = self.agents[task.agent_id]
        agent.queueing_time += execution.reservation.queue_delay
        agent.active_execution_time += execution.reservation.service_time
        self.metrics.task_reserved(execution)
        self.engine.schedule(execution.reservation.start_time, EventType.TASK_START, execution)
        self.engine.schedule(execution.reservation.finish_time, EventType.TASK_FINISH, execution)

    def _on_task_start(self, event: Event) -> None:
        execution = event.payload
        if not isinstance(execution, TaskExecution):
            raise TypeError("TASK_START payload must be a TaskExecution")
        agent = self.agents[execution.task.agent_id]
        if agent.start_time is None:
            agent.start_time = self.engine.now

    def _on_task_finish(self, event: Event) -> None:
        execution = event.payload
        if not isinstance(execution, TaskExecution):
            raise TypeError("TASK_FINISH payload must be a TaskExecution")
        task = execution.task
        self._finished_tasks.add(task.task_id)
        graph_id = self._task_graph[task.task_id]
        self._remaining_by_graph[graph_id] -= 1
        for child_id in self._children.get(task.task_id, []):
            child = self._tasks[child_id]
            if set(child.dependencies).issubset(self._finished_tasks):
                self._schedule_task_ready(child_id)
        if self._remaining_by_graph[graph_id] == 0:
            del self._remaining_by_graph[graph_id]
            callback = self._graph_callbacks.pop(graph_id)
            callback()

    def _complete_invocation(self, agent_id: int, invocation_id: int) -> None:
        agent = self.agents[agent_id]
        agent.complete_current_turn()
        self.metrics.invocation_finished(invocation_id, self.engine.now)
        if agent.is_last_turn:
            self.engine.schedule(self.engine.now, EventType.AGENT_FINISH, agent.agent_id)
            return

        self._turn_barriers[agent_id] = TurnBarrier(
            tool_complete=False,
            kv_complete=not self.task_builder.retains_kv,
        )
        self.engine.schedule(self.engine.now, EventType.TOOL_WAIT_START, agent.agent_id)
        if self.task_builder.retains_kv:
            store_tasks = self.task_builder.build_store(
                agent_id=agent.agent_id,
                invocation_id=invocation_id,
                context_tokens=agent.current_context_tokens,
            )
            self._schedule_graph(
                store_tasks,
                lambda current_agent_id=agent_id: self._on_kv_store_complete(current_agent_id),
            )

    def _on_tool_wait_start(self, event: Event) -> None:
        agent = self.agents[int(event.payload)]
        wait_time = agent.tool_wait_per_turn[agent.current_turn]
        agent.actual_tool_wait_time += wait_time
        agent.current_turn += 1
        self._transition(agent, AgentState.TOOL_WAIT)
        self.engine.schedule(self.engine.now + wait_time, EventType.TOOL_RETURN, agent.agent_id)

    def _on_tool_return(self, event: Event) -> None:
        agent = self.agents[int(event.payload)]
        barrier = self._turn_barriers[agent.agent_id]
        barrier.tool_complete = True
        if barrier.kv_complete:
            self._release_turn_barrier(agent)
        else:
            self._transition(agent, AgentState.KV_WAIT)

    def _on_kv_store_complete(self, agent_id: int) -> None:
        agent = self.agents[agent_id]
        barrier = self._turn_barriers[agent_id]
        barrier.kv_complete = True
        if barrier.tool_complete:
            self._release_turn_barrier(agent)

    def _release_turn_barrier(self, agent: Agent) -> None:
        barrier = self._turn_barriers[agent.agent_id]
        if not barrier.tool_complete or not barrier.kv_complete:
            raise RuntimeError("attempted to release an incomplete turn barrier")
        del self._turn_barriers[agent.agent_id]
        self._transition(agent, AgentState.READY)
        self.engine.schedule(self.engine.now, EventType.INVOCATION_READY, agent.agent_id)

    def _on_agent_finish(self, event: Event) -> None:
        agent = self.agents[int(event.payload)]
        agent.finish_time = self.engine.now
        self._transition(agent, AgentState.FINISHED)
