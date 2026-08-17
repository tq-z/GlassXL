"""Event-driven Phase 1.5 runtime with batching and tool/KV barriers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentic_sim.core.event import Event, EventType
from agentic_sim.core.event_engine import EventEngine
from agentic_sim.core.task import Task
from agentic_sim.hardware.hardware import HardwareState
from agentic_sim.hardware.simple_backend import SimpleHardwareBackend, TaskExecution
from agentic_sim.hardware.topology import seconds
from agentic_sim.metrics.batching_metrics import IngressFlowMetric
from agentic_sim.metrics.collector import MetricsCollector
from agentic_sim.runtime.decode_batcher import DecodeBatchController, DecodeStepCompletion
from agentic_sim.runtime.kv_ingress import KVIngressController
from agentic_sim.runtime.phase_state import InvocationPhase, InvocationSession
from agentic_sim.runtime.prefill_batcher import PrefillBatchCompletion, PrefillBatchController
from agentic_sim.runtime.reference_policy import StaticFCFSPolicy
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
        policy: StaticFCFSPolicy,
        task_builder: TaskBuilder,
        metrics: MetricsCollector,
        kv_ingress_config: dict[str, Any] | None = None,
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
        self._next_legacy_flow_id = 0
        self._legacy_ingress_active = 0
        self._tasks: dict[int, Task] = {}
        self._children: dict[int, list[int]] = {}
        self._finished_tasks: set[int] = set()
        self._scheduled_tasks: set[int] = set()
        self._task_graph: dict[int, int] = {}
        self._remaining_by_graph: dict[int, int] = {}
        self._graph_callbacks: dict[int, Callable[[], None]] = {}
        self._turn_barriers: dict[int, TurnBarrier] = {}
        self.sessions: dict[int, InvocationSession] = {}

        prefill_device = backend.topology.devices[policy.prefill_device]
        decode_device = backend.topology.devices[policy.decode_device]
        if prefill_device.prefill is None:
            raise ValueError("selected prefill device has no prefill settings")
        if decode_device.decode is None:
            raise ValueError("selected decode device has no decode settings")
        self.prefill_controller = PrefillBatchController(
            engine,
            prefill_device,
            prefill_device.prefill,
            metrics,
            self._on_prefill_started,
            self._on_prefill_complete,
        )
        self.decode_controller = DecodeBatchController(
            engine,
            decode_device,
            decode_device.decode,
            metrics,
            self._on_decode_admitted,
            self._on_decode_token,
            self._on_decode_complete,
        )
        self.ingress_controller: KVIngressController | None = None
        if kv_ingress_config is not None:
            self.ingress_controller = KVIngressController(
                engine=engine,
                fixed_latency=seconds(kv_ingress_config),
                per_session_max_bandwidth=float(kv_ingress_config["per_session_max_bandwidth_bytes_per_sec"]),
                max_system_bandwidth=float(kv_ingress_config["max_system_bandwidth_bytes_per_sec"]),
                metrics=metrics,
                on_session_complete=self._on_shared_ingress_complete,
            )
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
            EventType.PREFILL_BATCH_START: self._on_prefill_batch_start,
            EventType.PREFILL_BATCH_FINISH: self._on_prefill_batch_finish,
            EventType.KV_INGRESS_DATA_READY: self._on_ingress_data_ready,
            EventType.KV_INGRESS_FINISH: self._on_ingress_finish,
            EventType.DECODE_ITERATION_START: self._on_decode_iteration_start,
            EventType.DECODE_STEP_FINISH: self._on_decode_step_finish,
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
            input_tokens=agent.current_input_tokens,
            output_tokens=agent.output_tokens_per_turn[agent.current_turn],
            context_tokens=agent.current_context_tokens,
            ready_time=self.engine.now,
        )
        self._next_invocation_id += 1
        self.metrics.invocation_ready(invocation)
        restore_tasks = self.task_builder.build_restore(invocation)
        session = InvocationSession(
            invocation=invocation,
            effective_prefill_tokens=self.task_builder.effective_prefill_tokens(invocation),
            prompt_kv_bytes=self.task_builder.prompt_kv_bytes(invocation),
            remaining_output_tokens=invocation.output_tokens,
            phase=InvocationPhase.WAITING_RESTORE if restore_tasks else InvocationPhase.WAITING_PREFILL,
        )
        self.sessions[session.session_id] = session
        if restore_tasks:
            self._schedule_graph(restore_tasks, lambda current=session: self._enqueue_prefill(current))
        else:
            self._enqueue_prefill(session)

    def _enqueue_prefill(self, session: InvocationSession) -> None:
        self.metrics.invocation_prefill_queued(session.session_id, self.engine.now)
        self.prefill_controller.enqueue(session)

    def _on_prefill_batch_start(self, event: Event) -> None:
        self.prefill_controller.handle_start()

    def _on_prefill_batch_finish(self, event: Event) -> None:
        if not isinstance(event.payload, PrefillBatchCompletion):
            raise TypeError("PREFILL_BATCH_FINISH payload must be a PrefillBatchCompletion")
        self.prefill_controller.handle_finish(event.payload)

    def _on_prefill_started(self, session: InvocationSession, queue_delay: float, service_time: float) -> None:
        agent = self.agents[session.agent_id]
        if agent.start_time is None:
            agent.start_time = self.engine.now
        agent.queueing_time += queue_delay
        agent.active_execution_time += service_time
        self.metrics.invocation_prefill_started(session.session_id, self.engine.now, queue_delay, service_time)

    def _on_prefill_complete(self, session: InvocationSession) -> None:
        self.metrics.invocation_prefill_finished(session.session_id, self.engine.now)
        self.metrics.invocation_ingress_started(session.session_id, self.engine.now)
        if self.ingress_controller is not None:
            self.ingress_controller.submit(session)
            return
        self._submit_legacy_ingress(session)

    def _submit_legacy_ingress(self, session: InvocationSession) -> None:
        session.phase = InvocationPhase.KV_INGRESS_RUNNING
        session.ingress_submitted_at = self.engine.now
        flow_id = self._next_legacy_flow_id
        self._next_legacy_flow_id += 1
        self.metrics.ingress_flow_submitted(flow_id, session.session_id, session.prompt_kv_bytes)
        self._legacy_ingress_active += 1
        self.metrics.ingress_active_changed(self.engine.now, self._legacy_ingress_active, 0.0)
        submitted = self.engine.now

        def complete() -> None:
            duration = self.engine.now - submitted
            achieved = session.prompt_kv_bytes / duration if duration > 0.0 else 0.0
            session.ingress_finished_at = self.engine.now
            self._legacy_ingress_active -= 1
            self.metrics.ingress_active_changed(self.engine.now, self._legacy_ingress_active, 0.0)
            self.metrics.ingress_flow_finished(
                IngressFlowMetric(flow_id, session.session_id, submitted, submitted, self.engine.now, session.prompt_kv_bytes, achieved)
            )
            self.metrics.invocation_ingress_finished(session.session_id, self.engine.now, 0.0)
            self._enqueue_decode(session)

        self._schedule_graph(self.task_builder.build_ingress(session.invocation), complete)

    def _on_ingress_data_ready(self, event: Event) -> None:
        if self.ingress_controller is None:
            raise RuntimeError("received a shared-ingress event without a configured pool")
        self.ingress_controller.handle_data_ready(int(event.payload))

    def _on_ingress_finish(self, event: Event) -> None:
        if self.ingress_controller is None:
            raise RuntimeError("received a shared-ingress event without a configured pool")
        self.ingress_controller.handle_finish(event.payload)

    def _on_shared_ingress_complete(self, session: InvocationSession, duration: float) -> None:
        agent = self.agents[session.agent_id]
        agent.active_execution_time += duration
        self.metrics.invocation_ingress_finished(session.session_id, self.engine.now, duration)
        self._enqueue_decode(session)

    def _enqueue_decode(self, session: InvocationSession) -> None:
        self.metrics.invocation_decode_queued(session.session_id, self.engine.now)
        self.decode_controller.enqueue(session)

    def _on_decode_iteration_start(self, event: Event) -> None:
        self.decode_controller.handle_start()

    def _on_decode_step_finish(self, event: Event) -> None:
        if not isinstance(event.payload, DecodeStepCompletion):
            raise TypeError("DECODE_STEP_FINISH payload must be a DecodeStepCompletion")
        self.decode_controller.handle_finish(event.payload)

    def _on_decode_admitted(self, session: InvocationSession, queue_delay: float) -> None:
        agent = self.agents[session.agent_id]
        agent.queueing_time += queue_delay
        self.metrics.invocation_decode_admitted(session.session_id, self.engine.now, queue_delay)

    def _on_decode_token(self, session: InvocationSession, time: float, service_time: float) -> None:
        self.agents[session.agent_id].active_execution_time += service_time
        self.metrics.invocation_decode_token(session.session_id, time, service_time)

    def _on_decode_complete(self, session: InvocationSession) -> None:
        agent = self.agents[session.agent_id]
        agent.current_context_tokens = session.final_context_tokens
        self.metrics.invocation_finished(session.session_id, self.engine.now)
        if agent.is_last_turn:
            self.engine.schedule(self.engine.now, EventType.AGENT_FINISH, agent.agent_id)
            return
        self._turn_barriers[agent.agent_id] = TurnBarrier(False, not self.task_builder.uses_memory_tier)
        self.engine.schedule(self.engine.now, EventType.TOOL_WAIT_START, agent.agent_id)
        if self.task_builder.uses_memory_tier:
            store_tasks = self.task_builder.build_store(agent.agent_id, session.session_id, agent.current_context_tokens)
            self._schedule_graph(store_tasks, lambda agent_id=agent.agent_id: self._on_kv_store_complete(agent_id))

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
