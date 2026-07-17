from agentic_sim.core.event import EventType
from agentic_sim.core.event_engine import EventEngine


def test_event_engine_orders_by_time_then_sequence() -> None:
    engine = EventEngine()
    seen: list[str] = []
    engine.register_handler(EventType.TASK_READY, lambda event: seen.append(event.payload))
    engine.schedule(2.0, EventType.TASK_READY, "late")
    engine.schedule(1.0, EventType.TASK_READY, "first-tie")
    engine.schedule(1.0, EventType.TASK_READY, "second-tie")
    engine.run()
    assert seen == ["first-tie", "second-tie", "late"]
    assert engine.now == 2.0


def test_event_engine_stops_before_event_past_horizon() -> None:
    engine = EventEngine(end_time=1.5)
    seen: list[float] = []
    engine.register_handler(EventType.TASK_READY, lambda event: seen.append(event.time))
    engine.schedule(1.0, EventType.TASK_READY)
    engine.schedule(2.0, EventType.TASK_READY)
    engine.run()
    assert seen == [1.0]
    assert engine.now == 1.5
    assert engine.pending_events == 1
    assert engine.stopped_at_end_time


def test_event_engine_rejects_events_in_the_past() -> None:
    engine = EventEngine()
    engine.register_handler(EventType.TASK_READY, lambda event: None)
    engine.schedule(1.0, EventType.TASK_READY)
    engine.run()
    try:
        engine.schedule(0.5, EventType.TASK_READY)
    except ValueError:
        pass
    else:
        raise AssertionError("scheduling in the past should fail")
