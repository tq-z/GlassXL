import pytest

from agentic_sim.core.resource import FCFSResource


def test_fcfs_resource_serializes_and_measures_queueing() -> None:
    resource = FCFSResource("compute:0")
    first = resource.reserve(0.0, 2.0)
    second = resource.reserve(0.5, 1.0)
    third = resource.reserve(0.75, 1.0)
    assert (first.start_time, first.finish_time) == (0.0, 2.0)
    assert (second.start_time, second.finish_time) == (2.0, 3.0)
    assert third.start_time == 3.0
    assert second.queue_delay == 1.5
    assert resource.max_queue_length >= 2
    assert resource.busy_time(2.5) == 2.5
    assert resource.utilization(4.0) == pytest.approx(1.0)
