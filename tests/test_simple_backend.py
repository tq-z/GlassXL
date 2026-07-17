import pytest

from agentic_sim.core.task import ComputePhase, ComputeTask, MemoryTask, TransferTask
from agentic_sim.hardware.simple_backend import SimpleHardwareBackend
from agentic_sim.hardware.topology import Topology
from agentic_sim.runtime.reference_policy import Placement


def test_simple_backend_phase_and_bandwidth_equations(minimal_config: dict) -> None:
    backend = SimpleHardwareBackend(Topology.from_config(minimal_config["hardware"]))
    prefill = ComputeTask(
        task_id=0,
        agent_id=0,
        invocation_id=0,
        metadata={"phase": ComputePhase.PREFILL, "tokens": 10},
    )
    decode = ComputeTask(
        task_id=1,
        agent_id=0,
        invocation_id=0,
        metadata={"phase": ComputePhase.DECODE, "tokens": 10},
    )
    memory = MemoryTask(task_id=2, agent_id=0, invocation_id=0, metadata={"bytes": 100})
    transfer = TransferTask(task_id=3, agent_id=0, invocation_id=0, metadata={"bytes": 100})
    assert backend.submit(prefill, Placement(0), 0.0).reservation.service_time == pytest.approx(1.01)
    assert backend.submit(decode, Placement(1), 0.0).reservation.service_time == pytest.approx(2.01)
    assert backend.submit(memory, Placement(2), 0.0).reservation.service_time == pytest.approx(1.02)
    assert backend.submit(transfer, Placement((0, 1)), 0.0).reservation.service_time == pytest.approx(1.03)


def test_default_all_to_all_link_is_created_lazily(minimal_config: dict) -> None:
    config = dict(minimal_config["hardware"])
    config["links"] = []
    config["default_all_to_all_link"] = {
        "bandwidth_bytes_per_sec": 100,
        "fixed_latency_s": 0.1,
    }
    topology = Topology.from_config(config)
    assert topology.get_link(1, 0).service_time(10) == pytest.approx(0.2)
