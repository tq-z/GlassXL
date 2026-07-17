from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def minimal_config() -> dict[str, Any]:
    return {
        "seed": 11,
        "hardware": {
            "devices": [
                {
                    "id": 0,
                    "class": "compute",
                    "prefill_tokens_per_second": 10.0,
                    "decode_tokens_per_second": 5.0,
                    "fixed_latency_s": 0.01,
                },
                {
                    "id": 1,
                    "class": "compute",
                    "prefill_tokens_per_second": 8.0,
                    "decode_tokens_per_second": 5.0,
                    "fixed_latency_s": 0.01,
                },
            ],
            "memories": [
                {
                    "id": 2,
                    "capacity_bytes": 10_000,
                    "bandwidth_bytes_per_sec": 100.0,
                    "fixed_latency_s": 0.02,
                },
                {
                    "id": 3,
                    "capacity_bytes": 100_000,
                    "bandwidth_bytes_per_sec": 20.0,
                    "fixed_latency_s": 0.05,
                },
            ],
            "links": [
                {"src": 0, "dst": 1, "bandwidth_bytes_per_sec": 100.0, "fixed_latency_s": 0.03},
                {"src": 1, "dst": 2, "bandwidth_bytes_per_sec": 100.0, "fixed_latency_s": 0.01},
                {"src": 2, "dst": 0, "bandwidth_bytes_per_sec": 100.0, "fixed_latency_s": 0.01},
                {"src": 1, "dst": 3, "bandwidth_bytes_per_sec": 20.0, "fixed_latency_s": 0.05},
                {"src": 3, "dst": 0, "bandwidth_bytes_per_sec": 20.0, "fixed_latency_s": 0.05},
            ],
        },
        "workload": {
            "num_agents": 2,
            "arrival": {"type": "poisson", "rate_per_second": 10.0},
            "turns": {"type": "constant", "value": 2},
            "input_tokens": {"type": "constant", "value": 2},
            "output_tokens": {"type": "constant", "value": 1},
            "tool_wait_ms": {"type": "constant", "value": 50},
        },
        "invocation_model": {
            "kv": {"bytes_per_token": 10, "retention": "memory"},
        },
        "policy": {
            "type": "static_fcfs",
            "prefill_device": 0,
            "decode_device": 1,
            "memory_resource": 2,
            "far_memory_resource": 3,
        },
    }
