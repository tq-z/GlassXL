"""Run parallel rate_per_second scans for the Kimi K2 GPU configuration."""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


# Change this constant to redirect the complete scan output.
RESULTS_ROOT = Path("/mnt/bravo/tiz014/PanAgretti/results/Dell/kimiK2/GPU1M")

RATES_PER_SECOND = (0.1, 0.5, 1, 2, 4, 8, 16, 32, 64)
LOW_RATE_SESSION_DIVISOR = 10
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_TEMPLATE = REPOSITORY_ROOT / "configs/example/workload.yaml"
HARDWARE_CONFIG = REPOSITORY_ROOT / "configs/kimiK2/GPU1M/hardware.yaml"
MAPPING_CONFIG = REPOSITORY_ROOT / "configs/kimiK2/GPU1M/mapping.yaml"
SIMULATION_SCRIPT = REPOSITORY_ROOT / "examples/run_simulation.py"


def _rate_directory(rate: float) -> Path:
    return RESULTS_ROOT / f"rate_{rate:g}"


def _write_workload(rate: float) -> Path:
    output_directory = _rate_directory(rate)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "workload.yaml"

    with WORKLOAD_TEMPLATE.open(encoding="utf-8") as stream:
        workload: dict[str, Any] = yaml.safe_load(stream)

    workload["source"]["arrival"]["rate_per_second"] = rate
    if rate == 0.1:
        workload["source"]["num_sessions"] = max(
            1, workload["source"]["num_sessions"] // LOW_RATE_SESSION_DIVISOR
        )

    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(workload, stream, sort_keys=False)

    return output_path


def _run_one(rate: float) -> tuple[float, int]:
    output_directory = _rate_directory(rate)
    workload_path = output_directory / "workload.yaml"
    result_path = output_directory / "result.json"
    session_stats_path = output_directory / "tmp_session.json"

    command = [
        sys.executable,
        str(SIMULATION_SCRIPT),
        "--hardware",
        str(HARDWARE_CONFIG),
        "--workload",
        str(workload_path),
        "--mapping",
        str(MAPPING_CONFIG),
        "--session-stats",
        str(session_stats_path),
    ]
    with result_path.open("w", encoding="utf-8") as result_stream:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=result_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return rate, completed.returncode


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Generate every workload before any simulation process starts.
    for rate in RATES_PER_SECOND:
        _write_workload(rate)

    worker_count = min(len(RATES_PER_SECOND), os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count
    ) as executor:
        futures = {
            executor.submit(_run_one, rate): rate for rate in RATES_PER_SECOND
        }
        for future in concurrent.futures.as_completed(futures):
            rate, return_code = future.result()
            print(f"rate_per_second={rate:g}: exit_code={return_code}", flush=True)
            if return_code != 0:
                raise SystemExit(
                    f"simulation failed for rate_per_second={rate:g}; "
                    f"see {_rate_directory(rate)}/result.json"
                )


if __name__ == "__main__":
    main()
