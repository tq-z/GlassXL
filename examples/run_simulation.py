"""Run the checked-in reconstructed PD/AFD simulation example."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from agentic_sim import Simulation, load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the breaking PD/AFD simulator with three strict config files."
    )
    parser.add_argument("--hardware", type=Path, help="path to hardware.yaml", default=Path("configs/example/hardware.yaml"))
    parser.add_argument("--workload", type=Path, help="path to workload.yaml", default=Path("configs/example/workload.yaml"))
    parser.add_argument("--mapping", type=Path, help="path to mapping.yaml", default=Path("configs/example/mapping.yaml"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = Simulation(load_config(args.hardware, args.workload, args.mapping)).run()
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "records": {
                    "sessions": len(result.sessions),
                    "turns": len(result.turns),
                    "prefill_batches": len(result.prefill_batches),
                    "decode_steps": len(result.decode_steps),
                },
                "resources": [asdict(item) for item in result.resources],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
