"""Load and validate a Milestone 1 three-file configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from agentic_sim import load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and validate the breaking PD/AFD three-file configuration."
    )
    parser.add_argument("--hardware", required=True, type=Path, help="path to hardware.yaml")
    parser.add_argument("--workload", required=True, type=Path, help="path to workload.yaml")
    parser.add_argument("--mapping", required=True, type=Path, help="path to mapping.yaml")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.hardware, args.workload, args.mapping)
    print(
        f"Loaded schema v{config.hardware.schema_version}: "
        f"{len(config.hardware.devices)} devices, "
        f"{config.workload.source.num_sessions} sessions, "
        f"{config.mapping.model.num_experts} experts"
    )


if __name__ == "__main__":
    main()
