"""Run the PR1--PR3 example and print its in-memory metrics."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_sim import Simulation, load_config


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_config(repository_root / "configs" / "example.yaml")
    result = Simulation(config).run()
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print("\nResources:")
    for resource in result.resources:
        print(json.dumps(resource.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
