"""Run the checked-in reconstructed PD/AFD simulation example."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from agentic_sim import Simulation, SimulationResult, load_config


_SESSION_PHASES = (
    "ffn_s",
    "attn_s",
    "kv_ingress_s",
    "ffn_all_to_all_s",
    "prefill_s",
)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _session_statistics(result: SimulationResult) -> dict[str, object]:
    """Build observed phase totals and averages for every simulated session."""

    prefill_by_session: dict[int, list] = {}
    for item in result.prefill_sessions:
        prefill_by_session.setdefault(item.session_id, []).append(item)
    turns_by_session: dict[int, list] = {}
    for item in result.turns:
        turns_by_session.setdefault(item.session_id, []).append(item)
    decode_by_session: dict[int, list] = {}
    for step in result.decode_steps:
        for session_id in step.session_ids:
            decode_by_session.setdefault(session_id, []).append(step)

    phase_totals: dict[int, dict[str, float]] = {}
    session_rows: list[dict[str, object]] = []
    for session in result.sessions:
        prefill_rows = prefill_by_session.get(session.session_id, [])
        turn_rows = turns_by_session.get(session.session_id, [])
        decode_rows = decode_by_session.get(session.session_id, [])
        prefill_queue_wait_s = sum(item.prefill_queue_s for item in turn_rows)
        turn_details = [
            {
                "turn_index": item.turn_index,
                "ttft_s": item.ttft_s,
                "prefill_queue_wait_s": item.prefill_queue_s,
                "tool_wait_s": item.tool_wait_s,
            }
            for item in turn_rows
        ]
        phases = {
            "prefill_s": sum(item.service_latency_s for item in prefill_rows),
            "kv_ingress_s": sum(
                item.kv_pool_load_s + item.kv_p2p_s for item in turn_rows
            ),
            "attn_s": sum(item.attention_time_s for item in decode_rows),
            "ffn_s": sum(item.ffn_time_s for item in decode_rows),
            "ffn_all_to_all_s": sum(
                item.ffn_all_to_all_time_s for item in decode_rows
            ),
        }
        phase_totals[session.session_id] = phases
        session_rows.append(
            {
                "session_id": session.session_id,
                "arrival_time_s": session.arrival_time_s,
                "finish_time_s": session.finish_time_s,
                "completed": session.finish_time_s is not None,
                "completed_turns": session.completed_turns,
                "total_turns": session.total_turns,
                "generated_tokens": session.generated_tokens,
                "tool_wait_s": session.tool_wait_s,
                "prefill_queue_wait_s": prefill_queue_wait_s,
                "turns": turn_details,
                "phases_s": phases,
            }
        )

    completed_ids = tuple(
        session.session_id
        for session in result.sessions
        if session.finish_time_s is not None
    )
    averages = {
        phase: (
            sum(phase_totals[session_id][phase] for session_id in completed_ids)
            / len(completed_ids)
            if completed_ids
            else 0.0
        )
        for phase in _SESSION_PHASES
    }
    completed_session_rows = [
        session for session in result.sessions if session.finish_time_s is not None
    ]
    average_tool_wait_s = (
        sum(session.tool_wait_s for session in completed_session_rows)
        / len(completed_session_rows)
        if completed_session_rows
        else 0.0
    )
    average_prefill_queue_wait_s = (
        sum(
            sum(item.prefill_queue_s for item in result.turns
                if item.session_id == session.session_id)
            for session in completed_session_rows
        )
        / len(completed_session_rows)
        if completed_session_rows
        else 0.0
    )
    completed_ttfts = [
        item.ttft_s
        for item in result.turns
        if item.finish_time_s is not None and item.ttft_s is not None
    ]
    average_turn_ttft_s = (
        sum(completed_ttfts) / len(completed_ttfts)
        if completed_ttfts
        else 0.0
    )
    p90_turn_ttft_s = _percentile(completed_ttfts, 0.90)
    return {
        "units": "seconds",
        "average_scope": "completed_sessions",
        "session_count": len(session_rows),
        "completed_session_count": len(completed_ids),
        "average_phase_time_s": averages,
        "average_tool_wait_s": average_tool_wait_s,
        "average_prefill_queue_wait_s": average_prefill_queue_wait_s,
        "average_turn_ttft_s": average_turn_ttft_s,
        "average_turn_ttft_scope": "completed_turns",
        "p90_turn_ttft_s": p90_turn_ttft_s,
        "p90_turn_ttft_scope": "completed_turns",
        "sessions": session_rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the breaking PD/AFD simulator with three strict config files."
    )
    parser.add_argument("--hardware", type=Path, help="path to hardware.yaml", default=Path("configs/kimiK2/GPU/hardware.yaml"))
    parser.add_argument("--workload", type=Path, help="path to workload.yaml", default=Path("configs/example/workload.yaml"))
    parser.add_argument("--mapping", type=Path, help="path to mapping.yaml", default=Path("configs/kimiK2/GPU/mapping.yaml"))
    parser.add_argument(
        "--session-stats",
        type=Path,
        default=None,
        help="optional JSON path for per-session phase statistics",
    )
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
    if args.session_stats is not None:
        statistics = _session_statistics(result)
        args.session_stats.parent.mkdir(parents=True, exist_ok=True)
        args.session_stats.write_text(
            json.dumps(statistics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "session_stats": {
                        "path": str(args.session_stats),
                        "average_phase_time_s": statistics["average_phase_time_s"],
                    }
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
