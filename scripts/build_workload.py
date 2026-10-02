"""
Turn real Claude/Codex agent traces into PanAgretti session specs.

Source: TraceLab's syfi_coding_trace.jsonl.gz -- 665k LLM rounds across 8k real
coding-agent sessions.

The mapping is NOT the obvious one. Verified over all 665,453 rows:
`input_tokens_total = prefix_tokens + newly_append_tokens`, and
`prefix_tokens[k] ~= input_tokens_total[k-1]` (median difference -12). So the
cached prefix is the previous request's whole *input*, and the previous round's
*output* reappears inside `newly_append_tokens[k]`. Taking `newly_append_tokens`
as PanAgretti's incremental prefill P therefore double-counts the model's own
output. Against the workload this script actually emits (clamped, with
compaction restarts folded into warm context), the naive mapping inflates mean
P by ~4.4x -- enough to make a decode-bound workload look prefill-bound. (The
raw unclamped recurrence mean is 238 vs 7,472, but that 31x figure conflates
compaction rounds' negative drift with real prefill and overstates the error.)

Correct recurrence, given PanAgretti grows context as C <- C + P + O:

    C_0 = prefix_tokens[0]
    P_0 = newly_append_tokens[0]
    P_k = newly_append_tokens[k] - output_tokens[k-1]
          + (prefix_tokens[k] - input_tokens_total[k-1])
    O_k = output_tokens[k]

A large negative prefix jump means the agent compacted its context, which
PanAgretti cannot represent (context only grows), so we cut a new session there.

Emits sessions.json (replayed per-session structure, preserving within-session
correlation that independent marginal sampling would destroy) plus a
schema-valid workload.yaml carrying seed and horizon.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

TRACE_DEFAULT = Path(
    "/Users/jasonkong/Documents/glassML/TraceLab/trace/syfi_coding_trace.jsonl.gz"
)
COMPACTION_THRESHOLD = -4096


def _parse_time(stamp):
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def _tool_wait_ms(rounds, index):
    if index + 1 >= len(rounds):
        return 0
    here = rounds[index].get("timing_events") or []
    nxt = rounds[index + 1].get("timing_events") or []
    if not here or not nxt:
        return 0
    end = _parse_time(here[-1].get("timestamp"))
    start = _parse_time(nxt[0].get("timestamp"))
    if end is None or start is None or start < end:
        return 0
    return int(min((start - end) * 1000.0, 600_000))


def segment_sessions(path: Path, max_turns: int):
    """Replay each trace session into one or more PanAgretti-shaped segments."""
    raw = defaultdict(list)
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            raw[row["session_id"]].append(row)

    segments, compaction_cuts = [], 0
    for rounds in raw.values():
        rounds.sort(key=lambda item: item["round_index"])
        current = None
        last_output_carry = 0
        for index, row in enumerate(rounds):
            prefix = int(row.get("prefix_tokens") or 0)
            newly = int(row.get("newly_append_tokens") or 0)
            output = int(row.get("output_tokens") or 0)
            wait = _tool_wait_ms(rounds, index)

            # The reference point is input_tokens_total[k-1] = prefix + newly of
            # the previous round. It deliberately EXCLUDES that round's output:
            # the output is re-sent inside newly_append_tokens[k], which is the
            # whole reason the naive mapping double-counts.
            drift = 0 if current is None else prefix - current["last_input_total"]
            compacted = current is not None and drift < COMPACTION_THRESHOLD
            if compacted:
                compaction_cuts += 1
            split = current is not None and len(current["turns"]) >= max_turns

            if current is None or compacted or split:
                if current is not None:
                    segments.append(current)
                fresh_prompt = current is None or compacted
                # PanAgretti models a WARM-START session: the opening context is
                # already recoverable in the KV pool, and cold-start prefill is an
                # explicit V1 non-goal. So a segment's opening input becomes warm
                # context C, not prefill P. Without this a fresh segment's first
                # turn would carry the whole prompt -- up to 944k tokens, which at
                # the calibrated prefill rate monopolises a device for 375 s and
                # starves decode entirely.
                opening = prefix + newly if fresh_prompt else prefix
                current = dict(initial_context_tokens=max(1, opening), turns=[],
                               last_output=0, last_input_total=prefix)
                # A max-turns split is bookkeeping, not a real restart, so it
                # keeps the full incremental formula INCLUDING the prefix drift
                # -- dropping it skewed 9,520 segment openings (median +3 tok).
                prefill = 0 if fresh_prompt else max(0, newly - last_output_carry + drift)
            else:
                prefill = newly - current["last_output"] + drift

            prefill = max(0, prefill)
            current["turns"].append(dict(prefill_tokens=prefill,
                                         output_tokens=output,
                                         tool_wait_ms=wait))
            current["last_output"] = output
            current["last_input_total"] = prefix + newly
            last_output_carry = output
        if current is not None:
            segments.append(current)

    for segment in segments:
        segment.pop("last_output", None)
        segment.pop("last_input_total", None)
    return [s for s in segments if s["turns"]], compaction_cuts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=TRACE_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=Path("configs/glassml/workload"))
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args(argv)

    segments, cuts = segment_sessions(args.trace, args.max_turns)
    rng = random.Random(args.seed)
    rng.shuffle(segments)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sessions.json").write_text(json.dumps(dict(
        source=str(args.trace),
        max_turns=args.max_turns,
        compaction_cuts=cuts,
        segments=segments,
    )), encoding="utf-8")

    (args.out_dir / "workload.yaml").write_text(yaml.safe_dump(dict(
        schema_version=1, seed=args.seed, simulation_end_s=None,
        source=dict(
            type="synthetic", num_sessions=1,
            arrival=dict(type="poisson", rate_per_second=1.0),
            turns=dict(type="constant", value=1),
            context_tokens=dict(type="constant", value=1),
            prefill_tokens=dict(type="constant", value=1),
            output_tokens=dict(type="constant", value=1),
            tool_wait_ms=dict(type="constant", value=0),
        ),
    ), sort_keys=False), encoding="utf-8")

    if args.stats:
        def q(values, p):
            values = sorted(values)
            return values[min(len(values) - 1, int(p * len(values)))]
        turns = [len(s["turns"]) for s in segments]
        context = [s["initial_context_tokens"] for s in segments]
        prefill = [t["prefill_tokens"] for s in segments for t in s["turns"]]
        output = [t["output_tokens"] for s in segments for t in s["turns"]]
        wait = [t["tool_wait_ms"] for s in segments for t in s["turns"]]
        final = [s["initial_context_tokens"]
                 + sum(t["prefill_tokens"] + t["output_tokens"] for t in s["turns"])
                 for s in segments]
        print(f"  segments={len(segments)}  compaction_cuts={cuts}")
        for name, values in (("turns", turns), ("initial_context", context),
                             ("prefill P", prefill), ("output O", output),
                             ("tool_wait_ms", wait), ("final_context", final)):
            print(f"  {name:16s} p50={q(values,.5):8d} p90={q(values,.9):9d} "
                  f"p99={q(values,.99):9d} mean={int(sum(values)/len(values)):8d}")
    print(f"wrote {args.out_dir}/sessions.json and workload.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
