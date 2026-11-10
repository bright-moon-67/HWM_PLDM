#!/usr/bin/env python3
"""Build the stage-2 candidate table from an ECCHP collection run.

Stage 2 of the research plan compares, for the *same* decision state, the plan
the original HWM picked (candidate 0) with a few independently re-sampled
high-level plans.  The planner/evaluator side only writes raw
``candidate_outcome`` events into ``diagnostics.jsonl``; this script turns them
into the tidy table consumed by ``evaluate_candidate_oracle.py``:

    {"task_id": 4, "decision_id": 3, "plan_id": 9, "step": 40, "horizon": 40,
     "candidates": [{"candidate_id": 0, "source": "hwm",
                     "model_score": 0.81, "cross_mse": 0.11,
                     "predicted_progress": 0.42, "selected": true,
                     "real_return": 2.0, "graph_distance_before": 12,
                     "graph_distance_after": 10}, ...]}

Usage::

    python scripts/collect_plan_candidates.py \
        --diagnostics checkpoint/.../collect_task7_shard1 \
        --out results/candidates_task7

The output names are ``<out>_candidates.jsonl`` and ``<out>_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANDIDATE_EVENT = "candidate_outcome"
REPLAY_ERROR_EVENTS = ("candidate_replay_error", "candidate_replay_unavailable")


def _iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"[warn] {path}:{line_number} is not valid JSON, skipped",
                        file=sys.stderr,
                    )
                    continue
                if isinstance(record, dict):
                    yield path, line_number, record


def _find_diagnostics(paths: Iterable[str]) -> List[Path]:
    """Expand files / directories into a sorted list of JSONL files."""
    found: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.rglob("diagnostics*.jsonl")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(raw)
    # De-duplicate while keeping determinism.
    unique: Dict[str, Path] = {}
    for path in found:
        unique[str(path.resolve())] = path
    return list(unique.values())


def build_groups(files: Iterable[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group ``candidate_outcome`` events by ``(task, decision, horizon)``."""
    grouped: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    errors: Counter = Counter()
    event_counts: Counter = Counter()

    for _, _, record in _iter_jsonl(files):
        event = str(record.get("event", ""))
        event_counts[event] += 1
        if event in REPLAY_ERROR_EVENTS:
            errors[event] += 1
            continue
        if event != CANDIDATE_EVENT:
            continue

        task_id = int(record.get("task_id", -1))
        decision_id = int(record.get("decision_id", -1))
        horizon = int(record.get("horizon", -1))
        candidate_id = int(record.get("candidate_id", -1))
        if -1 in (decision_id, horizon, candidate_id):
            errors["malformed_candidate_outcome"] += 1
            continue

        key = (task_id, decision_id, horizon)
        group = grouped.setdefault(
            key,
            {
                "task_id": task_id,
                "decision_id": decision_id,
                "horizon": horizon,
                "plan_id": int(record.get("plan_id", -1)),
                "step": int(record.get("step", -1)),
                "replay_mode": record.get("replay_mode"),
                "candidates": {},
            },
        )
        group["candidates"][candidate_id] = {
            "candidate_id": candidate_id,
            "source": record.get("candidate_source", "unknown"),
            "selected": bool(record.get("selected", False)),
            "model_score": record.get("model_score"),
            "cross_mse": record.get("cross_mse"),
            "predicted_progress": record.get("predicted_progress"),
            "real_return": record.get("real_return"),
            "graph_distance_before": record.get("graph_distance_before"),
            "graph_distance_after": record.get("graph_distance_after"),
            "start_xy": record.get("start_xy"),
            "end_xy": record.get("end_xy"),
            "executed_steps": record.get("executed_steps"),
        }

    groups: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        group["candidates"] = [
            group["candidates"][candidate_id]
            for candidate_id in sorted(group["candidates"])
        ]
        groups.append(group)

    summary: Dict[str, Any] = {
        "n_files": len(list(files)),
        "n_groups": len(groups),
        "event_counts": dict(event_counts),
        "replay_errors": dict(errors),
    }
    return groups, summary


def describe_groups(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Coverage / sanity statistics for the collected candidate table."""
    per_horizon: Dict[str, Dict[str, Any]] = {}
    by_horizon: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_horizon[int(group["horizon"])].append(group)

    for horizon, items in sorted(by_horizon.items()):
        n_candidates = Counter(len(g["candidates"]) for g in items)
        labelled = [
            g
            for g in items
            if all(c.get("real_return") is not None for c in g["candidates"])
        ]
        with_baseline = sum(
            1 for g in items if any(c["candidate_id"] == 0 for c in g["candidates"])
        )
        model_scored = [
            g
            for g in items
            if all(c.get("model_score") is not None for c in g["candidates"])
        ]
        per_horizon[str(horizon)] = {
            "n_decisions": len(items),
            "n_tasks": len({int(g["task_id"]) for g in items}),
            "candidate_count_histogram": {
                str(k): int(v) for k, v in sorted(n_candidates.items())
            },
            "decisions_with_full_real_returns": len(labelled),
            "decisions_with_baseline_candidate": with_baseline,
            "decisions_with_full_model_scores": len(model_scored),
            "mean_candidates": (
                sum(len(g["candidates"]) for g in items) / len(items)
                if items
                else 0.0
            ),
        }
    return {"per_horizon": per_horizon}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        required=True,
        help="diagnostics.jsonl file(s) or run directories to read",
    )
    parser.add_argument(
        "--out",
        default="results/plan_candidates",
        help="output prefix (writes <out>_candidates.jsonl / <out>_summary.json)",
    )
    args = parser.parse_args()

    files = _find_diagnostics(args.diagnostics)
    if not files:
        print("no diagnostics files found", file=sys.stderr)
        return 2

    groups, summary = build_groups(files)
    summary.update(describe_groups(groups))

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    candidates_path = out_prefix.with_name(out_prefix.name + "_candidates.jsonl")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.json")

    with candidates_path.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"read {summary['n_files']} file(s) -> {len(groups)} decision groups")
    for horizon, stats in summary.get("per_horizon", {}).items():
        print(
            f"  horizon={horizon}: {stats['n_decisions']} decisions, "
            f"{stats['decisions_with_full_real_returns']} fully labelled, "
            f"mean {stats['mean_candidates']:.1f} candidates"
        )
    if summary["replay_errors"]:
        print(f"  replay problems: {summary['replay_errors']}")
    print(f"wrote {candidates_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
