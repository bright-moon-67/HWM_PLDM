#!/usr/bin/env python3
"""Summarise ``diagnostics.jsonl`` produced by the planning evaluator.

Usage::

    python scripts/summarize_diagnostics.py \
        --path checkpoint/maze2d_large_diverse/l2_wo_encoder/short_task1_diag

Prints, per task id:

* ``plan_gap``  : cross-level gap (mean / median / max) and path cost
* ``exec_error``: 4-step prediction error in XY (mean / max)
* ``retry``     : trigger / acceptance counts and extra planning time

Optionally joins ``--tasks`` (from ``collect_hwm_results.py``) to compare
succeeded vs failed tasks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _resolve_paths(raw_paths: List[str]) -> List[Path]:
    resolved: List[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.is_dir():
            resolved.extend(sorted(path.glob("diagnostics.jsonl")))
            resolved.extend(sorted(path.glob("*/diagnostics.jsonl")))
        elif path.exists():
            resolved.append(path)
        else:
            print(f"WARNING: {path} not found", file=sys.stderr)
    # de-duplicate while keeping order
    seen = set()
    unique = []
    for path in resolved:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _as_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [float(x) for x in value]
    return []


def _stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def _pair(task_ids: List[int], values: Any) -> Iterable[tuple]:
    floats = _as_list(values)
    for index, task_id in enumerate(task_ids):
        if index < len(floats):
            yield int(task_id), floats[index]


def collect(paths: List[Path]) -> Dict[str, Any]:
    gaps: Dict[int, List[float]] = defaultdict(list)
    selected_gaps: Dict[int, List[float]] = defaultdict(list)
    path_costs: Dict[int, List[float]] = defaultdict(list)
    exec_errors: Dict[int, List[float]] = defaultdict(list)
    events: Dict[str, int] = defaultdict(int)
    retries: List[Dict[str, Any]] = []

    for record in _iter_jsonl(paths):
        event = record.get("event")
        events[str(event)] += 1
        task_ids = [int(t) for t in record.get("task_ids", [])]

        if event == "plan_gap":
            for task_id, value in _pair(task_ids, record.get("cross_mse")):
                gaps[task_id].append(value)
            for task_id, value in _pair(task_ids, record.get("path_cost")):
                path_costs[task_id].append(value)
            for task_id, value in _pair(task_ids, record.get("gap_selected")):
                selected_gaps[task_id].append(value)

        elif event == "exec_error":
            for task_id, value in _pair(task_ids, record.get("exec_mse_xy")):
                exec_errors[task_id].append(value)

        elif event == "retry":
            retries.append(record)

    tasks: Dict[str, Any] = {}
    task_ids = sorted(set(gaps) | set(exec_errors) | set(selected_gaps))
    for task_id in task_ids:
        entry = {
            "cross_mse": _stats(gaps.get(task_id, [])),
            "path_cost": _stats(path_costs.get(task_id, [])),
            "exec_mse_xy": _stats(exec_errors.get(task_id, [])),
        }
        selected = selected_gaps.get(task_id, [])
        if selected:
            entry["gap_selected"] = _stats(selected)
        tasks[str(task_id)] = entry

    retry_summary = {
        "decisions": len(retries),
        "triggered": sum(int(r.get("n_triggered", 0)) for r in retries),
        "accepted": sum(int(r.get("n_accepted", 0)) for r in retries),
        "wall_ms_total": float(sum(float(r.get("wall_ms", 0.0)) for r in retries)),
    }

    return {
        "files": [str(path) for path in paths],
        "events": dict(events),
        "tasks": tasks,
        "retry": retry_summary,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path",
        nargs="+",
        default=["checkpoint/maze2d_large_diverse/l2_wo_encoder"],
        help="directories containing diagnostics.jsonl (searched one level deep)",
    )
    parser.add_argument("--tasks", default=None, help="tasks json from collect_hwm_results")
    parser.add_argument("--out", default=None, help="optional output json path")
    args = parser.parse_args(argv)

    paths = _resolve_paths(args.path)
    if not paths:
        print("no diagnostics.jsonl found", file=sys.stderr)
        return 1

    summary = collect(paths)

    labels: Dict[int, int] = {}
    if args.tasks:
        tasks_path = Path(args.tasks)
        if not tasks_path.is_absolute():
            tasks_path = REPO_ROOT / tasks_path
        for item in json.loads(tasks_path.read_text(encoding="utf-8")):
            labels[int(item["task_id"])] = int(item["success"])

    for task_id, entry in summary["tasks"].items():
        parts = []
        if entry["cross_mse"]:
            parts.append(
                "cross_mse mean={:.5f} median={:.5f} max={:.5f}".format(
                    entry["cross_mse"]["mean"],
                    entry["cross_mse"]["median"],
                    entry["cross_mse"]["max"],
                )
            )
        if entry["exec_mse_xy"]:
            parts.append(
                "exec_mse_xy mean={:.5f} max={:.5f}".format(
                    entry["exec_mse_xy"]["mean"], entry["exec_mse_xy"]["max"]
                )
            )
        if entry.get("gap_selected"):
            parts.append(
                "gap_selected mean={:.5f}".format(entry["gap_selected"]["mean"])
            )
        label = labels.get(int(task_id))
        prefix = f"task {task_id}"
        if label is not None:
            prefix += f" (baseline {'success' if label else 'failure'})"
        print(prefix + ": " + " | ".join(parts) if parts else prefix + ": no metrics")

    retry = summary["retry"]
    print(
        "retry decisions={decisions} triggered={triggered} accepted={accepted} "
        "extra_wall={wall_ms_total:.1f} ms".format(**retry)
    )
    print("events:", summary["events"])

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
