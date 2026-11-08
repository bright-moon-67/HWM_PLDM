#!/usr/bin/env python3
"""Which execution signal actually predicts failure?

The Step 1/2 diagnostics produce three families of per-decision numbers:

* ``cross_mse``            - absolute cross-level gap at the L1 plan terminal
* ``pred_progress_h``      - predicted fraction of the subgoal distance closed
                             in the ``replan_every`` primitives that will run
* ``execution_progress_gap``/``exec_mse_obs`` - what actually happened

This script aggregates them per task and computes a rank AUC for *failure*
(AUC > 0.5 means "larger value -> more likely to fail"), plus the mean/std of
each metric for succeeded vs failed tasks. It is pure post-processing of
``diagnostics.jsonl`` files and needs neither a GPU nor MuJoCo.

Usage::

    python scripts/analyze_exec_signals.py \
        --path checkpoint/maze2d_large_diverse/l2_wo_encoder/shard6_diag200_shard6 \
        --tasks analysis/baseline_tasks.json_tasks.json \
        --early-steps 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# metric name -> (event, field)
METRICS: Dict[str, Tuple[str, str]] = {
    "cross_mse": ("plan_gap", "cross_mse"),
    "path_cost": ("plan_gap", "path_cost"),
    "pred_progress_h": ("plan_gap", "pred_progress_h"),
    "pred_progress_4": ("plan_gap", "pred_progress_4"),
    "pred_progress_10": ("plan_gap", "pred_progress_10"),
    "l2_subgoal_norm": ("plan_gap", "l2_subgoal_norm"),
    "exec_mse_obs": ("exec_error", "exec_mse_obs"),
    "real_progress_h": ("exec_error", "real_progress_h"),
    "execution_progress_gap": ("exec_error", "execution_progress_gap"),
}


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
            resolved.extend(sorted(path.rglob("diagnostics.jsonl")))
        elif path.exists():
            resolved.append(path)
        else:
            print(f"WARNING: {path} not found", file=sys.stderr)
    seen, unique = set(), []
    for path in resolved:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _floats(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: List[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                out.append(float(item))
            elif isinstance(item, list):  # schema v2 nested payloads
                out.extend(x for x in item if isinstance(x, (int, float)))
        return out
    return []


def _pairs(task_ids: List[int], value: Any) -> Iterable[Tuple[int, float]]:
    floats = _floats(value)
    for index, task_id in enumerate(task_ids):
        if index < len(floats):
            yield int(task_id), floats[index]


def collect(paths: List[Path], early_steps: int) -> Dict[str, Dict[str, Any]]:
    """Per-task, per-metric samples (all steps and early steps)."""
    all_values: Dict[str, Dict[int, List[float]]] = {
        name: defaultdict(list) for name in METRICS
    }
    early_values: Dict[str, Dict[int, List[float]]] = {
        name: defaultdict(list) for name in METRICS
    }

    for record in _iter_jsonl(paths):
        event = record.get("event")
        step = int(record.get("step", -1))
        task_ids = [int(t) for t in record.get("task_ids", [])]
        for name, (expected_event, field) in METRICS.items():
            if event != expected_event:
                continue
            for task_id, value in _pairs(task_ids, record.get(field)):
                all_values[name][task_id].append(value)
                if early_steps > 0 and step < early_steps:
                    early_values[name][task_id].append(value)

    out: Dict[str, Dict[str, Any]] = {}
    for name in METRICS:
        out[name] = {
            "all": dict(all_values[name]),
            "early": dict(early_values[name]),
        }
    return out


def _rank(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def roc_auc(scores: List[float], labels: List[int]) -> Optional[float]:
    """AUC with ``labels == 1`` as the positive class (here: failure)."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rank(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _quantiles(values: List[float], qs: Tuple[float, ...]) -> List[float]:
    if not values:
        return []
    ordered = sorted(values)
    out = []
    for q in qs:
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        out.append(ordered[index])
    return out


def sweep_thresholds(
    buckets: Dict[str, Dict[int, List[float]]],
    labels: Dict[int, int],
    metric: str,
    direction: str,
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """Simulate how many decisions a given trigger threshold would fire.

    ``direction == "high"`` triggers when ``value > threshold`` (cross_mse);
    ``direction == "low"`` triggers when ``value < threshold`` (pred_progress).

    This needs no second plan: it only answers "how often would we have
    retried, and do the retried decisions concentrate on tasks that eventually
    failed?".
    """
    task_values: Dict[int, List[float]] = buckets[metric]
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        triggered_tasks, clean_tasks = [], []
        n_decisions = 0
        n_triggered = 0
        for task_id, samples in task_values.items():
            if task_id not in labels or not samples:
                continue
            n_decisions += len(samples)
            hits = (
                sum(1 for v in samples if v > threshold)
                if direction == "high"
                else sum(1 for v in samples if v < threshold)
            )
            n_triggered += hits
            (triggered_tasks if hits else clean_tasks).append(labels[task_id])
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "threshold": threshold,
                "decisions": n_decisions,
                "triggered": n_triggered,
                "trigger_rate": (
                    n_triggered / n_decisions if n_decisions else float("nan")
                ),
                "success_tasks_triggered": triggered_tasks.count(1),
                "failure_tasks_triggered": triggered_tasks.count(0),
                "success_tasks_clean": clean_tasks.count(1),
                "failure_tasks_clean": clean_tasks.count(0),
            }
        )
    return rows


def _mean_std(values: List[float]) -> str:
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    if len(values) < 2:
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} ± {statistics.stdev(values):.4f}"


def _report_block(
    title: str,
    buckets: Dict[str, Dict[int, List[float]]],
    labels: Dict[int, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    print(f"\n== {title} ==")
    print(
        f"{'metric':<24}{'AUC(fail)':>10}  "
        f"{'success':>22}  {'failure':>22}  n"
    )
    for name in METRICS:
        task_means: Dict[int, float] = {}
        for task_id, samples in buckets[name].items():
            if samples:
                task_means[task_id] = statistics.fmean(samples)

        common = [t for t in task_means if t in labels]
        if not common:
            print(f"{name:<24}{'n/a':>10}  (no labelled tasks)")
            continue

        scores = [task_means[t] for t in common]
        truth = [1 - labels[t] for t in common]  # 1 == failure
        auc = roc_auc(scores, truth)

        succeeded = [task_means[t] for t in common if labels[t] == 1]
        failed = [task_means[t] for t in common if labels[t] == 0]
        print(
            f"{name:<24}{(f'{auc:.3f}' if auc is not None else 'n/a'):>10}  "
            f"{_mean_std(succeeded):>22}  {_mean_std(failed):>22}  {len(common)}"
        )
        rows.append(
            {
                "metric": name,
                "window": title,
                "auc_failure": auc,
                "success_mean": statistics.fmean(succeeded) if succeeded else None,
                "failure_mean": statistics.fmean(failed) if failed else None,
                "n_tasks": len(common),
            }
        )
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path",
        nargs="+",
        required=True,
        help="directories containing diagnostics.jsonl (searched recursively)",
    )
    parser.add_argument(
        "--tasks", required=True, help="per-task json from collect_hwm_results.py"
    )
    parser.add_argument(
        "--early-steps",
        type=int,
        default=100,
        help="also report metrics computed only from steps < N (0 disables)",
    )
    parser.add_argument("--out", default=None, help="optional output json path")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also sweep trigger thresholds and report trigger rates",
    )
    args = parser.parse_args(argv)

    paths = _resolve_paths(args.path)
    if not paths:
        print("no diagnostics.jsonl found", file=sys.stderr)
        return 1

    tasks_path = Path(args.tasks)
    if not tasks_path.is_absolute():
        tasks_path = REPO_ROOT / tasks_path
    labels = {
        int(item["task_id"]): int(item["success"])
        for item in json.loads(tasks_path.read_text(encoding="utf-8"))
    }

    buckets = collect(paths, args.early_steps)
    print(f"files: {len(paths)}  labelled tasks: {len(labels)}")

    rows = _report_block("all steps", {n: b["all"] for n, b in buckets.items()}, labels)
    if args.early_steps > 0:
        rows += _report_block(
            f"steps < {args.early_steps}",
            {n: b["early"] for n, b in buckets.items()},
            labels,
        )

    print(
        "\nAUC(fail) > 0.5 => larger metric values go with failures. "
        "A metric with AUC ~ 0.5 carries no failure information."
    )

    if args.sweep:
        flat = {n: b["all"] for n, b in buckets.items()}
        sweep_rows: List[Dict[str, Any]] = []
        print("\n== trigger threshold sweep (all steps) ==")
        print(
            f"{'metric':<20}{'rule':<12}{'thr':>9}{'rate':>8}"
            f"{'trig_succ':>11}{'trig_fail':>11}{'clean_succ':>12}{'clean_fail':>12}"
        )
        for metric, direction in (
            ("cross_mse", "high"),
            ("pred_progress_h", "low"),
            ("pred_progress_4", "low"),
        ):
            samples = [v for task_vals in flat[metric].values() for v in task_vals]
            if not samples:
                continue
            thresholds = _quantiles(
                samples, (0.1, 0.25, 0.5, 0.75, 0.9)
            )
            if direction == "high":
                thresholds = [0.02, 0.04, 0.06, 0.08, 0.10]
            else:
                thresholds = [0.3, 0.5, 0.7, 0.85, 0.95]
            rows_m = sweep_thresholds(flat, labels, metric, direction, thresholds)
            sweep_rows.extend(rows_m)
            for row in rows_m:
                print(
                    f"{metric:<20}{direction:<12}{row['threshold']:>9.3f}"
                    f"{row['trigger_rate']:>8.3f}"
                    f"{row['success_tasks_triggered']:>11}"
                    f"{row['failure_tasks_triggered']:>11}"
                    f"{row['success_tasks_clean']:>12}"
                    f"{row['failure_tasks_clean']:>12}"
                )
        rows.extend(sweep_rows)
        print(
            "\ntrig_* = tasks with at least one triggered decision; "
            "clean_* = tasks never triggered."
        )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
