#!/usr/bin/env python3
"""Collect per-task results from sharded HWM hard-maze planning runs.

The evaluator writes one report per shard::

    checkpoint/maze2d_large_diverse/l2_wo_encoder/<run_tag>_shard<i>/
        planning_l2_mpc_report_l2_d4rl_hard
        config.yaml
        summary.json

This script merges those into a single, stable, *per-task* table so that every
later research step can be compared against the baseline with paired
statistics (rescued / regressed tasks) instead of an aggregate success rate.

Example::

    python scripts/collect_hwm_results.py \
        --run-tag hard_hwm --out results/baseline_hard

Outputs (``<out>`` is a path prefix)::

    <out>_tasks.json    one record per task (task_id 0..N-1)
    <out>_tasks.csv     same data, spreadsheet friendly
    <out>_summary.json  aggregate + per-shard + missing shards
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


def _torch_load(path: Path):
    """Load a torch pickle, tolerating the torch>=2.6 ``weights_only`` default."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # very old torch without the kwarg
        return torch.load(path, map_location="cpu")


def _field(obj: Any, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def _as_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return None


def _map_hash(map_layout: Any) -> str:
    return hashlib.sha1(str(map_layout).encode("utf-8")).hexdigest()[:12]


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except Exception:  # pragma: no cover - omegaconf is a hard dependency
        return {}
    if not path.exists():
        return {}
    try:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception:
        return {}


def _dig(mapping: Any, *keys: str, default: Any = None) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-root",
        default="checkpoint/maze2d_large_diverse/l2_wo_encoder",
        help="Directory containing <run-tag>_shard<i> sub-directories",
    )
    parser.add_argument("--run-tag", default="hard_hwm")
    parser.add_argument("--shards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--level", default="hard")
    parser.add_argument("--tasks-per-shard", type=int, default=5)
    parser.add_argument("--task-id-offset", type=int, default=0)
    parser.add_argument(
        "--trials",
        default=None,
        help="Full trial file used to attach map hash / start / target. "
        "Inferred from each shard's config.yaml when omitted.",
    )
    parser.add_argument("--out", default="results/baseline_hard")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root

    out_prefix = Path(args.out)
    if not out_prefix.is_absolute():
        out_prefix = REPO_ROOT / out_prefix

    prefix = f"l2_d4rl_{args.level}"
    shards = [int(s) for s in str(args.shards).replace(" ", ",").split(",") if s != ""]

    records: List[Dict[str, Any]] = []
    per_shard: Dict[str, Any] = {}
    missing: List[str] = []

    for shard in shards:
        shard_dir = run_root / f"{args.run_tag}_shard{shard}"
        report_path = shard_dir / f"planning_l2_mpc_report_{prefix}"
        if not report_path.exists():
            missing.append(str(report_path))
            continue

        report = _torch_load(report_path)
        success = _as_list(_field(report, "success"))
        if success is None:
            raise RuntimeError(f"{report_path} does not contain a `success` field")

        terminations = _as_list(_field(report, "terminations"))
        num_turns = _as_list(_field(report, "num_turns"))
        block_dists = _as_list(_field(report, "block_dists"))

        config = _load_yaml(shard_dir / "config.yaml")
        trials_path = args.trials or _dig(
            config, "eval_cfg", "h_d4rl_planning", "hard", "set_start_target_path"
        )
        trials: Dict[str, Any] = {}
        if trials_path:
            trials_file = Path(trials_path)
            if trials_file.exists():
                loaded = _torch_load(trials_file)
                trials = loaded if isinstance(loaded, dict) else {}

        maps = trials.get("map_layouts")
        starts = trials.get("starts")
        targets = trials.get("targets")

        shard_success = 0
        for local_idx, success_value in enumerate(success):
            task_id = (
                args.task_id_offset + shard * args.tasks_per_shard + local_idx
            )
            record: Dict[str, Any] = {
                "task_id": task_id,
                "shard": shard,
                "local_idx": local_idx,
                "success": int(success_value),
                "termination": (
                    int(terminations[local_idx])
                    if terminations is not None and local_idx < len(terminations)
                    else None
                ),
                "num_turns": (
                    int(num_turns[local_idx])
                    if num_turns is not None and local_idx < len(num_turns)
                    else None
                ),
                "block_dist": (
                    int(block_dists[local_idx])
                    if block_dists is not None and local_idx < len(block_dists)
                    else None
                ),
            }
            if maps is not None and local_idx < len(maps):
                record["map_hash"] = _map_hash(maps[local_idx])
            if starts is not None and local_idx < len(starts):
                record["start"] = [float(x) for x in list(starts[local_idx])[:2]]
            if targets is not None and local_idx < len(targets):
                record["target"] = [float(x) for x in list(targets[local_idx])[:2]]

            shard_success += record["success"]
            records.append(record)

        per_shard[str(shard)] = {
            "num_tasks": len(success),
            "num_success": shard_success,
            "success_rate": shard_success / len(success) if success else None,
            "report": str(report_path),
        }

    records.sort(key=lambda r: r["task_id"])
    task_ids = [r["task_id"] for r in records]
    if len(set(task_ids)) != len(task_ids):
        print("WARNING: duplicate task ids collected", file=sys.stderr)

    total = len(records)
    total_success = sum(r["success"] for r in records)
    summary = {
        "run_tag": args.run_tag,
        "run_root": str(run_root),
        "level": args.level,
        "num_tasks": total,
        "num_success": total_success,
        "success_rate": (total_success / total) if total else None,
        "per_shard": per_shard,
        "missing": missing,
        "task_ids": task_ids,
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_name(out_prefix.name + "_tasks.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    with out_prefix.with_name(out_prefix.name + "_summary.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    csv_path = out_prefix.with_name(out_prefix.name + "_tasks.csv")
    fieldnames = [
        "task_id",
        "shard",
        "local_idx",
        "success",
        "termination",
        "num_turns",
        "block_dist",
        "map_hash",
        "start",
        "target",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    rate = f"{100.0 * summary['success_rate']:.2f}%" if total else "n/a"
    print(f"collected {total} tasks from {len(per_shard)} shard(s): {total_success} success ({rate})")
    print(f"wrote {out_prefix.name}_tasks.json / .csv and {out_prefix.name}_summary.json")
    if missing:
        print(f"WARNING: {len(missing)} shard report(s) missing:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
