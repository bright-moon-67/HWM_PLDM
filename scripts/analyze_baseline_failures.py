#!/usr/bin/env python3
"""Characterise the 13 failures of the 40-task baseline hard-HWM run.

Reads the per-shard ``planning_l2_mpc_result_l2_d4rl_hard`` files produced by the
baseline evaluation (67.5% success, 27/40) and derives, per task, the real
execution trajectory diagnostics that tell *why* an episode failed:

* how close the robot ever got to the goal (min real distance and when),
* whether it was still moving at the end (stuck vs. time-out while progressing),
* path length vs. the ideal block distance (detours / oscillation),
* how often it walked away from the goal after approaching it,
* the world-model prediction error along the way.

Usage::

    /root/miniconda3/envs/pldm_test/bin/python scripts/analyze_baseline_failures.py \
        --out analysis/baseline_failures.json

Only the trajectory files are read; nothing is re-run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUN_ROOT = REPO_ROOT / "checkpoint/maze2d_large_diverse/l2_wo_encoder"
TRIALS_DIR = REPO_ROOT / "pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"

# RENDER_STATS["maze2d_large_diverse"] (pldm_envs/diverse_maze/data_generation/maze_stats.py)
OBS_MIN_TOTAL = -0.636125
OBS_RANGE_TOTAL = 10.068675
GRID_N = 10
GRID_SIZE = OBS_RANGE_TOTAL / GRID_N
SUCCESS_RADIUS = 0.5  # d4rl maze2d pays reward 1.0 inside this radius


# --------------------------------------------------------------------------- #
# grid helpers
# --------------------------------------------------------------------------- #
def xy_to_ij(xy: Tuple[float, float]) -> Tuple[int, int]:
    return (
        int((xy[0] - OBS_MIN_TOTAL) // GRID_SIZE),
        int((xy[1] - OBS_MIN_TOTAL) // GRID_SIZE),
    )


def bfs_block_dist(layout: str, start_ij: Tuple[int, int], goal_ij: Tuple[int, int]) -> Optional[int]:
    field = bfs_field(layout, goal_ij)
    return field.get(start_ij)


def bfs_field(layout: str, goal_ij: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    """BFS distance (in blocks) from every free cell to the goal cell."""
    rows = layout.split("\\")

    def free(i: int, j: int) -> bool:
        return 0 <= i < len(rows) and 0 <= j < len(rows[0]) and rows[i][j] == "O"

    if not free(*goal_ij):
        return {}
    dist = {goal_ij: 0}
    queue = deque([goal_ij])
    while queue:
        i, j = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (i + di, j + dj)
            if nxt not in dist and free(*nxt):
                dist[nxt] = dist[(i, j)] + 1
                queue.append(nxt)
    return dist


# --------------------------------------------------------------------------- #
# trajectory metrics
# --------------------------------------------------------------------------- #
def as_xy(traj: List[torch.Tensor]) -> torch.Tensor:
    """(T, B, 2) tensor of world positions."""
    return torch.stack([t[:, :2].double() for t in traj], dim=0)


def analyse_env(
    loc: torch.Tensor,
    goal: torch.Tensor,
    rewards: torch.Tensor,
    preds: Optional[torch.Tensor],
    layout: str,
) -> Dict[str, Any]:
    """loc: (T, 2), goal: (2,), rewards: (T-1,), preds: (n, 2) or None.

    The evaluation keeps stepping *all* environments until the 500-step budget,
    so a successful robot keeps wandering after it reached the goal. Metrics
    that describe "how the episode went" are therefore cut at ``success_step``;
    for failures the full episode is used.
    """
    T = loc.shape[0]
    d = (loc - goal).norm(dim=1)

    success = bool(float(rewards.max()) > 0.0) if rewards.numel() else False
    success_step = None
    if success:
        success_step = int((rewards > 0).nonzero()[0]) + 1  # reward at t -> step t+1

    horizon = success_step if success else T
    loc_w = loc[:horizon]
    d_w = d[:horizon]

    step_disp = (loc_w[1:] - loc_w[:-1]).norm(dim=1)
    path_len = float(step_disp.sum())

    d0 = float(d_w[0])
    dmin = float(d_w.min())
    tmin = int(d_w.argmin())
    dfinal = float(d_w[-1])

    def first_within(radius: float) -> Optional[int]:
        idx = (d_w <= radius).nonzero()
        return int(idx[0]) if len(idx) else None

    # distance sampled every 50 steps (so the table reads like a progress trace)
    trace = {int(t): round(float(d[t]), 2) for t in range(0, horizon, 50)}
    trace[horizon - 1] = round(dfinal, 2)

    # is it still moving at the end of its window?
    tail_len = min(100, horizon - 1)
    tail_disp = float(step_disp[-(tail_len + 1) :].sum()) if tail_len > 0 else 0.0
    tail_net = float((loc_w[-1] - loc_w[-1 - tail_len]).norm()) if tail_len > 0 else 0.0

    # oscillation: number of "approached a local minimum, then walked >= 1.5 away"
    # events on the smoothed distance trace
    win = min(25, max(1, horizon // 10))
    smooth = torch.nn.functional.avg_pool1d(
        d_w.view(1, 1, -1), kernel_size=win, stride=1
    ).view(-1)
    cycles = 0
    last_min = float(smooth[0])
    for value in smooth[1:]:
        value = float(value)
        if value < last_min:
            last_min = value
        elif value > last_min + 1.5:
            cycles += 1
            last_min = value

    # how often was the robot almost standing still (blocked / pinned to a wall)
    frozen = (step_disp < 0.005).double().mean() if step_disp.numel() else torch.tensor(0.0)

    # grid-cell visits: a looping high-level policy keeps re-entering the same cells
    cells = [xy_to_ij((float(x), float(y))) for x, y in loc_w]
    counts: Dict[Tuple[int, int], int] = {}
    for cell in cells:
        counts[cell] = counts.get(cell, 0) + 1
    unique_cells = len(counts)
    max_cell_visits = max(counts.values()) if counts else 0
    visited_cells = sum(1 for c in counts.values() if c > 2)

    # progress efficiency: how much of the travelled path actually reduced distance
    progress = max(0.0, d0 - dmin)

    # world-model prediction error (10 steps ahead) inside the analysed window
    pred_err = None
    if preds is not None and preds.shape[0] > 0:
        n = min(preds.shape[0], horizon - 10)
        if n > 0:
            err = (preds[:n] - loc[10 : 10 + n]).norm(dim=1)
            pred_err = {
                "mean": round(float(err.mean()), 3),
                "median": round(float(err.median()), 3),
                "p90": round(float(err.quantile(0.9)), 3),
                "max": round(float(err.max()), 3),
            }

    start_ij = xy_to_ij((float(loc[0, 0]), float(loc[0, 1])))
    goal_ij = xy_to_ij((float(goal[0]), float(goal[1])))
    field = bfs_field(layout, goal_ij)
    block_dist = field.get(start_ij)
    ideal_len = None if block_dist is None else block_dist * GRID_SIZE

    # graph-distance trace: how far (in blocks) the robot is from the goal cell
    graph = [field.get(cell) for cell in cells]
    graph_valid = [g for g in graph if g is not None]
    graph_stats = None
    if graph_valid:
        g_start = graph[0] if graph[0] is not None else graph_valid[0]
        g_min = min(graph_valid)
        g_final = graph[-1] if graph[-1] is not None else graph_valid[-1]
        farther = sum(1 for g in graph_valid if g > g_start) / len(graph_valid)
        # cell-to-cell transitions: does the walk reduce or increase the graph distance?
        steps_toward = steps_away = 0
        for prev, cur in zip(graph[:-1], graph[1:]):
            if prev is None or cur is None:
                continue
            if cur < prev:
                steps_toward += 1
            elif cur > prev:
                steps_away += 1
        idx_last = max(0, len(graph_valid) - 1)
        idx_200 = max(0, len(graph_valid) - 201)
        graph_stats = {
            "start": g_start,
            "min": g_min,
            "final": g_final,
            "max": max(graph_valid),
            "delta_last_200": g_final - graph_valid[idx_200],
            "frac_more_than_2_blocks_from_goal": round(
                sum(1 for g in graph_valid if g > 2) / len(graph_valid), 3
            ),
            "frac_went_farther_than_start": round(farther, 3),
            "cell_transitions_toward": steps_toward,
            "cell_transitions_away": steps_away,
            "trace": {
                int(t): graph[t]
                for t in range(0, horizon, 50)
                if t < len(graph)
            },
        }
        graph_stats["trace"][horizon - 1] = graph[-1]

    return {
        "success": success,
        "success_step": success_step,
        "steps": T - 1,
        "start": [round(float(loc[0, 0]), 3), round(float(loc[0, 1]), 3)],
        "goal": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
        "start_ij": list(start_ij),
        "goal_ij": list(goal_ij),
        "block_dist_recomputed": block_dist,
        "ideal_path_len": None if ideal_len is None else round(ideal_len, 2),
        "d_start": round(d0, 3),
        "d_final": round(dfinal, 3),
        "d_min": round(dmin, 3),
        "t_at_d_min": tmin,
        "d_trace": trace,
        "first_within_2": first_within(2.0),
        "first_within_1": first_within(1.0),
        "first_within_05": first_within(SUCCESS_RADIUS),
        "path_len": round(path_len, 2),
        "path_efficiency": round(progress / path_len, 4) if path_len > 0 else None,
        "detour_ratio_vs_ideal": (
            round(path_len / ideal_len, 2) if ideal_len else None
        ),
        "tail_total_disp": round(tail_disp, 3),
        "tail_net_disp": round(tail_net, 3),
        "oscillation_cycles": cycles,
        "frozen_frac": round(float(frozen), 3),
        "unique_cells": unique_cells,
        "cells_with_3plus_visits": visited_cells,
        "max_cell_visits": max_cell_visits,
        "graph": graph_stats,
        # NOTE: ``pred_locations`` are *not* stored in the world frame in these
        # baseline results (their range is ~3.3-6.0 while real positions span
        # ~0.5-8.5 and they do not correlate with the true trajectory). This
        # number is kept only for completeness and must not be read as an
        # execution error.
        "pred_err_raw": pred_err,
    }


def classify(rec: Dict[str, Any]) -> str:
    """Coarse, data-driven label for *why* an episode failed."""
    dmin = rec["d_min"]
    dfin = rec["d_final"]
    net_tail = rec["tail_net_disp"]
    eff = rec["path_efficiency"] or 0.0
    if dmin <= 1.0:
        return "near-miss (reached <=1.0 of goal, never inside 0.5)"
    if net_tail < 1.0 and dfin > 1.0:
        return "stuck (final 100 steps barely move)"
    if eff < 0.05:
        return "wandering (path spent, almost no net approach)"
    if dmin < 0.6 * rec["d_start"] and dfin > dmin + 1.5:
        return "approached then lost the goal (oscillation/backtrack)"
    return "never approached the goal"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--run-tag", default="hard_hwm")
    parser.add_argument("--shards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tasks-per-shard", type=int, default=5)
    parser.add_argument("--out", default="analysis/baseline_failures.json")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    prefix = "l2_d4rl_hard"
    shards = [int(s) for s in args.shards.split(",") if s.strip()]

    records: List[Dict[str, Any]] = []
    for shard in shards:
        shard_dir = run_root / f"{args.run_tag}_shard{shard}"
        result_path = shard_dir / f"planning_l2_mpc_result_{prefix}"
        trials_path = TRIALS_DIR / f"starts_targets_13_16_shard{shard}.pt"
        if not result_path.exists():
            print(f"[skip] missing {result_path}", file=sys.stderr)
            continue
        result = torch.load(result_path, map_location="cpu", weights_only=False)
        trials = torch.load(trials_path, map_location="cpu", weights_only=False)

        loc = as_xy(result.locations)  # (T, B, 2)
        goals = result.targets[:, :2].double()
        rewards = torch.stack([r[:5].double() for r in result.reward_history])  # (T-1, B)
        preds = None
        if getattr(result, "pred_locations", None):
            # predictions are (h, B, 2) with h >= 11 except for a few tail entries;
            # keep the 10-step-ahead position (index 10) only.
            preds = torch.stack(
                [p[10, :, :2].double() for p in result.pred_locations if p.shape[0] > 10]
            )  # (n, B, 2)


        n_envs = loc.shape[1]
        for env in range(n_envs):
            task_id = shard * args.tasks_per_shard + env
            layout = str(trials["map_layouts"][env])
            env_preds = preds[:, env, :] if preds is not None else None
            rec = analyse_env(
                loc[:, env, :],
                goals[env],
                rewards[:, env],
                env_preds,
                layout,
            )
            rec.update(
                {
                    "task_id": task_id,
                    "shard": shard,
                    "local_idx": env,
                    "block_dist": int(trials["block_dists"][env]),
                    "turns": int(trials["turns"][env]),
                    "layout": layout,
                }
            )
            records.append(rec)
        del result, preds
        print(f"[done] shard {shard}", file=sys.stderr)

    records.sort(key=lambda r: r["task_id"])

    fails = [r for r in records if not r["success"]]
    succ = [r for r in records if r["success"]]

    for r in fails:
        r["diagnosis"] = classify(r)

    def mean(values: List[float]) -> Optional[float]:
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    summary = {
        "num_tasks": len(records),
        "num_success": len(succ),
        "num_fail": len(fails),
        "success_rate": round(len(succ) / max(1, len(records)), 4),
        "failed_task_ids": [r["task_id"] for r in fails],
        "success_means": {
            "time_to_goal": mean([r["success_step"] for r in succ]),
            "block_dist": mean([r["block_dist"] for r in succ]),
            "turns": mean([r["turns"] for r in succ]),
            "path_len_to_goal": mean([r["path_len"] for r in succ]),
            "detour_ratio_vs_ideal": mean([r["detour_ratio_vs_ideal"] for r in succ]),
            "oscillation_cycles": mean([r["oscillation_cycles"] for r in succ]),
            "frozen_frac": mean([r["frozen_frac"] for r in succ]),
            "pred_err_raw": mean(
                [r["pred_err_raw"]["mean"] for r in succ if r["pred_err_raw"]]
            ),
        },        "fail_means": {
            "steps": mean([r["steps"] for r in fails]),
            "block_dist": mean([r["block_dist"] for r in fails]),
            "turns": mean([r["turns"] for r in fails]),
            "path_len": mean([r["path_len"] for r in fails]),
            "detour_ratio_vs_ideal": mean([r["detour_ratio_vs_ideal"] for r in fails]),
            "oscillation_cycles": mean([r["oscillation_cycles"] for r in fails]),
            "frozen_frac": mean([r["frozen_frac"] for r in fails]),
            "unique_cells": mean([r["unique_cells"] for r in fails]),
            "max_cell_visits": mean([r["max_cell_visits"] for r in fails]),
            "graph_min": mean([r["graph"]["min"] for r in fails if r["graph"]]),
            "graph_final": mean([r["graph"]["final"] for r in fails if r["graph"]]),
            "frac_more_than_2_blocks_from_goal": mean(
                [
                    r["graph"]["frac_more_than_2_blocks_from_goal"]
                    for r in fails
                    if r["graph"]
                ]
            ),
            "frac_went_farther_than_start": mean(
                [r["graph"]["frac_went_farther_than_start"] for r in fails if r["graph"]]
            ),
            "graph_transitions_toward": mean(
                [r["graph"]["cell_transitions_toward"] for r in fails if r["graph"]]
            ),
            "graph_transitions_away": mean(
                [r["graph"]["cell_transitions_away"] for r in fails if r["graph"]]
            ),
            "graph_delta_last_200": mean(
                [r["graph"]["delta_last_200"] for r in fails if r["graph"]]
            ),
            "pred_err_raw": mean(
                [r["pred_err_raw"]["mean"] for r in fails if r["pred_err_raw"]]
            ),
            "d_start": mean([r["d_start"] for r in fails]),
            "d_min": mean([r["d_min"] for r in fails]),
            "d_final": mean([r["d_final"] for r in fails]),
        },
        "diagnosis_counts": {},
        "tasks": records,
    }
    for r in fails:
        summary["diagnosis_counts"][r["diagnosis"]] = (
            summary["diagnosis_counts"].get(r["diagnosis"], 0) + 1
        )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    # ---------------- console report ----------------
    hdr = (
        f"{'tid':>3} {'ok':>2} {'blk':>3} {'tn':>2} {'t_goal':>6} {'d0':>6} {'dmin':>6} {'@':>4} "
        f"{'dfin':>6} {'path':>6} {'ideal':>6} {'det':>5} {'eff':>6} {'cyc':>3} {'frz':>5} {'cell':>9} {'pred':>5}"
    )
    print("\n=== all tasks ===")
    print(hdr)
    for r in records:
        print(
            f"{r['task_id']:>3} {int(r['success']):>2} {r['block_dist']:>3} {r['turns']:>2} "
            f"{str(r['success_step']):>6} {r['d_start']:>6.2f} {r['d_min']:>6.2f} {r['t_at_d_min']:>4} "
            f"{r['d_final']:>6.2f} {r['path_len']:>6.1f} {str(r['ideal_path_len']):>6} "
            f"{str(r['detour_ratio_vs_ideal']):>5} {str(r['path_efficiency']):>6} "
            f"{r['oscillation_cycles']:>3} {r['frozen_frac']:>5.2f} "
            f"{str(r['unique_cells']) + '/' + str(r['max_cell_visits']):>9} "
            f"{(r['pred_err_raw']['mean'] if r['pred_err_raw'] else float('nan')):>5.2f}"
        )

    print("\n=== failed tasks ===")
    for r in fails:
        print(
            f"\nTask {r['task_id']:>2} (shard {r['shard']}, block_dist={r['block_dist']}, turns={r['turns']})"
            f"  -> {r['diagnosis']}"
        )
        print(f"  start {r['start']} -> goal {r['goal']}  (start d={r['d_start']:.2f})")
        print(
            f"  d_min={r['d_min']:.2f} at step {r['t_at_d_min']}, d_final={r['d_final']:.2f}, "
            f"path={r['path_len']:.1f} (ideal {r['ideal_path_len']}), "
            f"detour x{r['detour_ratio_vs_ideal']}, oscillation cycles={r['oscillation_cycles']}"
        )
        print(
            f"  within(2.0/1.0/0.5) at steps "
            f"{r['first_within_2']}/{r['first_within_1']}/{r['first_within_05']}, "
            f"tail net disp={r['tail_net_disp']:.2f}, frozen={r['frozen_frac']:.2f}, "
            f"cells unique={r['unique_cells']} max_visits={r['max_cell_visits']} "
            f"(cells visited >2x: {r['cells_with_3plus_visits']})"
        )
        print("  distance trace:", r["d_trace"])
        if r["graph"]:
            g = r["graph"]
            print(
                f"  graph dist to goal: start {g['start']} -> min {g['min']} -> final {g['final']} "
                f"(max {g['max']}), >2 blocks {g['frac_more_than_2_blocks_from_goal'] * 100:.0f}% of time, "
                f"farther-than-start {g['frac_went_farther_than_start'] * 100:.0f}%, "
                f"cell moves toward/away {g['cell_transitions_toward']}/{g['cell_transitions_away']}, "
                f"last-200 delta {g['delta_last_200']:+d}"
            )
            print("  graph trace:", g["trace"])
        if r["pred_err_raw"]:
            print("  pred_locations error (NOT world frame, ignore):", r["pred_err_raw"])

    print("\n=== summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
