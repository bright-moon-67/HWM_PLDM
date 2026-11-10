#!/usr/bin/env python3
"""Stage-2 research gate G1: is candidate selection a real bottleneck?

Reads the table produced by ``scripts/collect_plan_candidates.py`` and reports,
per observation horizon:

* **Oracle Candidate Gain** -- ``max_i R_H(P_i) - R_H(P_baseline)``: how much a
  perfect ranker could have gained on the candidate set that was actually
  sampled.  A gain close to zero means better ranking cannot help; the
  bottleneck is candidate *generation*.
* **Candidate Ranking Accuracy** -- Kendall tau-b, pairwise accuracy and
  best-candidate hit rate between the model scores (``J_H``, lower is better)
  and the realised returns (``R_H``, higher is better).
* **Selection utility** -- regret of the model's own pick versus the plan the
  original planner chose, which separates "good candidates, bad ranking" from
  "bad candidates".

Example::

    python scripts/evaluate_candidate_oracle.py \
        --candidates results/candidates_task7_candidates.jsonl \
        --horizons 40 80 \
        --out results/oracle_task7

``--demo`` generates synthetic decisions with a known signal so the pipeline can
be smoke-tested without a planning run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from pldm.planning.candidate_evaluator import (  # noqa: E402
    bootstrap_mean_ci,
    candidate_ranking_summary,
    group_summary,
    groups_to_matrices,
    label_by_task,
    nanmedian,
    oracle_candidate_gain,
    read_candidate_groups,
    selection_utility,
)


def _demo_groups(
    n_decisions: int = 60, n_candidates: int = 3, seed: int = 0
) -> List[Dict[str, Any]]:
    """Synthetic decisions with a deliberately imperfect model ranking."""
    generator = torch.Generator().manual_seed(seed)
    groups: List[Dict[str, Any]] = []
    for decision in range(n_decisions):
        # True quality of each candidate, in graph steps.
        quality = torch.randn(n_candidates, generator=generator)
        # The model observes quality with noise, so it is right most of the
        # time but misses the best candidate in a sizeable minority.
        observed = quality + 0.9 * torch.randn(n_candidates, generator=generator)
        # Original planner tends to prefer high model score; make candidate 0
        # its pick most of the time.
        observed[0] -= 0.3
        horizon = 40
        candidates = []
        for index in range(n_candidates):
            candidates.append(
                {
                    "candidate_id": index,
                    "source": "hwm" if index == 0 else "resample",
                    "selected": index == 0,
                    # Model scores are *costs* (lower is better), so flip the
                    # sign of the observed quality.
                    "model_score": float(-observed[index].item()),
                    "real_return": float(4.0 * quality[index].item()),
                }
            )
        groups.append(
            {
                "task_id": decision % 5,
                "decision_id": decision,
                "plan_id": decision,
                "step": decision * 4,
                "horizon": horizon,
                "candidates": candidates,
            }
        )
    return groups


def _interpret(summary: Dict[str, float], utility: Dict[str, float]) -> str:
    """Short, descriptive reading of the G1 numbers (no hard thresholds)."""
    gain = summary.get("oracle_gain_mean", float("nan"))
    low = summary.get("oracle_gain_lo", float("nan"))
    positive = summary.get("oracle_gain_positive_rate", float("nan"))
    accuracy = summary.get("pairwise_accuracy_mean", float("nan"))
    regret = utility.get("model_pick_regret_mean", float("nan"))
    if not all(
        isinstance(v, float) and v == v for v in (gain, low, positive, accuracy)
    ):
        return "insufficient labelled data for an interpretation"
    if low > 0 and positive >= 0.2:
        return (
            "candidate selection is a plausible bottleneck: better candidates "
            "existed and the model ranking missed them (route A gets support)"
        )
    if gain <= 0:
        return (
            "no candidate in the sampled set beat the original plan: the "
            "bottleneck is candidate generation, not ranking"
        )
    if regret > 0 and accuracy < 0.7:
        return "ranking is weak but gains are small: increase candidate diversity"
    return "mixed evidence: report distributions before deciding"


def analyse(
    groups: List[Dict[str, Any]],
    *,
    horizons: List[int],
    baseline_index: int,
    n_boot: int,
    seed: int,
    task_groups: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"horizons": {}}
    for horizon in horizons:
        scores, returns, task_ids, _ = groups_to_matrices(groups, horizon=horizon)
        if scores.numel() == 0:
            result["horizons"][str(horizon)] = {"n_groups": 0}
            continue
        labelled = torch.isfinite(returns).any(dim=0)
        scores = scores[:, labelled]
        returns = returns[:, labelled]
        task_ids = [tid for tid, keep in zip(task_ids, labelled.tolist()) if keep]
        if scores.shape[1] == 0:
            result["horizons"][str(horizon)] = {"n_groups": 0}
            continue

        summary = candidate_ranking_summary(
            scores,
            returns,
            baseline_index=baseline_index,
            n_boot=n_boot,
            seed=seed,
        )
        utility = selection_utility(scores, returns, baseline_index=baseline_index)
        gain = oracle_candidate_gain(returns, baseline_index=baseline_index)
        gain_stats = bootstrap_mean_ci(gain, n_boot=n_boot, seed=seed)

        if task_groups:
            labels = [
                str(task_groups.get(str(task_id), "unknown")) for task_id in task_ids
            ]
        else:
            labels = label_by_task(task_ids)
        by_group = group_summary(
            labels,
            scores,
            returns,
            baseline_index=baseline_index,
            n_boot=n_boot,
            seed=seed,
        )

        result["horizons"][str(horizon)] = {
            "n_groups": int(scores.shape[1]),
            "n_candidates": int(scores.shape[0]),
            "overall": summary,
            "selection_utility": utility,
            "gain_median": nanmedian(gain),
            "gain_ci": gain_stats,
            "by_group": by_group,
            "interpretation": _interpret(summary, utility),
        }
    return result


def _print_table(result: Dict[str, Any]) -> None:
    print(
        f"{'horizon':>8} {'decisions':>10} {'oracle_gain':>12} {'95% CI':>20} "
        f"{'tau':>8} {'pair_acc':>9} {'hit':>6} {'regret':>8}"
    )
    for horizon, payload in result["horizons"].items():
        if not payload.get("n_groups"):
            print(f"{horizon:>8} {0:>10}   (no labelled decisions)")
            continue
        overall = payload["overall"]
        utility = payload["selection_utility"]
        ci = (
            f"[{payload['gain_ci']['lo']:.3f}, "
            f"{payload['gain_ci']['hi']:.3f}]"
        )
        print(
            f"{horizon:>8} {payload['n_groups']:>10} "
            f"{overall['oracle_gain_mean']:>12.3f} {ci:>20} "
            f"{overall['kendall_tau_mean']:>8.3f} "
            f"{overall['pairwise_accuracy_mean']:>9.3f} "
            f"{overall['best_candidate_hit_rate']:>6.3f} "
            f"{utility['model_pick_regret_mean']:>8.3f}"
        )
    for horizon, payload in result["horizons"].items():
        if payload.get("n_groups"):
            print(f"\nhorizon {horizon}: {payload['interpretation']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=None, help="candidate table JSONL")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="horizons to analyse (default: every horizon in the file)",
    )
    parser.add_argument("--baseline-id", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task-groups",
        default=None,
        help='optional JSON mapping task id -> group label, e.g. {"4": "failed"}',
    )
    parser.add_argument("--out", default=None, help="output JSON prefix")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="analyse synthetic decisions instead of a collected table",
    )
    args = parser.parse_args()

    if args.demo:
        groups = _demo_groups()
    else:
        if not args.candidates:
            parser.error("--candidates is required unless --demo is given")
        groups = read_candidate_groups(args.candidates)

    available = sorted({int(g["horizon"]) for g in groups})
    horizons = args.horizons or available
    task_groups = None
    if args.task_groups:
        task_groups = json.loads(Path(args.task_groups).read_text(encoding="utf-8"))

    result = analyse(
        groups,
        horizons=horizons,
        baseline_index=args.baseline_id,
        n_boot=args.bootstrap,
        seed=args.seed,
        task_groups=task_groups,
    )
    result["source"] = "demo" if args.demo else str(args.candidates)
    _print_table(result)

    if args.out:
        out_prefix = Path(args.out)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_prefix.with_name(out_prefix.name + "_oracle.json")
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
