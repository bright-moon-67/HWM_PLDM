"""High-level re-search conditioned on low-level reachability (Step 4/5).

Motivation (review-2 section 9): when the low-level planner keeps missing the
high-level subgoal, there are two different failure stories:

1. *search failure* - the low-level model is right but MPPI did not find a plan
   that reaches the subgoal (addressed by Step 3's conditional retry);
2. *model optimism* - the subgoal is planned under an over-optimistic L2 model
   and cannot be executed at all.

Story 2 cannot be fixed by re-sampling the low level; the high-level macro action
has to change. The cheapest way to test that hypothesis without inventing a new
subgoal generator is to **re-search the L2 plan and rank the candidates by their
low-level reachability**:

    for each of ``n_candidates`` fresh L2 plans:
        take its near subgoal  ->  run one L1 plan towards it
        record the predicted cross-level gap of that L1 plan

    keep, per environment, the candidate whose near subgoal the low-level planner
    is predicted to reach most closely.

This module implements exactly that. It is strictly opt-in
(``exec_repair.enabled and exec_repair.repair_enabled``) because it costs
``n_candidates`` extra L2 plans plus ``n_candidates`` extra L1 plans per decision,
and because review-2 is explicit that the two failure stories are hypotheses,
not established facts.

Still unimplemented on purpose: ``build_reachable_candidates`` /
``select_reachable_subgoal`` (top-k MPPI rollout candidates). Re-searching the L2
plan is the cheaper experiment and answers the same question first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch

from pldm.planning.level_retry import (
    blend_planning_results,
    restore_low_level_state,
    snapshot_low_level_state,
)
from pldm.planning.planners.planner import PlanningResult


class ReachableCandidates(NamedTuple):
    """Candidate terminal states of the low-level planner."""

    states: torch.Tensor  # (K, B, ...) predicted terminal L1 states
    actions: torch.Tensor  # (K, B, T, action_dim)
    costs: torch.Tensor  # (K, B) lower is better


@dataclass
class RepairInfo:
    """Per-decision statistics for the high-level re-search branch."""

    triggered: torch.Tensor
    accepted: torch.Tensor
    gap_before: torch.Tensor
    gap_after: torch.Tensor
    gap_selected: torch.Tensor
    selected_candidate: torch.Tensor  # (B,) index into the candidate list, -1 = keep
    n_candidates: int
    wall_time_s: float

    @property
    def n_triggered(self) -> int:
        return int(self.triggered.sum().item())

    @property
    def n_accepted(self) -> int:
        return int(self.accepted.sum().item())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "accepted": self.accepted,
            "gap_before": self.gap_before,
            "gap_after": self.gap_after,
            "gap_selected": self.gap_selected,
            "selected_candidate": self.selected_candidate,
            "n_candidates": self.n_candidates,
            "wall_ms": self.wall_time_s * 1000.0,
            "n_triggered": self.n_triggered,
            "n_accepted": self.n_accepted,
        }


def build_reachable_candidates(
    low_planner: Any,
    current_state: torch.Tensor,
    *,
    top_k: int = 8,
    plan_size: Optional[int] = None,
) -> ReachableCandidates:
    """Roll out the top-k cheapest low-level MPPI samples.

    Not implemented yet: the L2 re-search branch below already tests whether
    changing the *macro action* helps, which is the prerequisite question.
    """
    raise NotImplementedError(
        "top-k MPPI rollout candidates are not implemented yet; use "
        "plan_with_high_level_reselection instead."
    )


def select_reachable_subgoal(
    candidates: ReachableCandidates,
    original_goal: torch.Tensor,
    *,
    current_state: Optional[torch.Tensor] = None,
    risk: Optional[torch.Tensor] = None,
    risk_weight: float = 0.0,
) -> torch.Tensor:
    """Pick a replacement subgoal from the candidate terminal states.

    Linear interpolation between the subgoal and a low-level endpoint is
    intentionally **not** used: the interpolated latent need not correspond to a
    reachable maze state.
    """
    raise NotImplementedError(
        "reachable subgoal selection is not implemented yet; use "
        "plan_with_high_level_reselection instead."
    )


def should_repair(
    *,
    cross_mse: Optional[torch.Tensor],
    gap_threshold: float,
    subgoal_index: int,
    enabled: bool = True,
) -> torch.Tensor:
    """Per-environment trigger for the repair branch (Step 4).

    The repair branch only makes sense when a subgoal has been predicted to be
    unreachable *after* the low-level retry already had its chance, so the caller
    passes the post-retry gap.
    """
    if cross_mse is None:
        return torch.zeros(0, dtype=torch.bool)
    if not enabled or gap_threshold <= 0.0 or subgoal_index < 1:
        return torch.zeros_like(cross_mse, dtype=torch.bool)
    return (cross_mse > gap_threshold).to(dtype=torch.bool)


def plan_with_high_level_reselection(
    *,
    h_planner: Any,
    l2_plan_kwargs: Dict[str, Any],
    l1_plan_kwargs: Dict[str, Any],
    l1_first_result: PlanningResult,
    l1_first_metrics: Dict[str, torch.Tensor],
    l2_first_result: PlanningResult,
    metrics_fn: Callable[[PlanningResult, PlanningResult], Dict[str, torch.Tensor]],
    cfg: Any,
    subgoal_index: int = 1,
) -> Tuple[PlanningResult, PlanningResult, Optional[RepairInfo]]:
    """Re-search the L2 plan and keep the most low-level-reachable candidate.

    ``metrics_fn(l1_result, l2_result)`` returns the same cross-level metrics as
    the diagnostic path (must contain ``cross_mse``). Nothing changes for
    environments that are not triggered, or whose best candidate is not better
    than the plan already available.
    """
    gap_before = l1_first_metrics["cross_mse"].detach()
    n_candidates = int(getattr(cfg, "high_resample_count", 2) or 0)
    threshold = float(getattr(cfg, "retry_gap_threshold", 0.0))

    triggered = should_repair(
        cross_mse=gap_before,
        gap_threshold=threshold,
        subgoal_index=subgoal_index,
        enabled=bool(getattr(cfg, "repair_enabled", False)) and n_candidates > 0,
    )
    if triggered.numel() == 0 or not bool(triggered.any()):
        return l1_first_result, l2_first_result, None

    min_improvement = float(getattr(cfg, "repair_min_improvement", 0.0))

    # Candidate 0 is the plan we already have.
    # NOTE: the L1 target is left pointing at the *last* candidate's subgoal; it
    # is not used again inside this planning call and is reset at the next
    # replan, so it cannot leak into execution.
    candidate_gaps: List[torch.Tensor] = [gap_before]
    candidate_l1: List[PlanningResult] = [l1_first_result]
    candidate_l2: List[PlanningResult] = [l2_first_result]
    snapshots: List[Dict[str, Any]] = [
        {
            "l1": snapshot_low_level_state(h_planner.l1_planner),
            "l2": snapshot_low_level_state(h_planner.l2_planner),
        }
    ]

    start = time.time()
    try:
        for _ in range(n_candidates):
            l2_candidate = h_planner.l2_planner.plan(**l2_plan_kwargs)
            h_planner.l1_planner.reset_targets(
                l2_candidate.pred_obs[subgoal_index].detach(), repr_input=True
            )
            l1_candidate = h_planner.l1_planner.plan(**l1_plan_kwargs)
            candidate_metrics = metrics_fn(l1_candidate, l2_candidate)

            candidate_gaps.append(candidate_metrics["cross_mse"].detach())
            candidate_l1.append(l1_candidate)
            candidate_l2.append(l2_candidate)
            snapshots.append(
                {
                    "l1": snapshot_low_level_state(h_planner.l1_planner),
                    "l2": snapshot_low_level_state(h_planner.l2_planner),
                }
            )
        wall_time_s = time.time() - start
    except Exception:
        # Never leave a half-updated hierarchy behind.
        restore_low_level_state(
            h_planner.l1_planner, snapshots[0]["l1"], restore_rng=True
        )
        restore_low_level_state(
            h_planner.l2_planner, snapshots[0]["l2"], restore_rng=True
        )
        raise

    gaps = torch.stack(candidate_gaps, dim=0)  # (K, B)
    masked = gaps.clone()
    masked[:, ~triggered] = float("inf")
    selected = masked.argmin(dim=0)  # (B,)
    selected_gap = gaps.gather(0, selected.view(1, -1)).squeeze(0)

    accepted = triggered & ((gap_before - selected_gap) > min_improvement)
    accepted = accepted.to(dtype=torch.bool)
    selected = torch.where(accepted, selected, torch.zeros_like(selected))

    merged_l1 = candidate_l1[0]
    merged_l2 = candidate_l2[0]
    for index in range(1, len(candidate_l1)):
        mask = accepted & (selected == index)
        if not bool(mask.any()):
            continue
        merged_l1 = blend_planning_results(merged_l1, candidate_l1[index], mask)
        merged_l2 = blend_planning_results(merged_l2, candidate_l2[index], mask)

    # Restore the warm starts of the candidate that was actually chosen (or the
    # original one for rejected / non-triggered environments).
    for index, snapshot in enumerate(snapshots):
        mask = (selected == index) & accepted if index > 0 else ~accepted
        if not bool(mask.any()):
            continue
        restore_low_level_state(h_planner.l1_planner, snapshot["l1"], env_mask=mask)
        restore_low_level_state(h_planner.l2_planner, snapshot["l2"], env_mask=mask)

    return merged_l1, merged_l2, RepairInfo(
        triggered=triggered,
        accepted=accepted,
        gap_before=gap_before,
        gap_after=selected_gap,
        gap_selected=torch.where(accepted, selected_gap, gap_before),
        selected_candidate=torch.where(
            accepted, selected, torch.full_like(selected, -1)
        ),
        n_candidates=len(candidate_l1) - 1,
        wall_time_s=wall_time_s,
    )

