"""Controller-reachable subgoal repair (Step 4 scaffold).

Nothing here is called by the planning loop in Step 0. The module exists so the
interfaces of Step 4 can be reviewed before the algorithm is implemented.

Contract to implement in Step 4:

* candidates come from the **low-level** world model only (top-k cheapest MPPI
  rollouts plus the optimised action sequence);
* a candidate is accepted only if it (a) makes progress towards the original
  high-level subgoal and (b) is predicted reachable within the current MPPI
  budget;
* ``TwoLvlPlanner`` must restore the original target **and** the MPPI warm
  start (``ctrl.U``) whenever a repaired plan is rejected.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import torch


class ReachableCandidates(NamedTuple):
    """Candidate terminal states of the low-level planner."""

    states: torch.Tensor  # (K, B, ...) predicted terminal L1 states
    actions: torch.Tensor  # (K, B, T, action_dim)
    costs: torch.Tensor  # (K, B) lower is better


def build_reachable_candidates(
    low_planner: Any,
    current_state: torch.Tensor,
    *,
    top_k: int = 8,
    plan_size: Optional[int] = None,
) -> ReachableCandidates:
    """Roll out the top-k cheapest low-level MPPI samples.

    The low-level MPPI implementation keeps sampled actions and their costs but
    not the sampled trajectories, so the selected actions have to be rolled out
    once more with the level-1 world model. Actions must use the same
    normalisation as the actions actually sent to the environment.
    """
    raise NotImplementedError(
        "Step 4: controller-reachable subgoal candidates are not implemented yet."
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

    The first version should score candidates by their distance to the original
    subgoal plus an optional empirical risk term, while rejecting candidates
    that make no predicted progress. Linear interpolation between the subgoal
    and a low-level endpoint is intentionally **not** used: the interpolated
    latent need not correspond to a reachable maze state.
    """
    raise NotImplementedError(
        "Step 4: reachable subgoal selection is not implemented yet."
    )


def should_repair(
    *,
    retry_exhausted: torch.Tensor,
    cross_mse: Optional[torch.Tensor] = None,
    gap: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-environment trigger for the repair branch (Step 4)."""
    raise NotImplementedError(
        "Step 4: the repair trigger is not implemented yet."
    )
