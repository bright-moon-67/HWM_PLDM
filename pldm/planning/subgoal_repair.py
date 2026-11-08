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

from pldm.planning.diagnostics import project_l1_encoding_to_l2
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


class ReachableSelection(NamedTuple):
    """Result of :func:`select_reachable_subgoal`."""

    subgoal_l2: torch.Tensor  # (B, D) projected candidate, L2 representation
    subgoal_l2_map: Optional[torch.Tensor]  # (B, C, H, W) same, spatial layout
    index: torch.Tensor  # (B,) candidate index per environment
    valid: torch.Tensor  # (B,) bool: a candidate satisfied the constraints
    dist: torch.Tensor  # (K, B) candidate -> original subgoal distance
    progress: torch.Tensor  # (K, B) initial_dist - dist


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
    mode: str = "reselection"
    # Step 5/6 extras (None when not applicable).
    high_level_cost_before: Optional[torch.Tensor] = None
    high_level_cost_selected: Optional[torch.Tensor] = None
    score_before: Optional[torch.Tensor] = None
    score_selected: Optional[torch.Tensor] = None
    lambda_used: Optional[torch.Tensor] = None
    reachable_valid: Optional[torch.Tensor] = None

    @property
    def n_triggered(self) -> int:
        return int(self.triggered.sum().item())

    @property
    def n_accepted(self) -> int:
        return int(self.accepted.sum().item())

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
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
            "repair_mode": self.mode,
        }
        for name in (
            "high_level_cost_before",
            "high_level_cost_selected",
            "score_before",
            "score_selected",
            "lambda_used",
            "reachable_valid",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


def build_reachable_candidates(
    low_planner: Any,
    current_state: Any,
    *,
    top_k: int = 8,
    plan_size: Optional[int] = None,
) -> ReachableCandidates:
    """Roll out the top-k cheapest low-level MPPI samples.

    ``MPPI._compute_total_cost_batch`` keeps both ``perturbed_action`` (the
    bounded, nominal-plus-noise action sequences that were actually rolled out
    during the search) and ``cost_total`` (their costs). Re-rolling the cheapest
    few of them therefore needs no change to the MPPI inner loop: the same action
    convention is used as in ``MPPIPlanner.plan`` when it builds ``pred_obs``.

    ``current_state`` is the ``BackboneOutput`` produced for the level-1 model
    (the same object the planner receives with ``repr_input=True``), or a plain
    ``(B, C, H, W)`` encoding tensor.

    Note that this is the reachable set **of the low-level world model under the
    current MPPI sampling budget** - not the true reachable set of the robot.
    """
    ctrls = getattr(low_planner, "ctrls", None)
    if not ctrls:
        raise RuntimeError("low_planner has no ctrls; build_reachable_candidates "
                           "expects an MPPIPlanner")

    if hasattr(current_state, "encodings"):
        state = current_state.encodings.detach()
        proprio = getattr(current_state, "proprio_component", None)
        location = getattr(current_state, "location_component", None)
        raw_location = getattr(current_state, "raw_locations", None)
    else:
        state = current_state.detach()
        proprio = location = raw_location = None

    batch_size = state.shape[0]
    if batch_size != len(ctrls):
        raise RuntimeError(
            f"batch mismatch: {batch_size} encodings vs {len(ctrls)} low-level ctrls"
        )

    per_env_actions: List[torch.Tensor] = []
    per_env_costs: List[torch.Tensor] = []
    for ctrl in ctrls:
        actions = getattr(ctrl, "perturbed_action", None)
        costs = getattr(ctrl, "cost_total", None)
        if actions is None or costs is None:
            raise RuntimeError(
                "no MPPI samples available; build_reachable_candidates must run "
                "after at least one MPPIPlanner.plan() call"
            )
        k = min(int(top_k), actions.shape[0])
        order = torch.argsort(costs)[:k]
        acts = actions[order]
        if plan_size is not None and acts.shape[1] > plan_size:
            acts = acts[:, :plan_size]
        per_env_actions.append(acts)
        per_env_costs.append(costs[order])

    # All environments must contribute the same number of candidates.
    k = min(a.shape[0] for a in per_env_actions)
    actions = torch.stack([a[:k] for a in per_env_actions], dim=1)  # (k, B, T, nu)
    costs = torch.stack([c[:k] for c in per_env_costs], dim=1)  # (k, B)

    # Roll the candidates out with the low-level world model.
    dynamics = low_planner.dynamics
    predictor = dynamics.model.predictor
    steps = actions.shape[2]
    flat_actions = actions.reshape(k * batch_size, steps, -1).permute(1, 0, 2)

    def _repeat(tensor):
        if tensor is None:
            return None
        return tensor.repeat_interleave(k, dim=0)

    with torch.no_grad():
        out = dynamics(
            state=predictor._prepare_ensemble_input(_repeat(state)),
            proprio=predictor._prepare_ensemble_input(_repeat(proprio)),
            location=predictor._prepare_ensemble_input(_repeat(location)),
            raw_location=predictor._prepare_ensemble_input(_repeat(raw_location)),
            action=flat_actions,
            only_return_last=True,
            flatten_output=False,
        )
    terminal = out.obs_component.detach().view(batch_size * k, *out.obs_component.shape[1:])
    terminal = terminal.view(k, batch_size, *terminal.shape[1:])
    return ReachableCandidates(states=terminal, actions=actions, costs=costs)


def select_reachable_subgoal(
    candidates: ReachableCandidates,
    original_goal: torch.Tensor,
    objective: Any,
    *,
    current_dist: Optional[torch.Tensor] = None,
    risk_weight: float = 0.0,
    min_progress: float = 0.0,
    subgoal_shape: Optional[torch.Size] = None,
) -> ReachableSelection:
    """Pick a replacement subgoal from the candidate terminal states.

    ``original_goal`` is the high-level subgoal ``l2_result.pred_obs[i]`` in the
    L2 representation; the candidate terminals are projected into the same space
    with the exact mapping the MPC objective uses, so the distances are
    comparable.

    A candidate is admissible only if it makes **progress** towards the original
    subgoal (``current_dist - dist > min_progress``). Selecting the endpoint of a
    plan that simply drives into a wall is therefore impossible: such a candidate
    ends further from the subgoal than the current state.

    Linear interpolation between subgoal and endpoint is deliberately not used -
    the interpolated latent need not correspond to a reachable maze state.
    """
    from pldm.models.utils import flatten_conv_output

    if candidates.states.dim() < 5:
        raise ValueError(
            f"expected (K, B, C, H, W) candidate states, got "
            f"{tuple(candidates.states.shape)}"
        )
    k, batch_size = candidates.states.shape[0], candidates.states.shape[1]

    z_high = flatten_conv_output(original_goal.detach())  # (B, D)
    cand_l2 = project_l1_encoding_to_l2(
        candidates.states.detach(), objective
    )  # (K, B, D)

    dist = (cand_l2 - z_high.unsqueeze(0)).pow(2).mean(dim=-1)  # (K, B)
    if current_dist is None:
        # Fall back to the worst candidate so that "progress" stays meaningful.
        current_dist = dist.max(dim=0).values
    current_dist = current_dist.detach().view(1, -1).expand(k, batch_size)
    progress = current_dist - dist

    score = dist.clone()
    if risk_weight:
        costs = candidates.costs.detach().to(score.device)
        cmin = costs.min(dim=0, keepdim=True).values
        cmax = costs.max(dim=0, keepdim=True).values
        norm_cost = (costs - cmin) / (cmax - cmin).clamp_min(1e-8)
        score = score + float(risk_weight) * norm_cost

    admissible = progress > float(min_progress)
    masked = score.masked_fill(~admissible, float("inf"))
    index = masked.argmin(dim=0)  # (B,)
    valid = admissible.gather(0, index.view(1, -1)).squeeze(0)

    selected = cand_l2.gather(
        0, index.view(1, -1, 1).expand(1, batch_size, cand_l2.shape[-1])
    ).squeeze(0)  # (B, D)

    selected_map = None
    channels = getattr(objective.model.backbone, "output_obs_dim", None)
    if subgoal_shape is not None:
        selected_map = selected.view(batch_size, *subgoal_shape)
    elif channels is not None:
        selected_map = selected.view(batch_size, *tuple(channels))

    return ReachableSelection(
        subgoal_l2=selected,
        subgoal_l2_map=selected_map,
        index=index,
        valid=valid,
        dist=dist,
        progress=progress,
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
    lambda_vector: Optional[torch.Tensor] = None,
) -> Tuple[PlanningResult, PlanningResult, Optional[RepairInfo]]:
    """Re-search the L2 plan and keep the most low-level-reachable candidate.

    ``metrics_fn(l1_result, l2_result)`` returns the same cross-level metrics as
    the diagnostic path (must contain ``cross_mse``).

    Selection (step 5) minimises the *normalised* combination

        score = J_H_norm + lambda * C_L_norm

    where ``J_H`` is the high-level plan cost (distance of the L2 plan terminal to
    the L2 goal) and ``C_L`` is the low-level reachability cost (cross-level gap
    of the L1 plan towards the candidate subgoal). Both terms live on different
    scales, so each is min-max normalised across the candidates of the same
    environment before they are added - adding raw costs would be meaningless.

    ``lambda_vector`` (``(B,)``) is the per-environment weight, normally produced
    by :class:`pldm.planning.exec_feedback.ExecutionFeedback`. When ``None`` the
    constant ``cfg.high_level_cost_weight`` is used.

    Acceptance additionally requires that the selected candidate does not make
    the low-level reachability worse than the plan we already have
    (``gap_sel <= gap_before + cfg.repair_gap_tolerance``), so a better global
    cost can never justify a clearly unreachable subgoal.
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

    batch_size = gap_before.numel()
    min_improvement = float(getattr(cfg, "repair_min_improvement", 0.0))
    gap_tolerance = float(getattr(cfg, "repair_gap_tolerance", 0.0))
    base_weight = float(getattr(cfg, "high_level_cost_weight", 0.5))
    if lambda_vector is None:
        lam = torch.full(
            (batch_size,),
            base_weight,
            dtype=gap_before.dtype,
            device=gap_before.device,
        )
    else:
        lam = lambda_vector.detach().to(
            dtype=gap_before.dtype, device=gap_before.device
        ).view(-1)
        if lam.numel() != batch_size:
            lam = lam.expand(batch_size).clone()

    # Candidate 0 is the plan we already have.
    # NOTE: the L1 target is left pointing at the *last* candidate's subgoal; it
    # is not used again inside this planning call and is reset at the next
    # replan, so it cannot leak into execution.
    candidate_gaps: List[torch.Tensor] = [gap_before]
    candidate_l1: List[PlanningResult] = [l1_first_result]
    candidate_l2: List[PlanningResult] = [l2_first_result]
    candidate_jh: List[Optional[torch.Tensor]] = [
        _high_level_goal_distance(h_planner.l2_planner, l2_first_result)
    ]
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
            candidate_jh.append(
                _high_level_goal_distance(h_planner.l2_planner, l2_candidate)
            )
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
    scores = _combined_scores(gaps, candidate_jh, lam)  # (K, B)

    masked = scores.clone()
    masked[:, ~triggered] = float("inf")
    selected = masked.argmin(dim=0)  # (B,)
    selected_gap = gaps.gather(0, selected.view(1, -1)).squeeze(0)
    selected_score = scores.gather(0, selected.view(1, -1)).squeeze(0)
    score_before = scores[0]

    reachable_ok = selected_gap <= (gap_before + gap_tolerance)
    accepted = (
        triggered
        & reachable_ok
        & ((gap_before - selected_gap) > min_improvement)
        & (selected_score < score_before)
    )
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
        mode="reselection",
        high_level_cost_before=(
            candidate_jh[0] if candidate_jh[0] is not None else None
        ),
        high_level_cost_selected=(
            None
            if any(j is None for j in candidate_jh)
            else torch.stack(candidate_jh, dim=0)
            .gather(0, selected.view(1, -1))
            .squeeze(0)
        ),
        score_before=score_before,
        score_selected=torch.where(accepted, selected_score, score_before),
        lambda_used=lam,
    )


# --------------------------------------------------------------------------- #
# helpers shared by the two repair branches
# --------------------------------------------------------------------------- #
def _high_level_goal_distance(
    l2_planner: Any, l2_result: PlanningResult
) -> Optional[torch.Tensor]:
    """Distance of the L2 plan terminal to the L2 goal, i.e. ``J_H``.

    Uses the planner's own objective target, so it is exactly the quantity the
    high-level MPC minimises (up to the running-cost weighting).
    """
    from pldm.models.utils import flatten_conv_output

    objective = getattr(l2_planner, "objective", None)
    target = getattr(objective, "target_enc", None) if objective is not None else None
    pred_obs = getattr(l2_result, "pred_obs", None)
    if target is None or pred_obs is None:
        return None
    pred = flatten_conv_output(pred_obs[-1].detach())
    tgt = target.detach()
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0)
    if tgt.shape[0] == 1 and pred.shape[0] > 1:
        tgt = tgt.expand(pred.shape[0], -1)
    if tgt.shape != pred.shape:
        return None
    return (pred - tgt).pow(2).mean(dim=1)


def _minmax_per_env(values: torch.Tensor) -> torch.Tensor:
    """Min-max normalise ``(K, B)`` values along the candidate dimension."""
    vmin = values.min(dim=0, keepdim=True).values
    vmax = values.max(dim=0, keepdim=True).values
    return (values - vmin) / (vmax - vmin).clamp_min(1e-8)


def _combined_scores(
    gaps: torch.Tensor,
    high_level_costs: List[Optional[torch.Tensor]],
    lam: torch.Tensor,
) -> torch.Tensor:
    """``score = J_H_norm + lambda * C_L_norm`` with both terms normalised."""
    low = _minmax_per_env(gaps)
    if any(cost is None for cost in high_level_costs):
        return low
    high = _minmax_per_env(torch.stack(high_level_costs, dim=0))
    lam = lam.to(device=low.device, dtype=low.dtype).view(1, -1)
    return high + lam * low


def plan_with_reachable_projection(
    *,
    h_planner: Any,
    backbone_output: Any,
    l1_plan_kwargs: Dict[str, Any],
    l1_first_result: PlanningResult,
    l1_first_metrics: Dict[str, torch.Tensor],
    l2_result: PlanningResult,
    metrics_fn: Callable[[PlanningResult, PlanningResult], Dict[str, torch.Tensor]],
    objective: Any,
    cfg: Any,
    subgoal_index: int = 1,
) -> Tuple[PlanningResult, Optional[RepairInfo]]:
    """Step 4: project the high-level subgoal onto low-level reachable states.

    Pipeline:

    1. take the ``top_k`` cheapest MPPI samples of the *current* low-level plan
       and roll them out with the L1 world model (``build_reachable_candidates``);
    2. keep only candidates that are closer to the original subgoal than the
       current state is, and pick the best one (``select_reachable_subgoal``);
    3. re-plan the low level towards that candidate and accept per environment
       only if the resulting terminal state is closer to the *original* subgoal
       than the terminal state of the original plan.

    Step 3 is what keeps this from merely re-labelling a bad plan: a projection
    that only reaches a reachable-but-useless state is rejected, because the
    repaired plan would not get closer to the high-level subgoal.
    """
    gap_before = l1_first_metrics["cross_mse"].detach()
    threshold = float(getattr(cfg, "retry_gap_threshold", 0.0))
    triggered = should_repair(
        cross_mse=gap_before,
        gap_threshold=threshold,
        subgoal_index=subgoal_index,
        enabled=bool(getattr(cfg, "repair_enabled", False)),
    )
    if triggered.numel() == 0 or not bool(triggered.any()):
        return l1_first_result, None

    top_k = int(getattr(cfg, "repair_top_k", 8) or 0)
    if top_k <= 0:
        return l1_first_result, None

    min_improvement = float(getattr(cfg, "repair_min_improvement", 0.0))
    original_subgoal = l2_result.pred_obs[subgoal_index].detach()
    snapshot = snapshot_low_level_state(h_planner.l1_planner)

    start = time.time()
    try:
        candidates = build_reachable_candidates(
            h_planner.l1_planner,
            backbone_output,
            top_k=top_k,
            plan_size=int(l1_plan_kwargs.get("plan_size") or 0) or None,
        )
        selection = select_reachable_subgoal(
            candidates,
            original_subgoal,
            objective,
            current_dist=l1_first_metrics.get("initial_dist"),
            risk_weight=float(getattr(cfg, "repair_risk_weight", 0.0)),
            min_progress=float(getattr(cfg, "repair_min_progress", 0.0)),
            subgoal_shape=torch.Size(original_subgoal.shape[1:]),
        )
        if not bool(selection.valid.any()):
            restore_low_level_state(h_planner.l1_planner, snapshot, restore_rng=True)
            h_planner.l1_planner.reset_targets(original_subgoal, repr_input=True)
            return l1_first_result, None

        h_planner.l1_planner.reset_targets(selection.subgoal_l2_map, repr_input=True)
        repaired_result = h_planner.l1_planner.plan(**l1_plan_kwargs)
        repaired_metrics = metrics_fn(repaired_result, l2_result)
        wall_time_s = time.time() - start
    except Exception:
        restore_low_level_state(h_planner.l1_planner, snapshot, restore_rng=True)
        h_planner.l1_planner.reset_targets(original_subgoal, repr_input=True)
        if bool(getattr(cfg, "repair_raise_errors", False)):
            raise
        return l1_first_result, None
    finally:
        # The target must not stay on the projected candidate for the next
        # replan (it is reset there anyway, but leaving it consistent is cheap).
        try:
            h_planner.l1_planner.reset_targets(original_subgoal, repr_input=True)
        except Exception:  # pragma: no cover - defensive
            pass

    repaired_gap = repaired_metrics["cross_mse"].detach()
    accepted = triggered & ((gap_before - repaired_gap) > min_improvement)
    accepted = accepted.to(dtype=torch.bool)

    if bool(accepted.any()):
        merged = blend_planning_results(l1_first_result, repaired_result, accepted)
    else:
        merged = l1_first_result
    # Keep the repaired warm start only where it was accepted.
    restore_low_level_state(h_planner.l1_planner, snapshot, env_mask=~accepted)

    return merged, RepairInfo(
        triggered=triggered,
        accepted=accepted,
        gap_before=gap_before,
        gap_after=repaired_gap,
        gap_selected=torch.where(accepted, repaired_gap, gap_before),
        selected_candidate=torch.where(
            accepted, selection.index, torch.full_like(selection.index, -1)
        ),
        n_candidates=int(candidates.states.shape[0]),
        wall_time_s=wall_time_s,
        mode="projection",
        reachable_valid=selection.valid,
    )


