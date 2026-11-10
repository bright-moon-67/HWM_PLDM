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
    restore_rng_state,
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
    # Spread of the cross-level gap across the freshly sampled candidates. If
    # this is ~0 the candidates are essentially identical (chained warm start or
    # degenerate search), which makes extra resampling pointless.
    candidate_gap_spread: Optional[torch.Tensor] = None
    candidate_jh_spread: Optional[torch.Tensor] = None
    # Distance between this repair target and the previous one (review 4, 4.3):
    # large values on consecutive repairs mean the subgoal is oscillating.
    subgoal_shift: Optional[torch.Tensor] = None
    # ECCHP stage 0: which acceptance rule was applied, and whether the
    # selected candidate would also have passed the legacy "the gap must
    # improve" rule. Keeping both lets an analysis separate candidates that
    # are only good under the combined score (e.g. a necessary detour).
    acceptance_rule: str = "legacy_gap"
    gap_improved: Optional[torch.Tensor] = None
    # ECCHP stage 2: per-candidate evidence (model scores + the L1 action chunk
    # that would have been executed). Kept in-process only - it contains action
    # tensors, so it is deliberately excluded from :meth:`as_dict`.
    candidate_evidence: Optional[List[Dict[str, Any]]] = None

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
            "candidate_gap_spread",
            "candidate_jh_spread",
            "subgoal_shift",
            "gap_improved",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        payload["acceptance_rule"] = self.acceptance_rule
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
    # ``reshape`` below flattens in candidate-major order:
    #   candidate0-env0, candidate0-env1, candidate1-env0, candidate1-env1, ...
    # so the conditioning tensors must be expanded the same way (repeat, not
    # repeat_interleave, which would be environment-major and mis-pair states
    # with actions as soon as batch_size > 1).
    flat_actions = actions.reshape(k * batch_size, steps, -1).permute(1, 0, 2)

    def _repeat(tensor):
        if tensor is None:
            return None
        return tensor.repeat(k, *([1] * (tensor.dim() - 1)))

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
    terminal = out.obs_component.detach().view(
        batch_size * k, *out.obs_component.shape[1:]
    )
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



def _repair_threshold(cfg: Any) -> float:
    """Gap threshold that gates the repair branch.

    ``repair_gap_threshold`` is the dedicated knob (review 3, step 3); a value
    <= 0 falls back to ``retry_gap_threshold`` for backwards compatibility.
    """
    threshold = float(getattr(cfg, "repair_gap_threshold", 0.0) or 0.0)
    if threshold <= 0.0:
        threshold = float(getattr(cfg, "retry_gap_threshold", 0.0) or 0.0)
    return threshold


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
    trigger_mask: Optional[torch.Tensor] = None,
    min_improvement: Any = None,
) -> Tuple[PlanningResult, PlanningResult, Optional[RepairInfo]]:
    """Re-search the L2 plan and keep the most low-level-reachable candidate.

    ``metrics_fn(l1_result, l2_result)`` returns the same cross-level metrics as
    the diagnostic path (must contain ``cross_mse``, ideally ``pred_progress_h``).

    Selection (step 5):

    1. a candidate is *feasible* if the low level predicts progress towards its
       own subgoal (``repair_min_pred_progress``) and its cross-level gap is
       below ``repair_max_cross_mse`` (when set);
    2. among feasible candidates, minimise the normalised combination
       ``J_H_norm + lambda * C_L_norm``.

    Acceptance (two rules, selected by ``repair_require_gap_improvement``):

    * legacy (default): the candidate must also lower the terminal cross-level
      gap by ``min_improvement`` -- historically the only rule; it silently
      vetoes long-term-better detours;
    * ``repair_require_gap_improvement=false``: accept whenever the combined
      score improves. This is the rule ECCHP needs, and the per-decision
      ``gap_improved`` field keeps the two comparable in the logs.

    ``lambda_vector`` (``(B,)``) is the per-environment weight. When ``None``
    the constant ``cfg.high_level_cost_weight`` is used.
    """
    gap_before = l1_first_metrics["cross_mse"].detach()
    n_candidates = int(getattr(cfg, "high_resample_count", 2) or 0)
    threshold = _repair_threshold(cfg)

    triggered = should_repair(
        cross_mse=gap_before,
        gap_threshold=threshold,
        subgoal_index=subgoal_index,
        enabled=bool(getattr(cfg, "repair_enabled", False)) and n_candidates > 0,
    )
    if trigger_mask is not None:
        triggered = triggered & trigger_mask.to(
            device=gap_before.device, dtype=torch.bool
        )
    if triggered.numel() == 0 or not bool(triggered.any()):
        return l1_first_result, l2_first_result, None

    batch_size = gap_before.numel()
    min_improvement = (
        _as_tensor(min_improvement, gap_before)
        if min_improvement is not None
        else torch.full_like(
            gap_before, float(getattr(cfg, "repair_min_improvement", 0.0))
        )
    )
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
    candidate_progress: List[Optional[torch.Tensor]] = [
        _metric_or_none(l1_first_metrics, "pred_progress_h")
    ]
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
    independent = bool(getattr(cfg, "high_resample_independent", True))
    try:
        for _ in range(n_candidates):
            if independent:
                # Re-sample from the *same* pre-repair warm starts so each
                # candidate is an independent draw instead of a refinement of
                # the previous candidate (review 3, step 5). The RNG is *not*
                # rewound here on purpose: rewinding would make every candidate
                # draw the same MPPI noise and therefore produce the same plan.
                restore_low_level_state(
                    h_planner.l2_planner, snapshots[0]["l2"], restore_rng=False
                )
                restore_low_level_state(
                    h_planner.l1_planner, snapshots[0]["l1"], restore_rng=False
                )
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
            candidate_progress.append(
                _metric_or_none(candidate_metrics, "pred_progress_h")
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
    scores = _combined_scores(
        gaps,
        candidate_jh,
        lam,
        mode=str(getattr(cfg, "repair_normalize_mode", "baseline")),
    )  # (K, B)

    # Feasibility gate: a candidate is only considered if the low level predicts
    # progress towards its own subgoal (and, optionally, a hard gap cap). This
    # replaces "the terminal gap must improve", which rejected necessary detours.
    feasible = _feasibility_mask(
        candidate_progress,
        gaps,
        min_progress=float(getattr(cfg, "repair_min_pred_progress", 0.0)),
        max_gap=float(getattr(cfg, "repair_max_cross_mse", 0.0)),
    )
    feasible[0] = False  # candidate 0 is the fallback, never a "repair"

    masked = scores.clone()
    masked[~feasible] = float("inf")
    selected = masked.argmin(dim=0)  # (B,)
    finite = torch.isfinite(masked.gather(0, selected.view(1, -1)).squeeze(0))
    selected = torch.where(finite, selected, torch.zeros_like(selected))

    selected_gap = gaps.gather(0, selected.view(1, -1)).squeeze(0)
    selected_score = scores.gather(0, selected.view(1, -1)).squeeze(0)
    score_before = scores[0]

    reachable_ok = selected_gap <= (gap_before + gap_tolerance)
    gap_improved = ((gap_before - selected_gap) > min_improvement) & reachable_ok
    # ECCHP stage 0: the legacy rule required the *terminal cross-level gap* to
    # improve. That silently vetoes detours that are long-term better (the
    # whole point of execution-calibrated switching), so it can be turned off
    # (``repair_require_gap_improvement=false``). When off, acceptance relies on
    # the feasibility gate plus the combined score ``J_H + lambda * C_L`` only.
    require_gap_improvement = bool(
        getattr(cfg, "repair_require_gap_improvement", True)
    )
    legacy_ok = (
        gap_improved if require_gap_improvement else torch.ones_like(gap_improved)
    )
    accepted = (
        triggered
        & finite
        & legacy_ok
        & (selected_score < score_before)
    )
    accepted = accepted.to(dtype=torch.bool)
    selected = torch.where(accepted, selected, torch.zeros_like(selected))

    if bool(getattr(cfg, "repair_force_reject", False)):
        # Validation / shadow mode: build and rank the candidates, then drop
        # them. With the RNG restoration below, such a run must be identical to
        # the baseline (stage-0 isolation test).
        accepted = torch.zeros_like(accepted)
        selected = torch.zeros_like(selected)

    # RNG isolation: a triggered-but-fully-rejected repair must not advance the
    # episode's random stream. Candidate re-sampling draws fresh MPPI noise; if
    # that noise leaked into later plans, merely *considering* a repair would
    # change the trajectory, which makes every ablation unreadable.
    if bool(getattr(cfg, "repair_restore_rng_on_reject", True)) and not bool(
        accepted.any()
    ):
        restore_rng_state(snapshots[0]["l1"])

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

    # ECCHP stage 2: expose the candidate set so a collection run can replay it
    # from the same real state. ``PlanningResult.actions`` is the *executed*
    # action space (already un-normalised by ``MPPIPlanner.plan``), unlike the
    # MPPI warm start ``ctrl.U``, so it can be replayed in the environment
    # without a second conversion.
    candidate_evidence: Optional[List[Dict[str, Any]]] = None
    if bool(getattr(cfg, "collect_candidates", False)):
        candidate_evidence = []
        for index, candidate_l1_result in enumerate(candidate_l1):
            actions = getattr(candidate_l1_result, "actions", None)
            candidate_evidence.append(
                {
                    "candidate_id": int(index),
                    "source": "hwm" if index == 0 else "resample",
                    "high_level_cost": (
                        candidate_jh[index].detach().cpu()
                        if candidate_jh[index] is not None
                        else None
                    ),
                    "cross_mse": gaps[index].detach().cpu(),
                    "pred_progress_h": (
                        candidate_progress[index].detach().cpu()
                        if candidate_progress[index] is not None
                        else None
                    ),
                    "selected": (selected == index).detach().cpu(),
                    "actions": (
                        actions.detach().cpu() if torch.is_tensor(actions) else None
                    ),
                }
            )

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
        candidate_gap_spread=(
            gaps[1:].max(dim=0).values - gaps[1:].min(dim=0).values
            if gaps.shape[0] > 1
            else None
        ),
        candidate_jh_spread=(
            None
            if any(j is None for j in candidate_jh[1:])
            else torch.stack(candidate_jh[1:], dim=0).max(dim=0).values
            - torch.stack(candidate_jh[1:], dim=0).min(dim=0).values
        ),
        acceptance_rule=(
            "legacy_gap" if require_gap_improvement else "combined_score"
        ),
        gap_improved=gap_improved,
        candidate_evidence=candidate_evidence,
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


def _baseline_per_env(values: torch.Tensor) -> torch.Tensor:
    """Normalise ``(K, B)`` values by candidate 0 (the plan we already have).

    With only 2-3 candidates min-max would stretch tiny differences to 0 and 1,
    which makes the score scale jump between decisions. Dividing by the baseline
    keeps the scale comparable across decisions.
    """
    base = values[0:1].abs().clamp_min(1e-8)
    return values / base


def _normalize(values: torch.Tensor, mode: str) -> torch.Tensor:
    if str(mode).lower() == "minmax":
        return _minmax_per_env(values)
    return _baseline_per_env(values)


def _metric_or_none(metrics: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    value = metrics.get(key) if metrics else None
    return value.detach() if torch.is_tensor(value) else None


def _as_tensor(value: Any, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or tensor acceptance margin to ``like``'s shape."""
    if torch.is_tensor(value):
        tensor = value.detach().to(device=like.device, dtype=like.dtype).view(-1)
        if tensor.numel() == 1:
            return tensor.expand_as(like)
        return tensor
    return torch.full_like(like, float(value))


def _feasibility_mask(
    progress: List[Optional[torch.Tensor]],
    gaps: torch.Tensor,
    *,
    min_progress: float,
    max_gap: float,
) -> torch.Tensor:
    """``(K, B)`` bool: candidate reaches its own subgoal well enough to be used."""
    feasible = torch.ones_like(gaps, dtype=torch.bool)
    if min_progress > 0.0:
        if all(p is not None for p in progress):
            stacked = torch.stack(progress, dim=0)
            feasible = feasible & (stacked >= float(min_progress))
        # Without a progress signal every candidate would be rejected, so the
        # check is skipped rather than silently disabling the branch.
    if max_gap > 0.0:
        feasible = feasible & (gaps <= float(max_gap))
    return feasible


def _combined_scores(
    gaps: torch.Tensor,
    high_level_costs: List[Optional[torch.Tensor]],
    lam: torch.Tensor,
    mode: str = "baseline",
) -> torch.Tensor:
    """``score = J_H_norm + lambda * C_L_norm`` with both terms normalised."""
    low = _normalize(gaps, mode)
    if any(cost is None for cost in high_level_costs):
        return low
    high = _normalize(torch.stack(high_level_costs, dim=0), mode)
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
    trigger_mask: Optional[torch.Tensor] = None,
    min_improvement: Any = None,
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
    threshold = _repair_threshold(cfg)
    triggered = should_repair(
        cross_mse=gap_before,
        gap_threshold=threshold,
        subgoal_index=subgoal_index,
        enabled=bool(getattr(cfg, "repair_enabled", False)),
    )
    if trigger_mask is not None:
        triggered = triggered & trigger_mask.to(
            device=gap_before.device, dtype=torch.bool
        )
    if triggered.numel() == 0 or not bool(triggered.any()):
        return l1_first_result, None

    top_k = int(getattr(cfg, "repair_top_k", 8) or 0)
    if top_k <= 0:
        return l1_first_result, None

    min_improvement = (
        _as_tensor(min_improvement, gap_before)
        if min_improvement is not None
        else torch.full_like(gap_before, float(getattr(cfg, "repair_min_improvement", 0.0)))
    )
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
        # Review 4, P0-1: the extra MPPI call must not consume the episode's
        # random stream, otherwise even a fully rejected repair changes every
        # later plan.
        self_rng_devices = (
            [torch.cuda.current_device()] if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=self_rng_devices, enabled=True):
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
    # Goal jitter: how far is this repair target from the previous one?
    subgoal_shift = None
    previous_goal = getattr(h_planner, "last_repair_goal", None)
    if previous_goal is not None and previous_goal.shape == selection.subgoal_l2.shape:
        subgoal_shift = (
            (selection.subgoal_l2.detach() - previous_goal.to(selection.subgoal_l2.device))
            .pow(2)
            .mean(dim=1)
        )
    selected_mask = selection.valid.to(dtype=torch.bool)
    if subgoal_shift is not None and bool(selected_mask.any()):
        updated = selection.subgoal_l2.detach().clone()
        if previous_goal is not None and previous_goal.shape == updated.shape:
            updated[~selected_mask] = previous_goal.to(updated.device)[~selected_mask]
        h_planner.last_repair_goal = updated
    elif bool(selected_mask.any()):
        h_planner.last_repair_goal = selection.subgoal_l2.detach().clone()

    # Only environments that actually found a reachable candidate may accept a
    # repaired plan; the batch-level ``valid.any()`` check above is not enough
    # because some environments can lack an admissible candidate.
    improvement = gap_before - repaired_gap
    accepted = (
        triggered
        & selection.valid.to(device=repaired_gap.device, dtype=torch.bool)
        & torch.isfinite(repaired_gap)
        & (improvement > min_improvement)
    )
    if bool(getattr(cfg, "repair_force_reject", False)):
        # Validation mode: compute the repair but never use it. The resulting run
        # must be identical to the baseline if isolation is correct.
        accepted = torch.zeros_like(accepted)
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
        subgoal_shift=subgoal_shift,
    )


