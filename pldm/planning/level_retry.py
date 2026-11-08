"""Conditional low-level re-planning (Step 3).

The low-level MPPI planner is asked for a second plan only when the first plan
is predicted to fall short of the high-level subgoal (cross-level disagreement
above a threshold). The retry is accepted per environment and only if it
actually reduces the gap.

State handling matters here because ``MPPI`` keeps a warm start:

* ``ctrl.U`` is updated in-place by every ``plan()`` call, so it has to be
  snapshotted per environment and restored for rejected environments;
* ``last_plan_size`` controls horizon shifting and is restored as well;
* sampling consumes torch RNG, so when nothing is accepted the CPU/CUDA RNG
  states are restored to make the episode behave as if no retry happened.

Everything in this module is inert unless
``ExecRepairConfig.enabled and ExecRepairConfig.retry_enabled`` and a positive
``retry_gap_threshold`` are set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from pldm.planning.planners.planner import PlanningResult

# Batch dimension of every ``PlanningResult`` field (verified against
# ``MPPIPlanner.plan`` output shapes).
_FIELD_BATCH_DIMS: Dict[str, int] = {
    "actions": 0,  # (B, T, action_dim)
    "pred_obs": 1,  # (T, B, ...)
    "pred_proprio": 1,  # (T, B, ...)
    "pred_location": 1,  # (T, B, ...)
    "raw_locations": 1,  # (T, B, ...)
    "locations": 1,  # (T, B, 2)
    "ensemble_predictions": 2,  # (T, E, B, ...)
    "ensemble_obs_component": 2,
    "ensemble_proprio_component": 2,
    "ensemble_raw_locations": 2,
}


@dataclass
class RetryInfo:
    """Per-decision retry statistics (logged by ``MPCEvaluator``)."""

    triggered: torch.Tensor
    accepted: torch.Tensor
    gap_before: torch.Tensor
    gap_after: torch.Tensor
    gap_selected: torch.Tensor
    wall_time_s: float
    path_cost_before: Optional[torch.Tensor] = None
    path_cost_after: Optional[torch.Tensor] = None
    path_cost_selected: Optional[torch.Tensor] = None

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
            "wall_ms": self.wall_time_s * 1000.0,
            "n_triggered": self.n_triggered,
            "n_accepted": self.n_accepted,
        }
        for name in (
            "path_cost_before",
            "path_cost_after",
            "path_cost_selected",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


def snapshot_low_level_state(planner: Any) -> Dict[str, Any]:
    """Snapshot warm starts and RNG state before a retry."""
    ctrl_states = None
    ctrls = getattr(planner, "ctrls", None)
    if ctrls is not None:
        ctrl_states = []
        for ctrl in ctrls:
            warm_start = getattr(ctrl, "U", None)
            ctrl_states.append(
                warm_start.detach().clone() if torch.is_tensor(warm_start) else None
            )

    return {
        "ctrl_states": ctrl_states,
        "last_plan_size": getattr(planner, "last_plan_size", None),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(snapshot: Dict[str, Any]) -> None:
    """Restore CPU/CUDA RNG states captured by :func:`snapshot_low_level_state`."""
    torch.set_rng_state(snapshot["cpu_rng"])
    if snapshot.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda_rng"])


def restore_low_level_state(
    planner: Any,
    snapshot: Dict[str, Any],
    *,
    env_mask: Optional[torch.Tensor] = None,
    restore_rng: bool = False,
) -> None:
    """Restore warm starts for the selected environments (all when mask is None)."""
    ctrls = getattr(planner, "ctrls", None)
    saved = snapshot.get("ctrl_states")
    if ctrls is not None and saved is not None:
        if env_mask is None:
            indices = list(range(min(len(ctrls), len(saved))))
        else:
            mask = (
                env_mask.detach().cpu().tolist()
                if torch.is_tensor(env_mask)
                else list(env_mask)
            )
            indices = [i for i, should_restore in enumerate(mask) if should_restore]
        for i in indices:
            if i < len(saved) and saved[i] is not None and hasattr(ctrls[i], "U"):
                ctrls[i].U = saved[i].clone()

    if "last_plan_size" in snapshot:
        planner.last_plan_size = snapshot["last_plan_size"]

    if restore_rng:
        restore_rng_state(snapshot)


def blend_planning_results(
    first: PlanningResult,
    second: PlanningResult,
    mask: torch.Tensor,
) -> PlanningResult:
    """Per-environment select ``second`` where ``mask`` is True, else ``first``."""
    mask = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
    mask = mask.to(dtype=torch.bool)

    merged: Dict[str, Any] = {}
    for field in PlanningResult._fields:
        first_value = getattr(first, field)
        second_value = getattr(second, field)

        if field == "losses":
            # Both plans come from the same planner and report the same stub.
            merged[field] = first_value
            continue

        if first_value is None or second_value is None:
            if first_value is None and second_value is None:
                merged[field] = None
                continue
            raise RuntimeError(
                f"cannot blend PlanningResult field {field!r}: one side is None"
            )

        batch_dim = _FIELD_BATCH_DIMS.get(field)
        if batch_dim is None:
            raise NotImplementedError(
                f"blend_planning_results does not handle field {field!r}; "
                "add it to _FIELD_BATCH_DIMS"
            )

        if first_value.shape != second_value.shape:
            raise RuntimeError(
                f"field {field!r} shape mismatch: {tuple(first_value.shape)} vs "
                f"{tuple(second_value.shape)}"
            )
        if first_value.shape[batch_dim] != mask.numel():
            raise RuntimeError(
                f"field {field!r} batch dim {batch_dim} has size "
                f"{first_value.shape[batch_dim]}, expected {mask.numel()}"
            )

        view_shape = [1] * first_value.dim()
        view_shape[batch_dim] = mask.numel()
        merged[field] = torch.where(mask.view(view_shape), second_value, first_value)

    return PlanningResult(**merged)


def plan_with_retry(
    planner: Any,
    plan_kwargs: Dict[str, Any],
    first_result: PlanningResult,
    first_metrics: Dict[str, torch.Tensor],
    metrics_fn: Callable[[PlanningResult], Dict[str, torch.Tensor]],
    cfg: Any,
) -> Tuple[PlanningResult, RetryInfo]:
    """Optionally re-plan with the low-level planner and keep per-env improvements.

    ``first_metrics`` / ``metrics_fn`` must provide ``cross_mse`` and should
    provide ``path_cost``; the latter guards against accepting a retry that only
    improves the terminal prediction while making the whole path worse.
    """
    gap_before = first_metrics["cross_mse"].detach()
    path_before = first_metrics.get("path_cost")
    if path_before is not None:
        path_before = path_before.detach()

    threshold = float(getattr(cfg, "retry_gap_threshold", 0.0))
    if threshold > 0.0:
        triggered = gap_before > threshold
    else:
        triggered = torch.zeros_like(gap_before, dtype=torch.bool)
    triggered = triggered.to(dtype=torch.bool)

    if not bool(triggered.any()):
        no_retry = torch.zeros_like(triggered)
        return first_result, RetryInfo(
            triggered=triggered,
            accepted=no_retry,
            gap_before=gap_before,
            gap_after=gap_before,
            gap_selected=gap_before,
            wall_time_s=0.0,
            path_cost_before=path_before,
            path_cost_after=path_before,
            path_cost_selected=path_before,
        )

    snapshot = snapshot_low_level_state(planner)

    try:
        start = time.time()
        retry_result = planner.plan(**plan_kwargs)
        wall_time_s = time.time() - start
        retry_metrics = metrics_fn(retry_result)
    except Exception:
        # Leave no partially updated warm starts behind if the retry fails
        # (OOM, shape errors, ...).
        restore_low_level_state(planner, snapshot, restore_rng=True)
        raise

    gap_after = retry_metrics["cross_mse"].detach()
    path_after = retry_metrics.get("path_cost")
    if path_after is not None:
        path_after = path_after.detach()

    improvement = gap_before - gap_after
    min_abs = float(getattr(cfg, "retry_min_improvement", 0.0))
    min_ratio = float(getattr(cfg, "retry_min_improvement_ratio", 0.0))

    accepted = triggered & (improvement > min_abs)
    if min_ratio > 0.0:
        ratio = improvement / gap_before.clamp_min(1e-12)
        accepted = accepted & (ratio > min_ratio)
    if (
        bool(getattr(cfg, "retry_require_path_cost_improvement", False))
        and path_before is not None
        and path_after is not None
    ):
        accepted = accepted & (path_after <= path_before)
    accepted = accepted.to(dtype=torch.bool)

    merged = (
        blend_planning_results(first_result, retry_result, accepted)
        if bool(accepted.any())
        else first_result
    )

    # Keep the retry warm start for accepted envs, restore the rest.
    restore_low_level_state(planner, snapshot, env_mask=~accepted)

    if not bool(accepted.any()) and bool(
        getattr(cfg, "retry_restore_rng_on_reject", True)
    ):
        restore_rng_state(snapshot)

    gap_selected = torch.where(accepted, gap_after, gap_before)
    path_selected = (
        torch.where(accepted, path_after, path_before)
        if path_before is not None and path_after is not None
        else None
    )
    return merged, RetryInfo(
        triggered=triggered,
        accepted=accepted,
        gap_before=gap_before,
        gap_after=gap_after,
        gap_selected=gap_selected,
        wall_time_s=wall_time_s,
        path_cost_before=path_before,
        path_cost_after=path_after,
        path_cost_selected=path_selected,
    )
