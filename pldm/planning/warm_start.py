"""Warm-start bookkeeping for the low-level MPPI planner.

``MPPIPlanner`` shifts its nominal action trajectory (``ctrl.U``) by the number
of steps consumed since the previous plan only when ``plan_size`` changed:

    if self.last_plan_size is not None and plan_size < self.last_plan_size:
        actions = actions[l2.ctrl_state :]
        # shift nominal trajectory
        for ctrl in self.ctrls:
            ctrl.shift_nominal_trajectory()

For HWM the low-level plan length is constant (``num_pred_steps``), and the
planner is re-invoked every ``replan_every`` primitives, so ``plan_size`` never
shrinks and the warm start is *never* shifted. Index 0 of ``ctrl.U`` therefore
keeps pointing at an action that has already been executed, which defeats the
purpose of warm starting: MPPI keeps re-optimising a trajectory that is anchored
at a stale point in time.

This module provides the missing shift. It is deliberately a small, isolated
function so the behaviour can be ablated with a single config flag
(``exec_repair.shift_l1_warm_start``).
"""

from typing import Any, Optional

import torch


def _ctrl_has_shift(ctrl: Any) -> bool:
    return callable(getattr(ctrl, "shift_nominal_trajectory", None))


def shift_low_level_warm_start(planner: Any, n_steps: int = 1) -> int:
    """Advance every ctrl's nominal trajectory by ``n_steps``.

    ``planner`` is the low-level ``MPPIPlanner``; the ctrls (one per environment
    for the batched variant) are shifted in place. Passing ``n_steps <= 0`` is a
    no-op, which keeps the caller free of extra conditionals.

    Returns the number of ctrl objects that were actually shifted, so callers
    can cheaply assert that the shift took effect.
    """
    if n_steps is None or n_steps <= 0:
        return 0

    ctrls = getattr(planner, "ctrls", None)
    if ctrls is None:
        # Single-environment planners keep the ctrl directly on the planner.
        ctrl = getattr(planner, "ctrl", None)
        ctrls = [ctrl] if ctrl is not None else []
    if not ctrls:
        return 0

    shifted = 0
    for ctrl in ctrls:
        if not _ctrl_has_shift(ctrl):
            continue
        for _ in range(int(n_steps)):
            ctrl.shift_nominal_trajectory()
        shifted += 1
    return shifted


def shift_nominal_trajectories(ctrls: Optional[Any], n_steps: int = 1) -> int:
    """Same as :func:`shift_low_level_warm_start` but for an explicit ctrl list."""
    if not ctrls or n_steps is None or n_steps <= 0:
        return 0
    if isinstance(ctrls, torch.nn.Module):  # pragma: no cover - defensive
        ctrls = [ctrls]
    shifted = 0
    for ctrl in ctrls:
        if not _ctrl_has_shift(ctrl):
            continue
        for _ in range(int(n_steps)):
            ctrl.shift_nominal_trajectory()
        shifted += 1
    return shifted
