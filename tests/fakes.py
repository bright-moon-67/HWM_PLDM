"""Minimal fake planners/environments for the planner-contract tests.

The real ``MPPIPlanner`` needs a trained world model and CUDA; the planner
contract tests only need the *interface* (warm starts, RNG consumption, plan
results) so they can run on CPU in a fraction of a second. Every fake here
consumes the global torch RNG on purpose: that is what makes RNG-isolation
regressions observable.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from pldm.models.encoders.enums import BackboneOutput
from pldm.models.utils import flatten_conv_output
from pldm.planning.planners.planner import PlanningResult


class FakeBackbone:
    using_proprio = False
    using_location = False
    output_obs_dim = None
    output_proprio_dim = None

    def __call__(self, obs, proprio=None, locations=None):
        return BackboneOutput(encodings=obs, obs_component=obs)


class FakeObjective:
    """Objective stand-in with the two attributes the planner code touches."""

    def __init__(self):
        self.target_enc = None
        self.pred_encoder = None  # identity projection in the diagnostics
        self.model = None
        self.sum_all_diffs = False
        self.sum_last_n = None

    def set_target(self, targets, repr_input: bool = True):
        self.target_enc = targets


class FakeController:
    def __init__(self, plan_len: int, action_dim: int = 1):
        self.U = torch.zeros(plan_len, action_dim)
        self.perturbed_action = torch.zeros(4, plan_len, action_dim)
        self.cost_total = torch.zeros(4)
        self.shift_calls = 0
        # Every ``U`` written by a plan call, so tests can check that the
        # warm start of the *selected* candidate was kept.
        self.history: List[torch.Tensor] = []

    def shift_nominal_trajectory(self):
        self.U = torch.cat([self.U[1:], self.U[-1:]], dim=0)
        self.shift_calls += 1


class FakePlanner:
    """Deterministic-in-structure, RNG-consuming planner.

    ``script`` is a list of ``(B, 2)`` endpoint tensors, one per ``plan`` call.
    Calls beyond the script reuse its last entry. The endpoint is used as the
    plan terminal (L1) or the macro subgoal at every step >= 1 (L2).
    """

    def __init__(
        self,
        script: Sequence[torch.Tensor],
        *,
        batch: int,
        plan_len: int = 6,
        step_skip: int = 6,
        kind: str = "l1",
    ):
        self.script: List[torch.Tensor] = [torch.as_tensor(s).float() for s in script]
        self.batch = int(batch)
        self.plan_len = int(plan_len)
        self.kind = kind
        self.model = SimpleNamespace(
            backbone=FakeBackbone(),
            spatial_repr_dim=2,
            config=SimpleNamespace(
                step_skip=step_skip,
                predictor=SimpleNamespace(z_dim=0, action_dim=1, ensemble_size=1),
            ),
        )
        self.model.predictor = self.model.config.predictor
        self.objective = FakeObjective()
        self.ctrls = [FakeController(self.plan_len) for _ in range(self.batch)]
        self.last_plan_size = None
        self.calls = 0
        self.attempts = 0
        self.fail_on_calls: Dict[int, Exception] = {}

    # ------------------------------------------------------------------ utils
    def reset(self):
        self.calls = 0
        self.attempts = 0
        self.last_plan_size = None

    def _endpoint(self) -> torch.Tensor:
        if not self.script:
            raise RuntimeError("FakePlanner.script is empty")
        # Index by *attempts*, not by successful calls: a failed candidate
        # search must still consume its script slot, otherwise the script of a
        # later decision would silently shift.
        index = min(max(self.attempts - 1, 0), len(self.script) - 1)
        end = self.script[index]
        if end.shape[0] != self.batch:
            raise RuntimeError(
                f"script endpoint has batch {end.shape[0]}, expected {self.batch}"
            )
        return end

    # ------------------------------------------------------------------- plan
    @torch.no_grad()
    def plan(
        self,
        current_state,
        plan_size: int,
        repr_input: bool = True,
        curr_proprio_pos=None,
        curr_proprio_vel=None,
        curr_locations=None,
        diff_loss_idx=None,
        decision_step=None,
    ):
        call_number = self.calls + 1
        self.attempts += 1
        if self.attempts in self.fail_on_calls:
            raise self.fail_on_calls[self.attempts]
        end = self._endpoint()

        encodings = (
            current_state.encodings
            if hasattr(current_state, "encodings")
            else current_state
        )
        start = flatten_conv_output(encodings).detach().float()  # (B, 2)
        if start.shape[1] != 2:
            raise RuntimeError(
                f"fake planners expect 2-dim encodings, got {tuple(start.shape)}"
            )

        # Consume the global RNG on purpose (MPPI sampling does the same). The
        # *plan identity* is encoded in action dimension 0 (stable across runs,
        # so two runs can be compared), while the RNG draw is visible in
        # dimension 1 (so a leaked RNG draw changes the actions).
        noise = torch.rand(self.batch, 1)
        code = end[:, 0] * 1000.0 + end[:, 1]
        actions = torch.stack(
            [
                code.unsqueeze(1).expand(self.batch, int(plan_size)),
                noise.expand(self.batch, int(plan_size)),
            ],
            dim=-1,
        )

        steps = int(plan_size) + 1
        positions = torch.zeros(steps, self.batch, 2)
        for k in range(steps):
            if self.kind == "l2":
                # Index 0 is the current state, everything after it is the
                # macro subgoal (cross_level_metrics reads index 1).
                positions[k] = start if k == 0 else end
            else:
                frac = k / max(steps - 1, 1)
                positions[k] = start + (end - start) * frac

        pred_obs = positions.view(steps, self.batch, 2, 1, 1)
        ensemble = pred_obs.unsqueeze(1)  # (T, E=1, B, ...)
        locations = positions.clone()

        # Warm starts change on every call, so a missing restore is visible.
        for ctrl in self.ctrls:
            ctrl.U = torch.rand_like(ctrl.U)
            ctrl.perturbed_action = torch.rand(4, ctrl.U.shape[0], ctrl.U.shape[1])
            ctrl.cost_total = torch.rand(4)
            ctrl.history.append(ctrl.U.clone())
        self.last_plan_size = int(plan_size)
        self.calls = call_number

        return PlanningResult(
            ensemble_predictions=ensemble,
            ensemble_obs_component=ensemble,
            ensemble_proprio_component=None,
            ensemble_raw_locations=None,
            pred_obs=pred_obs,
            pred_proprio=None,
            pred_location=None,
            raw_locations=None,
            actions=actions,
            locations=locations,
            losses=[0],
        )

    def reset_targets(self, targets, repr_input: bool = True):
        self.objective.set_target(targets, repr_input=repr_input)


def make_planners(
    base_l2: torch.Tensor,
    base_l1: torch.Tensor,
    *,
    candidate_l2: Optional[torch.Tensor] = None,
    candidate_l1: Optional[torch.Tensor] = None,
    decisions: int = 1,
    extra_candidates: int = 2,
    plan_len: int = 6,
    step_skip: int = 6,
):
    """Build an L1/L2 planner pair with per-decision, per-candidate scripts.

    The returned scripts mirror the call order of ``TwoLvlPlanner.plan``:
    one L2 + one L1 plan per decision, plus ``extra_candidates`` L2/L1 pairs
    whenever the repair branch fires.
    """
    base_l2 = torch.as_tensor(base_l2).float()
    base_l1 = torch.as_tensor(base_l1).float()
    batch = base_l2.shape[0]
    if candidate_l2 is None:
        candidate_l2 = base_l2
    if candidate_l1 is None:
        candidate_l1 = base_l1
    candidate_l2 = torch.as_tensor(candidate_l2).float()
    candidate_l1 = torch.as_tensor(candidate_l1).float()

    l2_script: List[torch.Tensor] = []
    l1_script: List[torch.Tensor] = []
    for _ in range(decisions):
        l2_script.append(base_l2.clone())
        l1_script.append(base_l1.clone())
        for _ in range(extra_candidates):
            l2_script.append(candidate_l2.clone())
            l1_script.append(candidate_l1.clone())

    l2_planner = FakePlanner(
        l2_script, batch=batch, plan_len=plan_len, step_skip=step_skip, kind="l2"
    )
    l1_planner = FakePlanner(
        l1_script, batch=batch, plan_len=plan_len, step_skip=step_skip, kind="l1"
    )
    l1_planner.objective.target_enc = torch.zeros(2)
    return l1_planner, l2_planner


# --------------------------------------------------------------------------- #
# Fake environment supporting full state snapshots (dm_control style).
# --------------------------------------------------------------------------- #
class FakePhysics:
    def __init__(self):
        self.state = np.zeros(4, dtype=np.float64)

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = np.asarray(state, dtype=np.float64).copy()


class FakeNavWrapper:
    """Straight corridor of ``length`` cells: distance = |i - goal_i|."""

    def __init__(self, length: int = 10, goal_ij=(9, 0)):
        self.length = int(length)
        self.goal_ij = tuple(goal_ij)

    def xy_to_ij(self, xy):
        x = float(np.asarray(xy).reshape(-1)[0])
        return (int(np.clip(round(x), 0, self.length - 1)), 0)

    def find_shortest_path(self, current_ij, target_ij):
        start = int(current_ij[0])
        goal = int(target_ij[0])
        steps = abs(goal - start)
        if steps == 0:
            return [], False
        direction = 1 if goal > start else -1
        path = [(start + direction * k, 0) for k in range(1, steps + 1)]
        return path, False

    def get_target(self):
        return np.array([float(self.goal_ij[0]), 0.0])


class FakeEnv:
    """Tiny environment with a physics-like snapshot interface."""

    def __init__(self, start_x: float = 0.0, goal_x: float = 9.0):
        self.physics = FakePhysics()
        self.physics.state[0] = float(start_x)
        self.goal_x = float(goal_x)
        self.nav = FakeNavWrapper(length=int(goal_x) + 1)
        self.action_space = SimpleNamespace(shape=(1,))
        self.steps = 0
        self._obs = np.array([start_x, 0.0], dtype=np.float32)

    # --- gym-ish interface used by the evaluator -------------------------
    def get_obs(self):
        return np.array([self.physics.state[0], 0.0], dtype=np.float32)

    def get_info(self):
        return {"location": self.get_obs()}

    def get_proprio_pos(self, normalized=False):
        return self.get_obs()[:2]

    def get_proprio_vel(self, normalized=False):
        return np.zeros(2, dtype=np.float32)

    def get_pos(self, normalized=False):
        return self.get_obs()

    def get_target(self):
        return np.array([self.goal_x, 0.0], dtype=np.float32)

    # --- navigation wrapper API (delegated) ------------------------------
    def xy_to_ij(self, xy):
        return self.nav.xy_to_ij(xy)

    def find_shortest_path(self, current_ij, target_ij):
        return self.nav.find_shortest_path(current_ij, target_ij)

    def step(self, action):
        delta = float(np.asarray(action).reshape(-1)[0])
        self.physics.state[0] = float(
            np.clip(self.physics.state[0] + delta, 0.0, self.goal_x)
        )
        self.steps += 1
        return self.get_obs(), 0.0, False, False, self.get_info()


def stacked_zero_obs(batch: int) -> torch.Tensor:
    return torch.zeros(batch, 2, 1, 1)
