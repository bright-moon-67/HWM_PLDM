#!/usr/bin/env python3
"""Self-checks for Step 3 (conditional low-level retry).

These checks do not need MuJoCo, a checkpoint or a GPU:

* V1  projection used by the trigger matches ``RunningCost`` numerically
* V2  no trigger -> no second plan, no RNG consumption, planner untouched
* V3  all accepted -> retry result kept; all rejected -> first result, warm
      start and RNG restored
* V4  partial acceptance -> per-environment blending and per-env warm starts
* V5  every ``PlanningResult`` field is covered by the blend helper
* V6  the new config switches parse through OmegaConf

Run with::

    python scripts/check_step3_retry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from pldm.planning.diagnostics import (  # noqa: E402
    cross_level_metrics,
    project_l1_encoding_to_l2,
)
from pldm.planning.level_retry import (  # noqa: E402
    _FIELD_BATCH_DIMS,
    blend_planning_results,
    plan_with_retry,
)
from pldm.planning.planners.mppi_planner import RunningCost  # noqa: E402
from pldm.planning.planners.planner import PlanningResult  # noqa: E402
from pldm.planning.planners.two_lvl_planner import TwoLvlPlanner  # noqa: E402

TESTS = []


def case(fn):
    TESTS.append(fn)
    return fn


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _EncoderOutput:
    def __init__(self, encodings):
        self.encodings = encodings


class FakeEncoder(torch.nn.Module):
    """Behaves like ``IdentityEncoder``: concatenates obs and proprio channels."""

    def __init__(self, using_proprio: bool = True):
        super().__init__()
        self.using_proprio = using_proprio
        self.input_dim = (3, 4, 4)

    def forward(self, x, proprio=None, **kwargs):
        if self.using_proprio:
            if proprio is None:
                raise ValueError("proprio required")
            encodings = torch.cat([x, proprio], dim=1)
        else:
            encodings = x
        return _EncoderOutput(encodings)


class _FakeBackbone:
    output_obs_dim = (2, 4, 4)
    output_proprio_dim = (1, 4, 4)


class _FakeModel:
    backbone = _FakeBackbone()


class FakeObjective:
    def __init__(self, pred_encoder, target=None):
        self.pred_encoder = pred_encoder
        self.model = _FakeModel()
        self.sum_all_diffs = True
        self.target_enc = target


class RetryCfg:
    def __init__(
        self,
        threshold: float = 0.2,
        min_improvement: float = 0.0,
        min_improvement_ratio: float = 0.0,
        restore_rng_on_reject: bool = True,
    ):
        self.retry_gap_threshold = threshold
        self.retry_min_improvement = min_improvement
        self.retry_min_improvement_ratio = min_improvement_ratio
        self.retry_restore_rng_on_reject = restore_rng_on_reject


class FakeCtrl:
    def __init__(self, u):
        self.U = u


class FakeLowPlanner:
    """Minimal stand-in for MPPIPlanner: mutates U and consumes RNG on plan()."""

    def __init__(self, retry_result, n_envs=2):
        self.ctrls = [FakeCtrl(torch.ones(2, 2)) for _ in range(n_envs)]
        self.last_plan_size = 2
        self._retry_result = retry_result
        self.calls = 0

    def plan(self, **kwargs):
        self.calls += 1
        torch.randn(7)  # simulate MPPI sampling consuming the global RNG
        for ctrl in self.ctrls:
            ctrl.U = ctrl.U + 1.0
        return self._retry_result


def make_result(value: float, batch: int = 2, steps: int = 2) -> PlanningResult:
    return PlanningResult(
        ensemble_predictions=torch.full((steps, 2, batch, 3), value),
        ensemble_obs_component=torch.full((steps, 2, batch, 3), value),
        ensemble_proprio_component=torch.full((steps, 2, batch, 3), value),
        ensemble_raw_locations=torch.full((steps, 2, batch, 2), value),
        pred_obs=torch.full((steps, batch, 3), value),
        pred_proprio=torch.full((steps, batch, 2), value),
        pred_location=torch.full((steps, batch, 2), value),
        raw_locations=torch.full((steps, batch, 2), value),
        actions=torch.full((batch, steps, 2), value),
        locations=torch.full((steps, batch, 2), value),
        losses=[0],
    )


# --------------------------------------------------------------------------- #
# V1: projection equivalence with RunningCost
# --------------------------------------------------------------------------- #
@case
def v1_projection_matches_running_cost():
    torch.manual_seed(0)
    batch = 4
    obs = torch.randn(batch, 2, 4, 4)
    proprio = torch.randn(batch, 1, 4, 4)
    state = torch.cat([obs, proprio], dim=1)  # fused L1 encoding, (B, 3, 4, 4)

    objective = FakeObjective(FakeEncoder(True), torch.randn(1, 3 * 4 * 4))
    running_cost = RunningCost(objective, idx=0, cost_dim_range="0:99999999")

    reference = running_cost(state, None)
    projected = project_l1_encoding_to_l2(state, objective)
    ours = (projected - objective.target_enc[0]).pow(2).mean(dim=-1)

    assert projected.shape == (batch, 3 * 4 * 4), projected.shape
    assert torch.allclose(reference, ours, atol=1e-6), (reference, ours)

    # 5D inputs (T, B, ...) must match the per-step projection.
    state_5d = state.unsqueeze(0).repeat(3, 1, 1, 1, 1)
    projected_5d = project_l1_encoding_to_l2(state_5d, objective)
    assert projected_5d.shape == (3, batch, 3 * 4 * 4)
    assert torch.allclose(projected_5d[-1], projected)

    # Non-fused encoder branch.
    objective_plain = FakeObjective(FakeEncoder(False), torch.randn(1, 2 * 4 * 4))
    plain_state = obs
    running_cost_plain = RunningCost(objective_plain, idx=0)
    plain_reference = running_cost_plain(plain_state, None)
    plain_projected = project_l1_encoding_to_l2(plain_state, objective_plain)
    plain_ours = (plain_projected - objective_plain.target_enc[0]).pow(2).mean(-1)
    assert torch.allclose(plain_reference, plain_ours, atol=1e-6)


@case
def v1b_cross_level_metrics_shapes():
    torch.manual_seed(1)
    batch, steps = 3, 5
    obs = torch.randn(steps, batch, 2, 4, 4)
    proprio = torch.randn(steps, batch, 1, 4, 4)
    l1_pred_obs = torch.cat([obs, proprio], dim=2)

    objective = FakeObjective(FakeEncoder(True), torch.randn(1, 3 * 4 * 4))
    l2_pred_obs = torch.randn(steps, batch, 3, 4, 4)

    metrics = cross_level_metrics(l1_pred_obs, l2_pred_obs, objective, subgoal_index=1)
    for name in ("cross_mse", "cross_l2", "l2_subgoal_norm", "l1_terminal_norm"):
        assert metrics[name].shape == (batch,), (name, metrics[name].shape)

    try:
        cross_level_metrics(l1_pred_obs, l2_pred_obs[:, :, :1], objective)
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected a shape mismatch error")


# --------------------------------------------------------------------------- #
# V2 / V3 / V4: retry behaviour
# --------------------------------------------------------------------------- #
@case
def v2_no_trigger_is_inert():
    first = make_result(0.0)
    planner = FakeLowPlanner(make_result(1.0))
    rng_before = torch.get_rng_state()
    warm_before = [ctrl.U.clone() for ctrl in planner.ctrls]

    result, info = plan_with_retry(
        planner,
        {},
        first,
        torch.tensor([0.1, 0.2]),
        lambda _result: torch.zeros(2),
        RetryCfg(threshold=1.0),
    )

    assert result is first
    assert planner.calls == 0
    assert info.n_triggered == 0 and info.n_accepted == 0
    assert torch.equal(torch.get_rng_state(), rng_before)
    for ctrl, warm in zip(planner.ctrls, warm_before):
        assert torch.equal(ctrl.U, warm)


@case
def v3_accept_all_keeps_retry():
    first = make_result(0.0)
    planner = FakeLowPlanner(make_result(1.0))

    result, info = plan_with_retry(
        planner,
        {},
        first,
        torch.tensor([0.5, 0.5]),
        lambda _result: torch.tensor([0.1, 0.1]),
        RetryCfg(threshold=0.2),
    )

    assert planner.calls == 1
    assert info.n_triggered == 2 and info.n_accepted == 2
    assert torch.all(result.pred_obs == 1.0)
    assert torch.all(result.actions == 1.0)
    for ctrl in planner.ctrls:
        assert torch.all(ctrl.U == 2.0), ctrl.U


@case
def v3b_reject_all_restores_state():
    first = make_result(0.0)
    planner = FakeLowPlanner(make_result(1.0))
    rng_before = torch.get_rng_state()

    result, info = plan_with_retry(
        planner,
        {},
        first,
        torch.tensor([0.5, 0.5]),
        lambda _result: torch.tensor([0.9, 0.9]),
        RetryCfg(threshold=0.2),
    )

    assert planner.calls == 1
    assert info.n_accepted == 0
    assert result is first
    for ctrl in planner.ctrls:
        assert torch.all(ctrl.U == 1.0), ctrl.U
    assert planner.last_plan_size == 2
    if torch.cuda.is_available():
        pass  # CUDA RNG is restored too; CPU comparison below is the strict one
    assert torch.equal(torch.get_rng_state(), rng_before)


@case
def v4_partial_accept_blends_per_env():
    first = make_result(0.0)
    planner = FakeLowPlanner(make_result(1.0))

    result, info = plan_with_retry(
        planner,
        {},
        first,
        torch.tensor([0.5, 0.5]),
        lambda _result: torch.tensor([0.1, 0.9]),
        RetryCfg(threshold=0.2),
    )

    assert info.n_accepted == 1
    assert bool(info.accepted[0]) and not bool(info.accepted[1])
    assert torch.all(result.pred_obs[:, 0] == 1.0)
    assert torch.all(result.pred_obs[:, 1] == 0.0)
    assert torch.all(planner.ctrls[0].U == 2.0)
    assert torch.all(planner.ctrls[1].U == 1.0)
    assert torch.allclose(info.gap_selected, torch.tensor([0.1, 0.5]))


@case
def v5_all_result_fields_are_blended():
    expected = set(PlanningResult._fields) - {"losses"}
    assert set(_FIELD_BATCH_DIMS) == expected, set(_FIELD_BATCH_DIMS) ^ expected

    first = make_result(0.0)
    second = make_result(1.0)
    mask = torch.tensor([True, False])
    merged = blend_planning_results(first, second, mask)

    for field, batch_dim in _FIELD_BATCH_DIMS.items():
        value = getattr(merged, field)
        index = [slice(None)] * value.dim()
        for env, expected_value in ((0, 1.0), (1, 0.0)):
            index[batch_dim] = env
            assert torch.all(value[tuple(index)] == expected_value), field
    assert merged.losses == first.losses


@case
def v5b_unknown_field_is_rejected():
    from pldm.planning import level_retry

    saved = level_retry._FIELD_BATCH_DIMS.pop("pred_obs")
    try:
        try:
            blend_planning_results(
                make_result(0.0), make_result(1.0), torch.tensor([True, True])
            )
        except NotImplementedError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected NotImplementedError for an unmapped field")
    finally:
        level_retry._FIELD_BATCH_DIMS["pred_obs"] = saved


# --------------------------------------------------------------------------- #
# V6: config plumbing
# --------------------------------------------------------------------------- #
@case
def v6_config_switches_parse():
    from omegaconf import OmegaConf

    from pldm.planning.d4rl.enums import HierarchicalD4RLMPCConfig

    config_path = (
        REPO_ROOT / "pldm/configs/diverse_maze/icml/hard_hwm_exec.yaml"
    )
    full = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    node = OmegaConf.create(full["eval_cfg"]["h_d4rl_planning"])
    OmegaConf.update(node, "exec_repair.retry_enabled", True)
    OmegaConf.update(node, "exec_repair.retry_gap_threshold", 0.25)

    merged = OmegaConf.merge(
        OmegaConf.structured(HierarchicalD4RLMPCConfig), node
    )
    parsed = HierarchicalD4RLMPCConfig.parse_from_dict(
        OmegaConf.to_container(merged, resolve=True)
    )

    assert parsed.exec_repair.retry_enabled is True
    assert abs(parsed.exec_repair.retry_gap_threshold - 0.25) < 1e-9
    assert parsed.exec_repair.subgoal_index == 1
    assert parsed.exec_repair.retry_restore_rng_on_reject is True


@case
def v6b_two_level_planner_gating():
    off = RetryCfg(threshold=0.2)
    off.enabled = False
    off.retry_enabled = True
    off.subgoal_index = 1
    assert TwoLvlPlanner(None, None, 10, exec_cfg=off)._retry_on is False

    on = RetryCfg(threshold=0.2)
    on.enabled = True
    on.retry_enabled = True
    on.subgoal_index = 1
    assert TwoLvlPlanner(None, None, 10, exec_cfg=on)._retry_on is True

    threshold_zero = RetryCfg(threshold=0.0)
    threshold_zero.enabled = True
    threshold_zero.retry_enabled = True
    threshold_zero.subgoal_index = 1
    assert TwoLvlPlanner(None, None, 10, exec_cfg=threshold_zero)._retry_on is False


def main() -> int:
    failures = []
    for fn in TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {fn.__name__}")

    print()
    if failures:
        print(f"{len(failures)}/{len(TESTS)} checks failed")
        return 1
    print(f"all {len(TESTS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
