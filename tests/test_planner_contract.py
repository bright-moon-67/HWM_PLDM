"""Planner-contract tests (ECCHP stage 0).

These tests protect the invariants every later experiment depends on:

* with the research switches off, the planner is unchanged;
* building candidates without accepting any of them (shadow / collection mode)
  must not change the executed actions *or* the random stream;
* the per-environment choice is consistent with the merged plan;
* a failing candidate generator falls back safely, and re-raises only in the
  explicit debug mode;
* batching does not leak state between environments.
"""

from __future__ import annotations

import unittest

import torch

from pldm.planning.d4rl.enums import ExecRepairConfig
from pldm.planning.planners.two_lvl_planner import TwoLvlPlanner
from tests.fakes import make_planners, stacked_zero_obs


def _cuda_identity(self, *args, **kwargs):  # pragma: no cover - trivial
    return self


class PlannerContractTest(unittest.TestCase):
    """CPU-only contract tests for ``TwoLvlPlanner``."""

    @classmethod
    def setUpClass(cls):
        # ``TwoLvlPlanner.plan`` moves proprio/location tensors to CUDA. The
        # fakes are CPU tensors, so make ``.cuda()`` a no-op for the duration of
        # these tests (the planner logic itself is device agnostic).
        cls._original_cuda = torch.Tensor.cuda
        torch.Tensor.cuda = _cuda_identity

    @classmethod
    def tearDownClass(cls):
        torch.Tensor.cuda = cls._original_cuda

    # ------------------------------------------------------------------ utils
    def _scripts(self, batch: int):
        """Endpoints where env 0 should switch, env 1 must reject, rest idle."""
        base_l2 = torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
        )[:batch]
        base_l1 = torch.tensor(
            [[1.0, 1.0], [2.0, 0.5], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
        )[:batch]
        candidate_l2 = torch.tensor(
            [[0.5, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
        )[:batch]
        candidate_l1 = torch.tensor(
            [[0.5, 0.05], [2.0, 1.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
        )[:batch]
        return base_l2, base_l1, candidate_l2, candidate_l1

    def _config(self, **overrides) -> ExecRepairConfig:
        values = dict(
            enabled=True,
            repair_enabled=True,
            repair_mode="reselection",
            repair_gap_threshold=0.1,
            high_resample_count=2,
            high_level_cost_weight=0.5,
        )
        values.update(overrides)
        return ExecRepairConfig(**values)

    def _run(
        self,
        cfg: ExecRepairConfig,
        *,
        batch: int = 3,
        decisions: int = 2,
        seed: int = 0,
        extra_candidates: int = 2,
        fail_on_calls=None,
    ):
        base_l2, base_l1, cand_l2, cand_l1 = self._scripts(batch)
        # A run with the repair branch off only issues one L1/L2 plan per
        # decision, so its script must not contain candidate entries.
        effective_extra = (
            extra_candidates if bool(getattr(cfg, "repair_enabled", False)) else 0
        )
        l1, l2 = make_planners(
            base_l2,
            base_l1,
            candidate_l2=cand_l2,
            candidate_l1=cand_l1,
            decisions=decisions,
            extra_candidates=effective_extra,
        )
        if fail_on_calls:
            l1.fail_on_calls.update(fail_on_calls)

        torch.manual_seed(seed)
        planner = TwoLvlPlanner(
            l1_planner=l1,
            l2_planner=l2,
            l2_step_skip=6,
            exec_cfg=cfg,
            replan_every=4,
        )
        actions = []
        infos = []
        for k in range(decisions):
            result = planner.plan(
                stacked_zero_obs(batch),
                plan_size=3,
                decision_step=4 * k,
            )
            actions.append(result.level1.actions.detach().clone())
            infos.append(
                {
                    "repair": planner.last_repair_info,
                    "decision": planner.last_plan_decision,
                    "evidence": planner.last_candidate_evidence,
                }
            )
            planner.last_repair_info = None
            planner.last_plan_decision = None
            planner.last_candidate_evidence = None
        return actions, infos, planner

    # ------------------------------------------------------------------ tests
    def test_baseline_is_deterministic(self):
        cfg_off = ExecRepairConfig(enabled=False)
        first, _, _ = self._run(cfg_off, seed=7)
        second, _, _ = self._run(cfg_off, seed=7)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))

    def test_shadow_mode_matches_baseline(self):
        """All candidates rejected -> identical actions and RNG stream."""
        baseline, _, _ = self._run(ExecRepairConfig(enabled=False), seed=11)
        shadow, infos, _ = self._run(
            self._config(repair_force_reject=True), seed=11
        )
        for step, (a, b) in enumerate(zip(baseline, shadow)):
            self.assertTrue(
                torch.equal(a, b),
                f"shadow mode changed actions at decision {step}",
            )
        # The branch really did run: candidates were generated and rejected.
        self.assertIsNotNone(infos[0]["repair"])
        self.assertEqual(int(infos[0]["repair"].triggered.sum()), 2)
        self.assertEqual(int(infos[0]["repair"].accepted.sum()), 0)

    def test_candidate_collection_does_not_change_trajectory(self):
        """Collecting candidates in shadow mode must be observational only."""
        baseline, _, _ = self._run(ExecRepairConfig(enabled=False), seed=13)
        collected, infos, _ = self._run(
            self._config(repair_force_reject=True, collect_candidates=True),
            seed=13,
        )
        for a, b in zip(baseline, collected):
            self.assertTrue(torch.equal(a, b))

        evidence = infos[0]["evidence"]
        self.assertIsNotNone(evidence)
        self.assertEqual(len(evidence), 3)
        self.assertEqual(
            [item["candidate_id"] for item in evidence], [0, 1, 2]
        )
        self.assertTrue(
            all(item["actions"] is not None for item in evidence),
            "candidate actions must be captured for replay",
        )
        self.assertEqual(
            [item["source"] for item in evidence], ["hwm", "resample", "resample"]
        )

    def test_selected_candidate_is_consistent_per_environment(self):
        """The merged L1 plan must come from the candidate that was selected."""
        actions, infos, _ = self._run(self._config(), batch=3, decisions=1)
        repair = infos[0]["repair"]
        self.assertIsNotNone(repair)
        selected = repair.selected_candidate.tolist()
        self.assertEqual(selected[0], 1, "env 0 should accept candidate 1")
        self.assertEqual(selected[1], -1, "env 1 should reject its candidate")
        self.assertEqual(selected[2], -1, "env 2 should not be triggered")

        first_plan = actions[0]
        # Action dimension 0 encodes the plan endpoint (base plan: y=1.0 and
        # candidate: y=0.05), so the choice is directly observable.
        codes = first_plan[:, 0, 0].tolist()
        self.assertAlmostEqual(codes[0], 500.05, places=2)
        self.assertAlmostEqual(codes[1], 2000.5, places=2)
        self.assertAlmostEqual(codes[2], 3000.0, places=2)
        self.assertEqual(int(repair.accepted.sum()), 1)
        self.assertEqual(int(repair.triggered.sum()), 2)

    def test_warm_starts_follow_the_selected_candidate(self):
        """L1/L2 control caches must match the plan that was actually chosen."""
        _, infos, planner = self._run(self._config(), batch=3, decisions=1)
        repair = infos[0]["repair"]
        self.assertEqual(repair.selected_candidate.tolist(), [1, -1, -1])
        for name in ("l1_planner", "l2_planner"):
            with self.subTest(planner=name):
                planner_obj = getattr(planner, name)
                # env 0 kept the warm start of candidate 1 (the second plan).
                self.assertTrue(
                    torch.equal(
                        planner_obj.ctrls[0].U, planner_obj.ctrls[0].history[1]
                    ),
                    f"{name}: env 0 warm start is not the selected candidate",
                )
                # env 1 rejected the repair -> the original warm start is back.
                self.assertTrue(
                    torch.equal(
                        planner_obj.ctrls[1].U, planner_obj.ctrls[1].history[0]
                    ),
                    f"{name}: env 1 warm start was not restored",
                )

    def test_batch_isolation(self):
        """B = 1, 2, 5: no state or candidate leaks across environments."""
        for batch in (1, 2, 5):
            with self.subTest(batch=batch):
                actions, infos, planner = self._run(
                    self._config(), batch=batch, decisions=1
                )
                repair = infos[0]["repair"]
                self.assertIsNotNone(repair)
                if batch >= 2:
                    self.assertEqual(
                        int(repair.accepted.sum()),
                        1,
                        "only env 0 may accept the candidate",
                    )
                self.assertEqual(len(planner.l1_planner.ctrls), batch)
                self.assertEqual(len(planner.l2_planner.ctrls), batch)
                self.assertEqual(actions[0].shape[0], batch)

    def test_candidate_failure_falls_back_safely(self):
        """A failing candidate search must leave *this* decision untouched.

        Only the decision that failed is compared: a later decision is allowed
        to repair again, because that is the normal behaviour of the branch.
        """
        baseline, _, _ = self._run(
            ExecRepairConfig(enabled=False), seed=17, decisions=1
        )
        fallback, infos, planner = self._run(
            self._config(),
            seed=17,
            decisions=1,
            fail_on_calls={3: RuntimeError("boom")},
        )
        for a, b in zip(baseline, fallback):
            self.assertTrue(
                torch.equal(a, b),
                "a failed candidate search must leave the plan untouched",
            )
        self.assertIsNone(infos[0]["repair"])
        self.assertIn("boom", planner.last_repair_error or "")

    def test_candidate_failure_raises_in_debug_mode(self):
        with self.assertRaises(RuntimeError):
            self._run(
                self._config(repair_raise_errors=True),
                seed=19,
                fail_on_calls={3: RuntimeError("boom")},
            )

    def test_plan_identity_persists_across_decisions(self):
        actions, infos, planner = self._run(
            self._config(log_plan_events=True), batch=3, decisions=2
        )
        first = infos[0]["decision"]
        second = infos[1]["decision"]
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first["decision_id"], 0)
        self.assertEqual(second["decision_id"], 1)

        first_ids = [record.plan_id for record in first["records"]]
        second_ids = [record.plan_id for record in second["records"]]
        # env 0 accepts a repair at *both* decisions, so it gets a new id twice.
        self.assertTrue(first["records"][0].switched)
        self.assertNotEqual(first_ids[0], first["records"][0].previous_plan_id)
        self.assertTrue(second["records"][0].switched)
        self.assertNotEqual(second_ids[0], first_ids[0])
        # env 1 never switches: its plan (and therefore its id) persists and
        # only its age grows.
        self.assertFalse(first["records"][1].switched)
        self.assertEqual(second_ids[1], first_ids[1])
        self.assertFalse(second["records"][1].switched)
        self.assertEqual(first["records"][1].plan_age_replans, 1)
        self.assertEqual(second["records"][1].plan_age_replans, 2)
        self.assertEqual(first["records"][0].switch_reason, "repair_reselection")
        self.assertEqual(second["records"][1].switch_reason, "keep")


if __name__ == "__main__":
    unittest.main()
