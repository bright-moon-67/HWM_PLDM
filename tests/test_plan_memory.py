"""Tests for plan identity / persistence bookkeeping (ECCHP stage 1)."""

from __future__ import annotations

import json
import unittest

import torch

from pldm.planning.plan_memory import (
    REASON_KEEP,
    REASON_RETRY,
    SOURCE_HWM,
    SOURCE_RETRY,
    PlanMemory,
)


class PlanMemoryTest(unittest.TestCase):
    def test_initial_plan_ids_are_unique_per_environment(self):
        memory = PlanMemory(4)
        ids = memory.plan_ids.tolist()
        self.assertEqual(len(set(ids)), 4)
        self.assertTrue(all(age == 0 for age in memory.plan_ages.tolist()))

    def test_keep_preserves_plan_identity_and_ages_it(self):
        memory = PlanMemory(3)
        before = memory.plan_ids.clone()
        memory.begin_decision()
        records = memory.update(
            torch.zeros(3, dtype=torch.bool), [SOURCE_HWM] * 3, [REASON_KEEP] * 3
        )
        self.assertEqual(memory.plan_ids.tolist(), before.tolist())
        self.assertEqual([r.plan_age_replans for r in records], [1, 1, 1])
        self.assertFalse(any(r.switched for r in records))

    def test_switch_creates_a_new_id_only_for_that_environment(self):
        memory = PlanMemory(3)
        before = memory.plan_ids.clone()
        memory.begin_decision()
        switched = torch.tensor([True, False, False])
        records = memory.update(
            switched,
            ["repair_reselection", SOURCE_HWM, SOURCE_HWM],
            ["repair_reselection", REASON_KEEP, REASON_KEEP],
        )
        after = memory.plan_ids
        self.assertNotEqual(int(after[0]), int(before[0]))
        self.assertEqual(int(after[1]), int(before[1]))
        self.assertEqual(int(after[2]), int(before[2]))
        self.assertEqual(records[0].plan_age_replans, 0)
        self.assertEqual(records[1].plan_age_replans, 1)
        self.assertEqual(records[0].previous_plan_id, int(before[0]))
        self.assertNotEqual(records[0].plan_id, records[0].previous_plan_id)

    def test_decision_ids_are_monotonic(self):
        memory = PlanMemory(2)
        first = memory.begin_decision()
        memory.update(
            torch.zeros(2, dtype=torch.bool), [SOURCE_HWM] * 2, [REASON_RETRY] * 2
        )
        second = memory.begin_decision()
        self.assertEqual(first, 0)
        self.assertEqual(second, 1)

    def test_retry_keeps_the_plan(self):
        memory = PlanMemory(1)
        before = int(memory.plan_ids[0])
        memory.begin_decision()
        memory.update(
            torch.tensor([False]), [SOURCE_RETRY], [REASON_RETRY]
        )
        self.assertEqual(int(memory.plan_ids[0]), before)
        self.assertEqual(memory.sources[0], SOURCE_RETRY)

    def test_batch_size_change_reallocates(self):
        memory = PlanMemory(2)
        memory.begin_decision()
        records = memory.update(
            torch.tensor([True, False, True]),
            ["repair_reselection", SOURCE_HWM, "repair_reselection"],
            ["repair_reselection", REASON_KEEP, "repair_reselection"],
        )
        self.assertEqual(len(records), 3)
        self.assertEqual(memory.n_envs, 3)

    def test_task_ids_follow_offset(self):
        memory = PlanMemory(2, task_id_offset=10)
        memory.begin_decision()
        records = memory.update(
            torch.zeros(2, dtype=torch.bool), [SOURCE_HWM] * 2, [REASON_KEEP] * 2
        )
        self.assertEqual([r.task_id for r in records], [10, 11])

    def test_records_are_json_serialisable(self):
        memory = PlanMemory(1)
        memory.begin_decision()
        records = memory.update(
            torch.tensor([True]),
            ["repair_projection"],
            ["repair_projection"],
            step=12,
            n_candidates=3,
            high_level_cost=torch.tensor([0.75]),
            cross_mse=torch.tensor([0.125]),
            predicted_progress=torch.tensor([0.4]),
        )
        payload = records[0].as_dict()
        payload["start_xy"] = [0.5, 1.5]
        text = json.dumps(payload)
        self.assertIn("repair_projection", text)
        self.assertAlmostEqual(payload["cross_mse"], 0.125)
        self.assertEqual(payload["n_candidates"], 3)
        self.assertEqual(payload["step"], 12)

    def test_state_snapshot_is_plain_python(self):
        memory = PlanMemory(2)
        memory.begin_decision()
        memory.update(
            torch.zeros(2, dtype=torch.bool), [SOURCE_HWM] * 2, [REASON_KEEP] * 2
        )
        state = memory.state()
        self.assertEqual(state["decision_id"], 0)
        json.dumps(state)  # must not raise


if __name__ == "__main__":
    unittest.main()
