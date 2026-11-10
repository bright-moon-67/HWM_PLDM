"""Tests for the plan-level diagnostics events (ECCHP stage 1)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pldm.planning.d4rl.enums import ExecRepairConfig
from pldm.planning.diagnostics import (
    EXECUTION_OUTCOME_EVENT,
    PLAN_DECISION_EVENT,
    PLAN_TRANSITION_EVENT,
    DiagnosticsRecorder,
    ExecutionOutcomeRecord,
    PlanTransitionRecord,
)
from pldm.planning.plan_memory import PlanMemory


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class DiagnosticsEventTest(unittest.TestCase):
    def test_recorder_is_disabled_without_a_switch(self):
        cfg = ExecRepairConfig()  # everything off
        recorder = DiagnosticsRecorder(exec_cfg=cfg, output_dir=".")
        self.assertFalse(recorder.enabled)

    def test_log_plan_events_enables_the_recorder(self):
        cfg = ExecRepairConfig(enabled=True, log_plan_events=True)
        recorder = DiagnosticsRecorder(exec_cfg=cfg, output_dir=".")
        self.assertTrue(recorder.enabled)

    def test_all_three_events_are_written_and_linked(self):
        cfg = ExecRepairConfig(
            enabled=True, log_plan_events=True, diagnostics_filename="d.jsonl"
        )
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticsRecorder(exec_cfg=cfg, output_dir=directory)
            memory = PlanMemory(2, task_id_offset=5)
            memory.begin_decision()
            decision = memory.update(
                torch.tensor([True, False]),
                ["repair_reselection", "hwm"],
                ["repair_reselection", "keep"],
                step=8,
                cross_mse=torch.tensor([0.4, 0.1]),
            )
            for record in decision:
                recorder.record_plan_decision(record, step=8)
                recorder.record_plan_transition(
                    PlanTransitionRecord(
                        task_id=record.task_id,
                        env_index=record.env_index,
                        step=8,
                        decision_id=record.decision_id,
                        plan_id_before=record.previous_plan_id,
                        plan_id_after=record.plan_id,
                        switched=record.switched,
                        reason=record.switch_reason,
                        plan_age_replans=record.plan_age_replans,
                    ),
                    step=8,
                )
            recorder.record_execution_outcome(
                ExecutionOutcomeRecord(
                    task_id=5,
                    env_index=0,
                    step=12,
                    decision_id=0,
                    plan_id=decision[0].plan_id,
                    start_xy=[0.0, 1.0],
                    end_xy=[1.0, 1.0],
                    displacement=1.0,
                    route_path_length=1.0,
                    route_efficiency=1.0,
                    route_revisit_count=0.0,
                    real_progress_h=0.5,
                    stuck=False,
                ),
                step=12,
            )
            recorder.close()

            records = _read_jsonl(Path(directory) / "d.jsonl")

        events = [r["event"] for r in records]
        self.assertEqual(events.count(PLAN_DECISION_EVENT), 2)
        self.assertEqual(events.count(PLAN_TRANSITION_EVENT), 2)
        self.assertEqual(events.count(EXECUTION_OUTCOME_EVENT), 1)

        decisions = [r for r in records if r["event"] == PLAN_DECISION_EVENT]
        self.assertEqual([r["task_id"] for r in decisions], [5, 6])
        self.assertEqual([r["task_ids"] for r in decisions], [[5], [6]])
        self.assertTrue(decisions[0]["switched"])
        self.assertEqual(decisions[0]["candidate_source"], "repair_reselection")
        self.assertAlmostEqual(decisions[0]["cross_mse"], 0.4)
        self.assertEqual(decisions[0]["decision_id"], 0)

        transitions = [r for r in records if r["event"] == PLAN_TRANSITION_EVENT]
        self.assertEqual(transitions[0]["reason"], "repair_reselection")
        self.assertEqual(transitions[0]["plan_id_after"], decisions[0]["plan_id"])
        self.assertFalse(transitions[1]["switched"])

        outcome = [r for r in records if r["event"] == EXECUTION_OUTCOME_EVENT][0]
        self.assertEqual(outcome["plan_id"], decisions[0]["plan_id"])
        self.assertEqual(outcome["start_xy"], [0.0, 1.0])
        self.assertEqual(outcome["route_efficiency"], 1.0)
        self.assertFalse(outcome["stuck"])

    def test_outcome_accepts_plain_dicts(self):
        cfg = ExecRepairConfig(enabled=True, log_plan_events=True)
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticsRecorder(exec_cfg=cfg, output_dir=directory)
            recorder.record_execution_outcome(
                {
                    "task_id": 1,
                    "env_index": 0,
                    "step": 4,
                    "decision_id": 2,
                    "plan_id": 3,
                    "route_path_length": 0.5,
                }
            )
            recorder.close()
            records = _read_jsonl(recorder.path)
        self.assertEqual(records[0]["event"], EXECUTION_OUTCOME_EVENT)
        self.assertEqual(records[0]["task_id"], 1)


if __name__ == "__main__":
    unittest.main()
