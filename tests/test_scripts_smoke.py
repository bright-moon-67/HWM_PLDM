"""End-to-end smoke test of the stage-2 collection / oracle scripts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _append(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


class ScriptSmokeTest(unittest.TestCase):
    def test_collect_then_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            diagnostics = run_dir / "diagnostics.jsonl"
            # Two decisions, three candidates, horizon 40. Candidate 1 is the
            # real best in decision 0 but the model does not pick it.
            for decision in range(2):
                for candidate in range(3):
                    _append(
                        diagnostics,
                        {
                            "event": "candidate_outcome",
                            "task_id": decision,
                            "env_index": 0,
                            "step": 4 * decision,
                            "decision_id": decision,
                            "plan_id": 100 + decision,
                            "horizon": 40,
                            "candidate_id": candidate,
                            "candidate_source": "hwm" if candidate == 0 else "resample",
                            "selected": candidate == 0,
                            "model_score": float(candidate),
                            "cross_mse": 0.1 * (candidate + 1),
                            "predicted_progress": 0.5,
                            "real_return": (
                                3.0 if candidate == 1 else 2.0 - candidate
                            ),
                            "graph_distance_before": 12,
                            "graph_distance_after": 9,
                            "replay_mode": "candidate_chunk+follow_up_hwm",
                        },
                    )
            # A replay problem must be counted, not crash the collector.
            _append(
                diagnostics,
                {
                    "event": "candidate_replay_unavailable",
                    "task_id": 0,
                    "reason": "no snapshots",
                },
            )

            prefix = Path(directory) / "out" / "cand"
            collect = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "collect_plan_candidates.py"),
                    "--diagnostics",
                    str(run_dir),
                    "--out",
                    str(prefix),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(collect.returncode, 0, collect.stderr)

            candidates_path = prefix.with_name(prefix.name + "_candidates.jsonl")
            summary_path = prefix.with_name(prefix.name + "_summary.json")
            self.assertTrue(candidates_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["n_groups"], 2)
            self.assertIn("40", summary["per_horizon"])
            self.assertEqual(
                summary["per_horizon"]["40"]["decisions_with_full_real_returns"], 2
            )
            self.assertEqual(summary["replay_errors"]["candidate_replay_unavailable"], 1)

            oracle_prefix = Path(directory) / "out" / "oracle"
            oracle = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "evaluate_candidate_oracle.py"),
                    "--candidates",
                    str(candidates_path),
                    "--horizons",
                    "40",
                    "--bootstrap",
                    "200",
                    "--out",
                    str(oracle_prefix),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(oracle.returncode, 0, oracle.stderr)
            oracle_path = oracle_prefix.with_name(
                oracle_prefix.name + "_oracle.json"
            )
            payload = json.loads(oracle_path.read_text(encoding="utf-8"))
            horizon = payload["horizons"]["40"]
            self.assertEqual(horizon["n_groups"], 2)
            # Candidate 1 returned 3.0, the baseline 2.0 -> oracle gain 1.0.
            self.assertAlmostEqual(horizon["overall"]["oracle_gain_mean"], 1.0)

    def test_oracle_demo_runs(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_candidate_oracle.py"),
                "--demo",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("oracle_gain", result.stdout)


if __name__ == "__main__":
    unittest.main()
