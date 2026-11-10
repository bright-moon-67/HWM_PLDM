"""Tests for the stage-2 candidate statistics."""

from __future__ import annotations

import math
import unittest

import torch

from pldm.planning.candidate_evaluator import (
    best_candidate_hit,
    bootstrap_mean_ci,
    candidate_ranking_summary,
    group_summary,
    groups_to_matrices,
    kendall_tau,
    oracle_candidate_gain,
    paired_bootstrap_diff_ci,
    pairwise_accuracy,
    selection_utility,
)


class CandidateEvaluatorTest(unittest.TestCase):
    def test_oracle_gain(self):
        returns = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 5.0]])
        gain = oracle_candidate_gain(returns)
        self.assertEqual(gain.tolist(), [2.0, 3.0])

    def test_oracle_gain_ignores_nan_candidates(self):
        returns = torch.tensor([[1.0], [float("nan")], [4.0]])
        gain = oracle_candidate_gain(returns)
        self.assertAlmostEqual(float(gain[0]), 3.0)

    def test_kendall_tau_perfect_and_reversed(self):
        scores = torch.tensor([[0.0], [1.0], [2.0]])  # lower is better
        aligned = torch.tensor([[3.0], [2.0], [1.0]])
        reversed_returns = torch.tensor([[1.0], [2.0], [3.0]])
        self.assertAlmostEqual(float(kendall_tau(scores, aligned)[0]), 1.0)
        self.assertAlmostEqual(float(kendall_tau(scores, reversed_returns)[0]), -1.0)

    def test_kendall_tau_is_nan_with_a_single_candidate(self):
        tau = kendall_tau(torch.tensor([[0.5]]), torch.tensor([[1.0]]))
        self.assertTrue(math.isnan(float(tau[0])))

    def test_pairwise_accuracy(self):
        scores = torch.tensor([[0.0], [1.0], [2.0]])
        self.assertEqual(
            float(pairwise_accuracy(scores, torch.tensor([[3.0], [2.0], [1.0]]))[0]),
            1.0,
        )
        self.assertEqual(
            float(pairwise_accuracy(scores, torch.tensor([[1.0], [2.0], [3.0]]))[0]),
            0.0,
        )

    def test_best_candidate_hit(self):
        scores = torch.tensor([[0.1, 0.3], [0.5, 0.2]])
        returns = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        hits = best_candidate_hit(scores, returns)
        self.assertEqual(hits.tolist(), [1.0, 1.0])

    def test_selection_utility_signs(self):
        # The model picks candidate 0 (cheapest); it is better than not acting.
        scores = torch.tensor([[0.1], [0.9]])
        returns = torch.tensor([[2.0], [1.0]])
        utility = selection_utility(scores, returns)
        self.assertAlmostEqual(utility["model_pick_regret_mean"], 2.0 - 2.0)
        self.assertEqual(utility["model_pick_better_rate"], 0.0)

        # Now the model picks the worse candidate.
        scores = torch.tensor([[0.9], [0.1]])
        utility = selection_utility(scores, returns)
        self.assertAlmostEqual(utility["model_pick_regret_mean"], 1.0)
        self.assertEqual(utility["model_pick_worse_rate"], 1.0)

    def test_bootstrap_is_deterministic_and_brackets_the_mean(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, float("nan")])
        first = bootstrap_mean_ci(values, n_boot=500, seed=3)
        second = bootstrap_mean_ci(values, n_boot=500, seed=3)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 2.5)
        self.assertLessEqual(first["lo"], first["mean"])
        self.assertGreaterEqual(first["hi"], first["mean"])

    def test_paired_bootstrap(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([0.0, 1.0, 1.0])
        result = paired_bootstrap_diff_ci(a, b, n_boot=200, seed=0)
        self.assertAlmostEqual(result["mean_diff"], 4.0 / 3.0)

    def test_groups_to_matrices_pads_with_nan(self):
        groups = [
            {
                "task_id": 0,
                "decision_id": 0,
                "horizon": 40,
                "candidates": [
                    {"candidate_id": 0, "model_score": 0.1, "real_return": 1.0},
                    {"candidate_id": 1, "model_score": 0.2, "real_return": 2.0},
                    {"candidate_id": 2, "model_score": 0.3, "real_return": 3.0},
                ],
            },
            {
                "task_id": 1,
                "decision_id": 0,
                "horizon": 40,
                "candidates": [
                    {"candidate_id": 0, "model_score": 0.4, "real_return": 2.0},
                    {"candidate_id": 1, "model_score": 0.5, "real_return": 1.0},
                ],
            },
            {
                "task_id": 2,
                "decision_id": 0,
                "horizon": 80,
                "candidates": [
                    {"candidate_id": 0, "model_score": 0.1, "real_return": 5.0},
                ],
            },
        ]
        scores, returns, task_ids, _ = groups_to_matrices(groups, horizon=40)
        self.assertEqual(scores.shape, (3, 2))
        self.assertEqual(task_ids, [0, 1])
        self.assertTrue(math.isnan(float(scores[2, 1])))
        self.assertTrue(math.isnan(float(returns[2, 1])))

    def test_candidate_ranking_summary_keys(self):
        scores = torch.tensor([[0.1, 0.2], [0.4, 0.3], [0.9, 0.5]])
        returns = torch.tensor([[3.0, 1.0], [2.0, 4.0], [1.0, 2.0]])
        summary = candidate_ranking_summary(scores, returns, n_boot=100, seed=0)
        for key in (
            "n_decisions",
            "n_candidates",
            "oracle_gain_mean",
            "oracle_gain_lo",
            "oracle_gain_hi",
            "oracle_gain_median",
            "oracle_gain_positive_rate",
            "kendall_tau_mean",
            "pairwise_accuracy_mean",
            "best_candidate_hit_rate",
        ):
            self.assertIn(key, summary)

    def test_group_summary_splits_tasks(self):
        scores = torch.tensor([[0.1, 0.2], [0.9, 0.3]])
        returns = torch.tensor([[3.0, 2.0], [1.0, 1.0]])
        summary = group_summary(["failed", "solved"], scores, returns, n_boot=50)
        self.assertEqual(set(summary), {"failed", "solved"})
        self.assertEqual(summary["failed"]["n_decisions"], 1)


if __name__ == "__main__":
    unittest.main()
