"""Tests for route indicators, env snapshots and graph-distance helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pldm.planning.env_snapshot import (
    restore_env,
    snapshot_env,
    supports_snapshot,
    verify_snapshot_roundtrip,
)
from pldm.planning.maze_metrics import (
    graph_distance,
    graph_progress,
    find_navigation_wrapper,
)
from pldm.planning.progress_estimator import RouteHistory
from tests.fakes import FakeEnv


class RouteHistoryTest(unittest.TestCase):
    def test_straight_line_has_full_efficiency(self):
        history = RouteHistory(1, window=4)
        metrics = None
        for x in range(5):
            metrics = history.update(torch.tensor([[float(x), 0.0]]))
        self.assertAlmostEqual(float(metrics["route_path_length"][0]), 4.0, places=5)
        self.assertAlmostEqual(float(metrics["route_efficiency"][0]), 1.0, places=5)
        self.assertEqual(float(metrics["route_revisit_count"][0]), 0.0)

    def test_back_and_forth_has_low_efficiency_and_revisits(self):
        history = RouteHistory(1, window=6, revisit_radius=0.1)
        points = [0.0, 1.0, 0.0, 1.0, 0.0]
        metrics = None
        for x in points:
            metrics = history.update(torch.tensor([[x, 0.0]]))
        self.assertAlmostEqual(float(metrics["route_path_length"][0]), 4.0, places=5)
        self.assertAlmostEqual(float(metrics["route_displacement"][0]), 0.0, places=5)
        self.assertAlmostEqual(float(metrics["route_efficiency"][0]), 0.0, places=5)
        # The final point (0.0) coincides with two earlier points.
        self.assertGreaterEqual(float(metrics["route_revisit_count"][0]), 2.0)

    def test_window_slides(self):
        history = RouteHistory(1, window=2)
        for x in [0.0, 1.0, 2.0, 3.0, 4.0]:
            metrics = history.update(torch.tensor([[x, 0.0]]))
        # Only the last two moves are inside the window.
        self.assertAlmostEqual(float(metrics["route_path_length"][0]), 2.0, places=5)

    def test_batch_independence_and_reset_mask(self):
        history = RouteHistory(2, window=3)
        history.update(torch.tensor([[0.0, 0.0], [5.0, 5.0]]))
        history.update(torch.tensor([[1.0, 0.0], [5.0, 5.0]]))
        metrics = history.update(torch.tensor([[2.0, 0.0], [5.0, 5.0]]))
        self.assertAlmostEqual(float(metrics["route_path_length"][0]), 2.0, places=5)
        self.assertAlmostEqual(float(metrics["route_path_length"][1]), 0.0, places=5)

        history.reset(torch.tensor([True, False]))
        metrics = history.update(torch.tensor([[2.0, 0.0], [5.0, 5.0]]))
        self.assertAlmostEqual(float(metrics["route_path_length"][0]), 0.0, places=5)
        self.assertEqual(len(history.history[0]), 1)

    def test_none_input_is_a_noop(self):
        history = RouteHistory(1, window=3)
        metrics = history.update(None)
        self.assertEqual(float(metrics["route_path_length"][0]), 0.0)
        metrics = history.update(torch.tensor([[float("nan"), 0.0]]))
        self.assertEqual(float(metrics["route_path_length"][0]), 0.0)


class SnapshotAndGraphTest(unittest.TestCase):
    def test_snapshot_roundtrip(self):
        env = FakeEnv(start_x=2.0)
        self.assertTrue(supports_snapshot(env))
        self.assertTrue(verify_snapshot_roundtrip(env, n_probe_steps=3))
        before = env.get_obs().copy()
        snapshot = snapshot_env(env)
        env.step(np.array([1.0], dtype=np.float32))
        restore_env(env, snapshot)
        np.testing.assert_allclose(env.get_obs(), before, atol=1e-8)

    def test_snapshot_from_dict_payload(self):
        env = FakeEnv(start_x=1.0)
        snapshot = snapshot_env(env).as_dict()
        env.step(np.array([2.0], dtype=np.float32))
        restore_env(env, snapshot)
        self.assertAlmostEqual(float(env.get_obs()[0]), 1.0, places=6)

    def test_graph_distance_and_progress(self):
        env = FakeEnv(start_x=0.0, goal_x=9.0)
        self.assertIsNotNone(find_navigation_wrapper(env))
        self.assertEqual(graph_distance(env, [0.0, 0.0]), 9)
        self.assertEqual(graph_distance(env, [9.0, 0.0]), 0)
        self.assertAlmostEqual(graph_progress(env, [0.0, 0.0], [4.0, 0.0]), 4.0)

    def test_missing_wrapper_returns_none(self):
        class NoNav:
            pass

        self.assertIsNone(graph_distance(NoNav(), [0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
