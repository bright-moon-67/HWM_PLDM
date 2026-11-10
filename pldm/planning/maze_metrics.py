"""Offline graph-distance helpers for candidate diagnostics (ECCHP stage 2).

The stage-2 oracle uses the *true* shortest-path distance on the maze graph as
the real outcome of a candidate::

    R_H(P_i) = D_graph(s_t, g) - D_graph(s_{t+H}^{(i)}, g)

``D_graph`` is available only in simulation and only for diagnostics / offline
labels -- the online HWM controller never calls it.  This module keeps that
constraint explicit by depending only on the wrapper API
(``find_shortest_path`` / ``xy_to_ij``) and returning ``None`` when the agent
is outside the navigation graph instead of inventing a number.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def find_navigation_wrapper(env: Any):
    """Return the wrapper that can map XY <-> grid indices, or ``None``."""
    for obj in _iter_chain(env):
        if callable(getattr(obj, "find_shortest_path", None)) and callable(
            getattr(obj, "xy_to_ij", None)
        ):
            return obj
    return None


def _iter_chain(env: Any):
    seen = set()
    stack = [env]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        for attr in ("env", "unwrapped", "_env"):
            child = getattr(obj, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)


def _as_xy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 2:
        return None
    return array[:2]


def graph_distance(env: Any, xy: Any, goal_xy: Any = None) -> Optional[int]:
    """Number of grid cells on the shortest path from ``xy`` to the goal.

    Returns ``None`` when the wrapper is missing or either point is off-graph.
    The distance counts *moves*, matching ``len(find_shortest_path(...)[0])``.
    """
    wrapper = find_navigation_wrapper(env)
    if wrapper is None:
        return None
    start = _as_xy(xy)
    goal = _as_xy(goal_xy if goal_xy is not None else _goal_xy(env))
    if start is None or goal is None:
        return None
    try:
        start_ij = wrapper.xy_to_ij(start)
        goal_ij = wrapper.xy_to_ij(goal)
        path, _ = wrapper.find_shortest_path(start_ij, goal_ij)
    except Exception:  # pragma: no cover - defensive, wrapper specific
        return None
    if path is None:
        return None
    return int(len(path))


def graph_progress(env: Any, start_xy: Any, end_xy: Any) -> Optional[float]:
    """``D_graph(start) - D_graph(end)`` towards the environment's goal."""
    goal = _goal_xy(env)
    before = graph_distance(env, start_xy, goal)
    after = graph_distance(env, end_xy, goal)
    if before is None or after is None:
        return None
    return float(before - after)


def euclidean_distance(env: Any, xy: Any, goal_xy: Any = None) -> Optional[float]:
    """Fallback metric when the navigation wrapper is unavailable."""
    start = _as_xy(xy)
    goal = _as_xy(goal_xy if goal_xy is not None else _goal_xy(env))
    if start is None or goal is None:
        return None
    return float(np.linalg.norm(start - goal))


def _goal_xy(env: Any) -> Any:
    for obj in _iter_chain(env):
        getter = getattr(obj, "get_target", None)
        if callable(getter):
            try:
                return getter()
            except Exception:  # pragma: no cover - defensive
                continue
    return None
