"""Route-level progress diagnostics (ECCHP stage 1).

The per-window execution error says whether *one* prediction was off; it does
not say whether the agent is making progress through the maze or oscillating
around the same junction.  Stage 1 of the research plan asks for a small set of
route-level indicators computed from the real XY position at every replanning
window:

.. math::

    L_t = \\sum_{i=t-W}^{t-1} \\|p_{i+1} - p_i\\|

    R_t = \\frac{\\|p_t - p_{t-W}\\|}{L_t + \\epsilon}

``L_t`` is the travelled path length and ``R_t`` the *trajectory efficiency*
(1.0 = straight line, 0.0 = returned to the start of the window).  A third
indicator counts how often the agent has been within ``revisit_radius`` of its
current position inside the window.

These are **diagnostics only**.  A legitimate detour around a wall has a low
efficiency, so none of these values may be used as a hard failure signal; the
research plan is explicit about that.

Inputs are real positions (``info["location"]`` / ``env.get_pos``), never
``pred_locations``: the predicted location is model output and using it here
would defeat the purpose of the diagnostic.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch

Point = Tuple[float, float]


class RouteHistory:
    """Sliding window of real XY positions, one deque per environment."""

    def __init__(
        self,
        n_envs: int = 0,
        *,
        window: int = 8,
        revisit_radius: float = 0.25,
        eps: float = 1e-8,
    ):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = int(window)
        self.revisit_radius = float(revisit_radius)
        self.eps = float(eps)
        self.n_envs = 0
        self._points: List[Deque[Point]] = []
        if n_envs:
            self._ensure(n_envs)

    # ------------------------------------------------------------------ state
    def _ensure(self, batch_size: int) -> None:
        if batch_size == self.n_envs:
            return
        if batch_size < self.n_envs:
            self._points = self._points[:batch_size]
        else:
            self._points.extend(
                deque(maxlen=self.window + 1)
                for _ in range(batch_size - self.n_envs)
            )
        self.n_envs = int(batch_size)

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        """Forget history for all environments, or only the ones in ``mask``."""
        if mask is None:
            for points in self._points:
                points.clear()
            return
        mask = mask.detach().cpu().to(dtype=torch.bool).view(-1).tolist()
        for i, flag in enumerate(mask[: self.n_envs]):
            if flag:
                self._points[i].clear()

    @property
    def history(self) -> List[List[Point]]:
        """Copy of the stored trajectories (oldest first)."""
        return [list(points) for points in self._points]

    # ----------------------------------------------------------------- update
    def update(self, xy: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Append the current real positions and recompute the indicators.

        ``xy`` is ``(B, 2)`` (extra columns are ignored).  Environments whose
        row is not finite are skipped; their history stays unchanged and the
        returned metrics repeat the last valid value.  ``None`` is a no-op that
        still returns zero metrics, so callers do not need conditionals.
        """
        if xy is not None:
            xy = xy.detach().cpu()
            xy = xy.reshape(xy.shape[0], -1)[:, :2]
            self._ensure(xy.shape[0])
            for i in range(self.n_envs):
                x, y = float(xy[i, 0]), float(xy[i, 1])
                if math.isfinite(x) and math.isfinite(y):
                    self._points[i].append((x, y))
        elif self.n_envs == 0:
            return self._empty_metrics(0)

        return self._metrics()

    # ---------------------------------------------------------------- metrics
    def _metrics(self) -> Dict[str, torch.Tensor]:
        batch = self.n_envs
        path_length = torch.zeros(batch, dtype=torch.float32)
        displacement = torch.zeros(batch, dtype=torch.float32)
        efficiency = torch.zeros(batch, dtype=torch.float32)
        revisit = torch.zeros(batch, dtype=torch.float32)
        steps_used = torch.zeros(batch, dtype=torch.float32)

        for i, points in enumerate(self._points):
            seq: Sequence[Point] = list(points)
            if len(seq) < 2:
                # A single point is a valid start: efficiency is defined as 1.0
                # so "not moving yet" does not look like a failed detour.
                efficiency[i] = 1.0 if seq else 0.0
                steps_used[i] = float(max(len(seq) - 1, 0))
                continue

            length = 0.0
            for (x0, y0), (x1, y1) in zip(seq[:-1], seq[1:]):
                length += math.hypot(x1 - x0, y1 - y0)
            disp = math.hypot(seq[-1][0] - seq[0][0], seq[-1][1] - seq[0][1])
            path_length[i] = length
            displacement[i] = disp
            efficiency[i] = disp / (length + self.eps)
            steps_used[i] = float(len(seq) - 1)

            current = seq[-1]
            count = 0
            for previous in seq[:-1]:
                if (
                    math.hypot(current[0] - previous[0], current[1] - previous[1])
                    < self.revisit_radius
                ):
                    count += 1
            revisit[i] = float(count)

        return {
            "route_path_length": path_length,
            "route_displacement": displacement,
            "route_efficiency": efficiency,
            "route_revisit_count": revisit,
            "route_window_steps": steps_used,
        }

    def _empty_metrics(self, batch: int) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros(batch, dtype=torch.float32)
        return {
            "route_path_length": zeros.clone(),
            "route_displacement": zeros.clone(),
            "route_efficiency": zeros.clone(),
            "route_revisit_count": zeros.clone(),
            "route_window_steps": zeros.clone(),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"RouteHistory(n_envs={self.n_envs}, window={self.window}, "
            f"revisit_radius={self.revisit_radius})"
        )
