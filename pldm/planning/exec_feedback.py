"""Online execution feedback (plan step 6).

The high-level re-ranking weight in :mod:`pldm.planning.subgoal_repair` decides
how much the low-level reachability term matters. A fixed weight assumes the
low-level world model is equally trustworthy at all times, which is exactly what
Step 2's execution diagnostics showed to be false: predicted and realised motion
drift apart.

This module keeps a tiny, parameter-free estimate of that drift per environment::

    ema_t = (1 - beta) * ema_{t-1} + beta * E_exec,t

and turns it into a weight multiplier::

    lambda_t = clip(lambda_0 * (1 + kappa * ema_t), lambda_min, lambda_max)

so that the planner trusts low-level reachability more when the low level has
recently been inaccurate.

Two important caveats, both from the plan:

* a large execution error does not necessarily mean the subgoal is unreachable -
  it can also mean the low-level dynamics model is wrong. A later version should
  keep "model trust" and "subgoal reachability" as separate penalties;
* ``kappa`` is scale dependent. ``exec_mse_obs`` on hard mazes is O(0.1-1), so
  ``kappa=1`` roughly doubles the weight at the observed error level. It must be
  calibrated on a development set, never on the reported 40-task test set.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class ExecutionFeedback:
    """Per-environment exponential moving average of the execution error."""

    def __init__(
        self,
        *,
        beta: float = 0.1,
        kappa: float = 1.0,
        lambda_min: float = 0.1,
        lambda_max: float = 10.0,
        n_envs: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        if not 0.0 < self.beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if self.lambda_min > self.lambda_max:
            raise ValueError("lambda_min must be <= lambda_max")
        self.device = device
        self.error_ema: Optional[torch.Tensor] = None
        self.updates = 0
        if n_envs is not None:
            self.reset(n_envs)

    @classmethod
    def from_config(cls, cfg: Any, *, n_envs: Optional[int] = None, device=None):
        """Build from ``ExecRepairConfig``; returns ``None`` when disabled."""
        if cfg is None or not bool(getattr(cfg, "exec_feedback_enabled", False)):
            return None
        return cls(
            beta=float(getattr(cfg, "feedback_beta", 0.1)),
            kappa=float(getattr(cfg, "feedback_kappa", 1.0)),
            lambda_min=float(getattr(cfg, "feedback_lambda_min", 0.1)),
            lambda_max=float(getattr(cfg, "feedback_lambda_max", 10.0)),
            n_envs=n_envs,
            device=device,
        )

    # ------------------------------------------------------------------ state
    def reset(self, n_envs: int) -> None:
        self.error_ema = torch.zeros(int(n_envs), dtype=torch.float32)
        if self.device is not None:
            self.error_ema = self.error_ema.to(self.device)
        self.updates = 0

    def _ensure(self, batch_size: int, device=None) -> torch.Tensor:
        if self.error_ema is None or self.error_ema.numel() != batch_size:
            self.error_ema = torch.zeros(batch_size, dtype=torch.float32)
        if device is not None and self.error_ema.device != device:
            self.error_ema = self.error_ema.to(device)
        return self.error_ema

    # ----------------------------------------------------------------- update
    def update(
        self,
        errors: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fold a batch of execution errors into the EMA.

        ``errors``: ``(B,)`` execution error per environment.
        ``mask``: optional ``(B,)`` bool; entries that are False are left
        untouched (e.g. environments whose prediction was not evaluated at this
        step). The first valid update initialises the EMA instead of pulling it
        towards zero.
        """
        if errors is None:
            return self.error_ema if self.error_ema is not None else torch.zeros(0)
        errors = errors.detach().to(dtype=torch.float32)
        ema = self._ensure(errors.numel(), device=errors.device)

        if mask is None:
            mask = torch.ones_like(errors, dtype=torch.bool)
        else:
            mask = mask.to(device=errors.device, dtype=torch.bool)

        if valid is not None:
            mask = mask & valid.to(device=errors.device, dtype=torch.bool)
        if not bool(mask.any()):
            return ema

        initialised = mask & (self.updates > 0)
        fresh = mask & (self.updates == 0)
        blended = (1.0 - self.beta) * ema + self.beta * errors
        ema = torch.where(initialised, blended, ema)
        ema = torch.where(fresh, errors, ema)
        self.error_ema = ema
        self.updates += 1
        return ema

    # ------------------------------------------------------------ consumption
    def penalty_weight(
        self,
        base_weight: float,
        *,
        n_envs: Optional[int] = None,
        device=None,
    ) -> torch.Tensor:
        """``lambda_t = clip(lambda_0 * (1 + kappa * ema), min, max)`` per env."""
        if n_envs is None:
            n_envs = (
                self.error_ema.numel()
                if self.error_ema is not None
                else 1
            )
        ema = self._ensure(int(n_envs), device=device)
        weight = float(base_weight) * (1.0 + self.kappa * ema)
        return weight.clamp(self.lambda_min, self.lambda_max)

    @property
    def mean_error(self) -> float:
        if self.error_ema is None or self.error_ema.numel() == 0:
            return 0.0
        return float(self.error_ema.mean().item())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "kappa": self.kappa,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "updates": self.updates,
            "mean_error": self.mean_error,
            "error_ema": self.error_ema,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"ExecutionFeedback(beta={self.beta}, kappa={self.kappa}, "
            f"updates={self.updates}, mean_error={self.mean_error:.4f})"
        )
