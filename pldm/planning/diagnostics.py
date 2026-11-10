"""Diagnostics plumbing for the execution-aware HWM research line.

Step 0 only ships the recorder: it is **not** wired into the planning loop yet.
``TwoLvlPlanner`` already stores the ``ExecRepairConfig`` and the chunk offset,
so Step 1 can call this recorder right after the low-level plan is produced.

Design constraints:

* never write full prediction tensors to disk (they are large); only scalars
  and per-task vectors are serialised;
* one JSONL line per planning decision so partial / crashed runs stay readable;
* no dependency beyond ``torch`` for turning tensors into python floats.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from pldm.models.utils import flatten_conv_output

ScalarLike = Union[int, float, bool, torch.Tensor]

#: Event names introduced by ECCHP stage 1.  They are written to the same
#: JSONL file as ``plan_gap`` / ``retry`` / ``repair`` and linked by the
#: ``decision_id`` / ``plan_id`` pair.
PLAN_DECISION_EVENT = "plan_decision"
EXECUTION_OUTCOME_EVENT = "execution_outcome"
PLAN_TRANSITION_EVENT = "plan_transition"


def to_jsonable(value: Any) -> Any:
    """Convert tensors / containers into JSON-serialisable python objects.

    Tensor shapes are preserved (``(B, 2)`` becomes a list of pairs), which is
    the ``schema_version=2`` behaviour. Callers that read older logs must
    tolerate the flattened ``schema_version=1`` layout.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    if hasattr(value, "tolist"):  # numpy scalars / arrays
        return to_jsonable(value.tolist())
    return str(value)


def _py_float(value: Any) -> Optional[float]:
    """Scalar -> python float, ``None`` when the value is missing/not scalar."""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().double().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _py_xy(value: Any) -> Optional[List[float]]:
    """``(..., 2)`` tensor / sequence -> ``[x, y]`` or ``None``."""
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
    return None


@dataclass
class ExecutionOutcomeRecord:
    """What actually happened during one replanning window.

    Connects back to :class:`pldm.planning.plan_memory.PlanDecision` through
    ``decision_id``/``plan_id``, so a bad window can be attributed either to a
    bad plan or to unstable execution of a plan that was otherwise fine.
    """

    task_id: int
    env_index: int
    step: int
    decision_id: int
    plan_id: int
    start_xy: Optional[List[float]] = None
    end_xy: Optional[List[float]] = None
    predicted_xy: Optional[List[float]] = None
    displacement: Optional[float] = None
    predicted_displacement: Optional[float] = None
    subgoal_dist_before: Optional[float] = None
    subgoal_dist_after: Optional[float] = None
    real_progress_h: Optional[float] = None
    pred_progress_h: Optional[float] = None
    execution_progress_gap: Optional[float] = None
    exec_mse_obs: Optional[float] = None
    exec_mse_xy: Optional[float] = None
    route_path_length: Optional[float] = None
    route_efficiency: Optional[float] = None
    route_revisit_count: Optional[float] = None
    stuck: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": int(self.task_id),
            "env_index": int(self.env_index),
            "step": int(self.step),
            "decision_id": int(self.decision_id),
            "plan_id": int(self.plan_id),
            "start_xy": _py_xy(self.start_xy),
            "end_xy": _py_xy(self.end_xy),
            "predicted_xy": _py_xy(self.predicted_xy),
            "displacement": _py_float(self.displacement),
            "predicted_displacement": _py_float(self.predicted_displacement),
            "subgoal_dist_before": _py_float(self.subgoal_dist_before),
            "subgoal_dist_after": _py_float(self.subgoal_dist_after),
            "real_progress_h": _py_float(self.real_progress_h),
            "pred_progress_h": _py_float(self.pred_progress_h),
            "execution_progress_gap": _py_float(self.execution_progress_gap),
            "exec_mse_obs": _py_float(self.exec_mse_obs),
            "exec_mse_xy": _py_float(self.exec_mse_xy),
            "route_path_length": _py_float(self.route_path_length),
            "route_efficiency": _py_float(self.route_efficiency),
            "route_revisit_count": _py_float(self.route_revisit_count),
            "stuck": None if self.stuck is None else bool(self.stuck),
        }


@dataclass
class PlanTransitionRecord:
    """Start/end of a high-level plan inside the replanning timeline."""

    task_id: int
    env_index: int
    step: int
    decision_id: int
    plan_id_before: int
    plan_id_after: int
    switched: bool
    reason: str
    plan_age_replans: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": int(self.task_id),
            "env_index": int(self.env_index),
            "step": int(self.step),
            "decision_id": int(self.decision_id),
            "plan_id_before": int(self.plan_id_before),
            "plan_id_after": int(self.plan_id_after),
            "switched": bool(self.switched),
            "reason": str(self.reason),
            "plan_age_replans": int(self.plan_age_replans),
        }


class DiagnosticsRecorder:
    """Append-only JSONL recorder for planning diagnostics.

    Parameters
    ----------
    exec_cfg:
        ``pldm.planning.d4rl.enums.ExecRepairConfig`` instance. Typed as
        ``Any`` on purpose to keep this module free of import cycles.
    output_dir:
        Where the JSONL file lives. Falls back to ``exec_cfg.output_dir`` and
        finally to "disabled" (``path`` is ``None``).
    chunk_offset:
        Offset of the current chunk inside ``MPCEvaluator.envs``. Global task
        ids are ``exec_cfg.task_id_offset + chunk_offset + env_index``.
    enabled:
        Explicit override used by unit tests; ``None`` follows the config
        (``enabled and log_diagnostics``).
    """

    def __init__(
        self,
        exec_cfg: Any = None,
        output_dir: Optional[Union[str, Path]] = None,
        chunk_offset: int = 0,
        filename: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.exec_cfg = exec_cfg
        self.chunk_offset = int(chunk_offset)
        self.task_id_offset = int(getattr(exec_cfg, "task_id_offset", 0) or 0)
        self._enabled_override = enabled

        if output_dir is None:
            cfg_dir = getattr(exec_cfg, "output_dir", "") if exec_cfg else ""
            output_dir = cfg_dir or None
        self.output_dir = Path(output_dir) if output_dir else None

        self.filename = (
            filename
            or getattr(exec_cfg, "diagnostics_filename", None)
            or "diagnostics.jsonl"
        )
        self._handle = None

    # ------------------------------------------------------------------ config
    @property
    def enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        cfg = self.exec_cfg
        if cfg is None:
            return False
        return bool(
            getattr(cfg, "enabled", False)
            and (
                getattr(cfg, "log_diagnostics", False)
                or getattr(cfg, "retry_enabled", False)
                # A repair-only run still needs its events recorded.
                or getattr(cfg, "repair_enabled", False)
                # ECCHP stage 1: plan-level events are worth recording even
                # when the per-window gap diagnostics are off.
                or getattr(cfg, "log_plan_events", False)
                # Stage-2 candidate collection also needs the event stream.
                or getattr(cfg, "collect_candidates", False)
            )
        )

    @property
    def path(self) -> Optional[Path]:
        if self.output_dir is None:
            return None
        return self.output_dir / self.filename

    def task_ids(self, batch_size: int) -> List[int]:
        """Global task ids covered by the current chunk."""
        base = self.task_id_offset + self.chunk_offset
        return [base + i for i in range(int(batch_size))]

    # ----------------------------------------------------------------- writing
    def record(
        self,
        event: str,
        *,
        batch_size: Optional[int] = None,
        task_ids: Optional[Sequence[int]] = None,
        step: Optional[int] = None,
        level: Optional[Union[int, str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one diagnostics record. No-op when disabled."""
        if not self.enabled:
            return

        payload: Dict[str, Any] = {
            "schema_version": 2,
            "time": round(time.time(), 3),
            "event": str(event),
        }
        if level is not None:
            payload["level"] = level
        if step is not None:
            payload["step"] = int(step)
        if task_ids is not None:
            payload["task_ids"] = [int(t) for t in task_ids]
        elif batch_size is not None:
            payload["task_ids"] = self.task_ids(batch_size)
        if metrics:
            payload.update({str(k): to_jsonable(v) for k, v in metrics.items()})

        handle = self._ensure_open()
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()

    # --------------------------------------------------- plan-level events
    def record_plan_decision(self, record: Any, *, step: Optional[int] = None) -> None:
        """Append a ``plan_decision`` line for one environment.

        ``record`` is a :class:`pldm.planning.plan_memory.PlanDecision` (typed
        as ``Any`` to avoid an import cycle).
        """
        payload = record.as_dict() if hasattr(record, "as_dict") else dict(record)
        self.record(
            PLAN_DECISION_EVENT,
            task_ids=[int(payload.get("task_id", 0))],
            step=int(payload.get("step", 0) if step is None else step),
            level="hier",
            metrics=payload,
        )

    def record_execution_outcome(
        self, record: Any, *, step: Optional[int] = None
    ) -> None:
        """Append an ``execution_outcome`` line for one environment."""
        payload = record.as_dict() if hasattr(record, "as_dict") else dict(record)
        self.record(
            EXECUTION_OUTCOME_EVENT,
            task_ids=[int(payload.get("task_id", 0))],
            step=int(payload.get("step", 0) if step is None else step),
            level="hier",
            metrics=payload,
        )

    def record_plan_transition(
        self, record: Any, *, step: Optional[int] = None
    ) -> None:
        """Append a ``plan_transition`` line for one environment."""
        payload = record.as_dict() if hasattr(record, "as_dict") else dict(record)
        self.record(
            PLAN_TRANSITION_EVENT,
            task_ids=[int(payload.get("task_id", 0))],
            step=int(payload.get("step", 0) if step is None else step),
            level="hier",
            metrics=payload,
        )

    def _ensure_open(self):
        if self._handle is not None:
            return self._handle
        path = self.path
        if path is None:
            raise RuntimeError(
                "DiagnosticsRecorder has no output directory; set "
                "exec_repair.output_dir or pass output_dir explicitly."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        return self._handle

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # ---------------------------------------------------------------- context
    def __enter__(self) -> "DiagnosticsRecorder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"DiagnosticsRecorder(enabled={self.enabled}, "
            f"path={self.path}, task_id_offset={self.task_id_offset}, "
            f"chunk_offset={self.chunk_offset})"
        )


def project_l1_encoding_to_l2(encodings: torch.Tensor, objective: Any) -> torch.Tensor:
    """Project fused level-1 encodings into the level-2 representation.

    This mirrors the representation mapping used by ``RunningCost.__call__`` in
    ``pldm/planning/planners/mppi_planner.py``. Reusing that exact mapping is
    important: the low-level MPC objective already compares L1 predictions
    against the L2 subgoal this way, so diagnostics must not invent a different
    projection.

    ``objective`` is a ``ReprTargetMPCObjective`` whose ``model`` is the level-1
    model and whose ``pred_encoder`` is the level-2 backbone.

    Accepts a single step ``(B, C, H, W)`` or a full prediction
    ``(T, B, C, H, W)`` and returns ``(B, D)`` / ``(T, B, D)``.
    """
    if not torch.is_tensor(encodings):
        raise TypeError(f"encodings must be a tensor, got {type(encodings)}")

    pred_encoder = getattr(objective, "pred_encoder", None)
    if pred_encoder is None:
        return flatten_conv_output(encodings)

    if encodings.dim() == 5:
        t, b = encodings.shape[0], encodings.shape[1]
        flat = encodings.reshape(t * b, *encodings.shape[2:])
    elif encodings.dim() == 4:
        t, b = None, None
        flat = encodings
    else:
        raise ValueError(
            f"expected a 4D or 5D encoding tensor, got {tuple(encodings.shape)}"
        )

    if bool(getattr(pred_encoder, "using_proprio", False)):
        backbone = objective.model.backbone
        obs_dim_info = getattr(backbone, "output_obs_dim", None)
        proprio_dim_info = getattr(backbone, "output_proprio_dim", None)
        if obs_dim_info is None or proprio_dim_info is None:
            raise RuntimeError(
                "cannot split fused L1 encodings: backbone is missing "
                "output_obs_dim/output_proprio_dim"
            )
        obs_channels = (
            obs_dim_info[0] if isinstance(obs_dim_info, tuple) else int(obs_dim_info)
        )
        proprio_channels = (
            proprio_dim_info[0]
            if isinstance(proprio_dim_info, tuple)
            else int(proprio_dim_info)
        )
        if obs_channels <= 0 or proprio_channels <= 0:
            raise RuntimeError(
                "cannot split fused L1 encodings: "
                f"output_obs_dim={obs_dim_info}, "
                f"output_proprio_dim={proprio_dim_info}"
            )
        obs_state = flat[:, :obs_channels]
        proprio_state = flat[:, obs_channels : obs_channels + proprio_channels]
        projected = pred_encoder(obs_state, proprio=proprio_state).encodings
    else:
        projected = pred_encoder(flat).encodings

    projected = flatten_conv_output(projected)
    if t is not None:
        projected = projected.view(t, b, *projected.shape[1:])
    return projected


def cross_level_metrics(
    l1_pred_obs: torch.Tensor,
    l2_pred_obs: torch.Tensor,
    objective: Any,
    *,
    subgoal_index: int = 1,
    progress_index: int = 4,
) -> Dict[str, torch.Tensor]:
    """Cross-level prediction disagreement for one planning decision.

    ``pred_obs`` tensors include the current timestep at index 0 (the MPC
    objective drops it via ``encodings[1:]``), so ``subgoal_index=1`` compares
    the L2 subgoal one macro step ahead with the terminal state of the L1 plan,
    which is exactly ``l2_step_skip`` primitives ahead.

    ``progress_index`` selects the primitive step inside the L1 prediction at
    which "predicted progress" is measured. It should equal the number of
    primitives that will actually be executed before the next replan
    (``replan_every``); otherwise the progress metric is evaluated at the wrong
    horizon. It is clamped to the available horizon.

    Returns a dict of per-environment tensors of shape ``(B,)``:
    ``cross_mse``, ``cross_l2``, ``l2_subgoal_norm``, ``l1_terminal_norm``,
    ``path_cost``, ``initial_dist``, ``pred_dist_h``, ``pred_dist_T``,
    ``pred_progress_h``, ``pred_progress_T``, ``progress_horizon``.
    ``subgoal_l2`` is additionally returned with shape ``(B, D)``; it is *not*
    JSON-serialisable-friendly for logging and must stay out of diagnostics.
    """
    if not torch.is_tensor(l1_pred_obs) or not torch.is_tensor(l2_pred_obs):
        raise TypeError("l1_pred_obs and l2_pred_obs must be tensors")
    if l1_pred_obs.dim() < 4 or l2_pred_obs.dim() < 4:
        raise ValueError(
            "expected (T, B, ...) predictions, got "
            f"{tuple(l1_pred_obs.shape)} and {tuple(l2_pred_obs.shape)}"
        )
    if l2_pred_obs.shape[0] <= subgoal_index:
        raise RuntimeError(
            f"l2_pred_obs has {l2_pred_obs.shape[0]} steps, cannot index "
            f"subgoal at {subgoal_index}"
        )

    with torch.no_grad():
        z_low_all = project_l1_encoding_to_l2(l1_pred_obs, objective)
        z_high = flatten_conv_output(l2_pred_obs[subgoal_index])

    if z_low_all.dim() != 3:
        raise RuntimeError(
            "expected (T, B, D) projected L1 predictions, got "
            f"{tuple(z_low_all.shape)}"
        )
    if z_low_all.shape[1:] != z_high.shape:
        raise RuntimeError(
            "cross-level representation mismatch: L1->L2 projection has shape "
            f"{tuple(z_low_all.shape[1:])} but the L2 subgoal has shape "
            f"{tuple(z_high.shape)}"
        )

    z_low = z_low_all[-1]
    diff = z_low - z_high

    # Path cost: same cross-level discrepancy accumulated over the whole L1
    # horizon (excluding the current state), matching the spirit of
    # ``ReprTargetMPCObjective`` with ``sum_all_diffs=True``.
    if z_low_all.shape[0] > 1:
        path_diff = z_low_all[1:] - z_high.unsqueeze(0)
        path_cost = path_diff.pow(2).mean(dim=-1).sum(dim=0)
    else:  # pragma: no cover - degenerate horizon
        path_cost = torch.zeros_like(diff.pow(2).mean(dim=1))

    # ---------------------------------------------------------------- progress
    # How much of the distance to the subgoal does the L1 plan predict to close
    # if we execute ``progress_index`` primitives? This is the quantity the
    # retry trigger *should* care about: a plan that is far from the subgoal but
    # moving quickly towards it is not necessarily a bad plan.
    horizon = int(z_low_all.shape[0])
    if progress_index is None:
        progress_index = horizon - 1
    progress_index = int(progress_index)
    progress_index = max(1, min(progress_index, horizon - 1)) if horizon > 1 else 0

    z_start = z_low_all[0]
    d0 = (z_start - z_high).pow(2).mean(dim=1)
    d_h = (z_low_all[progress_index] - z_high).pow(2).mean(dim=1)
    d_T = (z_low - z_high).pow(2).mean(dim=1)
    denom = d0.clamp_min(1e-8)

    metrics = {
        "cross_mse": diff.pow(2).mean(dim=1).detach(),
        "cross_l2": torch.linalg.vector_norm(diff, dim=1).detach(),
        "l2_subgoal_norm": torch.linalg.vector_norm(z_high, dim=1).detach(),
        "l1_terminal_norm": torch.linalg.vector_norm(z_low, dim=1).detach(),
        "path_cost": path_cost.detach(),
        "initial_dist": d0.detach(),
        "pred_dist_h": d_h.detach(),
        "pred_dist_T": d_T.detach(),
        "pred_progress_h": ((d0 - d_h) / denom).detach(),
        "pred_progress_T": ((d0 - d_T) / denom).detach(),
        "progress_horizon": torch.full_like(
            d0.detach(), float(progress_index), dtype=torch.float32
        ),
        "subgoal_l2": z_high.detach(),
    }

    # Fixed short/long horizons so different runs can be compared without
    # depending on the configured ``replan_every`` (clamped when the L1 horizon
    # is shorter than the requested step).
    for fixed_index in (4, 10):
        index = min(max(fixed_index, 0), horizon - 1)
        dist = (z_low_all[index] - z_high).pow(2).mean(dim=1)
        metrics[f"pred_dist_{fixed_index}"] = dist.detach()
        metrics[f"pred_progress_{fixed_index}"] = ((d0 - dist) / denom).detach()

    return metrics


def execution_prediction_index(
    replan_every: int, num_pred_steps: int
) -> Optional[int]:
    """Index of the prediction that matches ``replan_every`` executed steps.

    ``pred_obs``/``locations`` include the current state at index 0, so the
    prediction after ``replan_every`` primitive steps lives at index
    ``replan_every``. Returns ``None`` when the plan is too short.
    """
    if replan_every <= 0 or num_pred_steps <= replan_every:
        return None
    return int(replan_every)


#: Keys of :func:`cross_level_metrics` that may be written to JSONL.
#: ``subgoal_l2`` is a large ``(B, D)`` tensor used only in-process.
LOGGABLE_GAP_KEYS = (
    "cross_mse",
    "cross_l2",
    "l2_subgoal_norm",
    "l1_terminal_norm",
    "path_cost",
    "initial_dist",
    "pred_dist_h",
    "pred_dist_T",
    "pred_progress_h",
    "pred_progress_T",
    "pred_dist_4",
    "pred_progress_4",
    "pred_dist_10",
    "pred_progress_10",
    "progress_horizon",
)


def loggable_gap_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter a metrics dict down to the JSONL-loggable scalar entries."""
    if not metrics:
        return {}
    out: Dict[str, Any] = {}
    for key in LOGGABLE_GAP_KEYS:
        if key in metrics and metrics[key] is not None:
            out[key] = metrics[key]
    for key in ("gap_selected", "path_cost_selected", "retry_triggered", "retry_accepted"):
        if key in metrics and metrics[key] is not None:
            out[key] = metrics[key]
    return out

