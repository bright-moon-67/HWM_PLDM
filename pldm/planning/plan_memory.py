"""Plan identity and persistence bookkeeping (ECCHP).

The execution-aware line so far only reasons about *decisions* ("should the low
level retry?", "should the subgoal be repaired?").  ECCHP additionally needs to
reason about *plans* that persist across several replanning windows:

* a high-level plan receives an identity (``plan_id``) when it is created;
* every replanning decision either keeps that plan or replaces it;
* an execution outcome must be attributable to the plan that produced it,
  otherwise "the plan was bad" and "this particular execution of the plan was
  unlucky" cannot be separated (research plan, stage 1).

This module only keeps that bookkeeping.  It does not touch planner tensors, it
does not consume RNG, and it is only advanced when the caller explicitly asks
for it (``exec_repair.log_plan_events``).  With the switch off, nothing in this
module is instantiated, so the baseline behaviour is unchanged.

Important detail for batched evaluation (``B > 1``): the plan id is a *per
environment* state.  A batch must never share one ``plan_id``; otherwise a
switch in environment 3 would look like a switch in every environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

# --------------------------------------------------------------------------- #
# Switch reasons (shared with the planner and the diagnostics writer).
# --------------------------------------------------------------------------- #
REASON_INITIAL = "initial"
REASON_KEEP = "keep"
REASON_RETRY = "retry_accepted"
REASON_REPAIR_RESElection = "repair_reselection"
REASON_REPAIR_PROJECTION = "repair_projection"
REASON_GOAL_REACHED = "goal_reached"

#: Sources a plan can come from.  ``hwm`` is the untouched high-level planner,
#: ``retry`` only changed the low-level actions, ``repair_*`` replaced the
#: macro plan (and possibly the subgoal).
SOURCE_HWM = "hwm"
SOURCE_RETRY = "retry"
SOURCE_REPAIR_RESElection = "repair_reselection"
SOURCE_REPAIR_PROJECTION = "repair_projection"


@dataclass
class PlanDecision:
    """One per-environment planning decision (``plan_decision`` event).

    Every field is scalar on purpose: the JSONL recorder never receives
    tensors, and a record can be inspected without the planner that produced it.
    """

    task_id: int
    env_index: int
    step: int
    decision_id: int
    plan_id: int
    previous_plan_id: int
    plan_age_replans: int
    candidate_source: str
    switch_reason: str
    switched: bool
    n_candidates: int = 0
    high_level_cost: Optional[float] = None
    cross_mse: Optional[float] = None
    predicted_progress: Optional[float] = None
    # Real position at decision time (metres, XY only).  Filled in by the MPC
    # loop, which is the only component that knows the environment state.
    start_xy: Optional[List[float]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": int(self.task_id),
            "env_index": int(self.env_index),
            "step": int(self.step),
            "decision_id": int(self.decision_id),
            "plan_id": int(self.plan_id),
            "previous_plan_id": int(self.previous_plan_id),
            "plan_age_replans": int(self.plan_age_replans),
            "candidate_source": str(self.candidate_source),
            "switch_reason": str(self.switch_reason),
            "switched": bool(self.switched),
            "n_candidates": int(self.n_candidates),
            "high_level_cost": _scalar(self.high_level_cost),
            "cross_mse": _scalar(self.cross_mse),
            "predicted_progress": _scalar(self.predicted_progress),
            "start_xy": _xy(self.start_xy),
        }


def _scalar(value: Any) -> Optional[float]:
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


def _xy(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [float(value[0]), float(value[1])]
    return None


class PlanMemory:
    """Per-environment plan identity across replanning windows.

    Parameters
    ----------
    n_envs:
        Number of environments in the batch.  The memory calls
        :meth:`reset` immediately and re-allocates if the batch size changes.
    task_id_offset:
        ``exec_repair.task_id_offset + chunk_offset``; used to give every
        decision a globally unique, shard-aware ``task_id``.
    """

    def __init__(self, n_envs: int, *, task_id_offset: int = 0):
        self.n_envs = int(n_envs)
        self.task_id_offset = int(task_id_offset)
        self._next_plan_id = 0
        self.decision_id = -1
        self.plan_ids: torch.Tensor = torch.zeros(0, dtype=torch.int64)
        self.plan_ages: torch.Tensor = torch.zeros(0, dtype=torch.int64)
        self.sources: List[str] = []
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        """Start new plans (all environments, or only the ones in ``mask``).

        A reset environment gets a fresh plan id so that a plan which spans a
        reset can never be mistaken for a continuation.
        """
        if self.plan_ids.numel() != self.n_envs:
            self._allocate()

        if mask is None:
            for i in range(self.n_envs):
                self.plan_ids[i] = self._new_plan_id()
                self.plan_ages[i] = 0
                self.sources[i] = SOURCE_HWM
            return

        mask = mask.detach().cpu().to(dtype=torch.bool).view(-1)
        if mask.numel() != self.n_envs:
            raise ValueError(
                f"reset mask has {mask.numel()} entries for {self.n_envs} envs"
            )
        for i in range(self.n_envs):
            if bool(mask[i]):
                self.plan_ids[i] = self._new_plan_id()
                self.plan_ages[i] = 0
                self.sources[i] = SOURCE_HWM

    def _allocate(self) -> None:
        self.plan_ids = torch.zeros(self.n_envs, dtype=torch.int64)
        self.plan_ages = torch.zeros(self.n_envs, dtype=torch.int64)
        self.sources = [SOURCE_HWM] * self.n_envs

    def _new_plan_id(self) -> int:
        plan_id = self._next_plan_id
        self._next_plan_id += 1
        return plan_id

    def _ensure_batch(self, batch_size: int) -> None:
        """Defensive re-allocation when the batch size changed between calls."""
        if int(batch_size) == self.n_envs:
            return
        self.n_envs = int(batch_size)
        self._allocate()
        self.reset()

    # --------------------------------------------------------------- decision
    def begin_decision(self) -> int:
        """Advance the decision counter (once per replanning window)."""
        self.decision_id += 1
        return self.decision_id

    def update(
        self,
        switched: torch.Tensor,
        sources: Sequence[str],
        reasons: Sequence[str],
        *,
        step: int = 0,
        n_candidates: int = 0,
        high_level_cost: Optional[torch.Tensor] = None,
        cross_mse: Optional[torch.Tensor] = None,
        predicted_progress: Optional[torch.Tensor] = None,
    ) -> List[PlanDecision]:
        """Fold one replanning decision into the plan identities.

        ``switched[i]`` is True when environment ``i`` replaced its high-level
        plan at this decision.  A non-switching decision only ages the existing
        plan; it must not create a new ``plan_id`` (that is the whole point of
        "plan persistence").
        """
        switched = switched.detach().cpu().to(dtype=torch.bool).view(-1)
        batch_size = switched.numel()
        self._ensure_batch(batch_size)

        if len(sources) != batch_size or len(reasons) != batch_size:
            raise ValueError(
                "sources/reasons must have one entry per environment "
                f"(got {len(sources)}/{len(reasons)} for {batch_size})"
            )

        decision_id = self.decision_id
        records: List[PlanDecision] = []
        for i in range(batch_size):
            previous = int(self.plan_ids[i])
            did_switch = bool(switched[i])
            plan_id = self._new_plan_id() if did_switch else previous
            age = 0 if did_switch else int(self.plan_ages[i]) + 1
            self.plan_ids[i] = plan_id
            self.plan_ages[i] = age
            self.sources[i] = str(sources[i])

            records.append(
                PlanDecision(
                    task_id=self.task_id_offset + i,
                    env_index=i,
                    step=int(step),
                    decision_id=int(decision_id),
                    plan_id=int(plan_id),
                    previous_plan_id=int(previous),
                    plan_age_replans=int(age),
                    candidate_source=str(sources[i]),
                    switch_reason=str(reasons[i]),
                    switched=did_switch,
                    n_candidates=int(n_candidates),
                    high_level_cost=_scalar_at(high_level_cost, i),
                    cross_mse=_scalar_at(cross_mse, i),
                    predicted_progress=_scalar_at(predicted_progress, i),
                )
            )
        return records

    # -------------------------------------------------------------- snapshots
    def plan_id_snapshot(self) -> torch.Tensor:
        """Copy of the current per-environment plan ids."""
        return self.plan_ids.clone()

    def state(self) -> Dict[str, Any]:
        """Small, JSON-friendly summary (used inside ``pending_exec``)."""
        return {
            "decision_id": int(self.decision_id),
            "plan_ids": [int(v) for v in self.plan_ids.tolist()],
            "plan_ages": [int(v) for v in self.plan_ages.tolist()],
            "sources": list(self.sources),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"PlanMemory(n_envs={self.n_envs}, decision_id={self.decision_id}, "
            f"plan_ids={self.plan_ids.tolist() if self.plan_ids.numel() else []})"
        )


def _scalar_at(value: Optional[torch.Tensor], index: int) -> Optional[float]:
    """Element ``index`` of an optional ``(B,)`` tensor as a python float."""
    if value is None:
        return None
    if not torch.is_tensor(value):
        return _scalar(value)
    flat = value.detach().cpu().reshape(-1)
    if flat.numel() == 0:
        return None
    if flat.numel() == 1:
        index = 0
    if index >= flat.numel():
        return None
    return float(flat[index].double().item())
