from typing import NamedTuple, Optional, TYPE_CHECKING
from .planner import Planner, PlanningResult
import torch

from pldm.planning.diagnostics import cross_level_metrics
from pldm.planning.level_retry import plan_with_retry

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from pldm.planning.d4rl.enums import ExecRepairConfig


class TwoLvlPlanningResult(NamedTuple):
    level1: PlanningResult
    level2: PlanningResult


class TwoLvlPlanner:
    def __init__(
        self,
        l1_planner: Planner,
        l2_planner: Planner,
        l2_step_skip: int,
        exec_cfg: Optional["ExecRepairConfig"] = None,
        chunk_offset: int = 0,
    ):
        self.l1_planner = l1_planner
        self.l2_planner = l2_planner
        self.l2_step_skip = l2_step_skip
        # Research switches (pldm.planning.d4rl.enums.ExecRepairConfig).
        # Step 0 keeps them inert: with every switch off, plan() below is
        # byte-for-byte identical to the baseline implementation.
        self.exec_cfg = exec_cfg
        self.chunk_offset = int(chunk_offset)
        # Research switches. ``enabled`` is the master switch: with it off the
        # planner behaves exactly like the baseline (including subgoal index 1)
        # no matter what the other research fields say.
        self._research_on = bool(
            exec_cfg is not None and getattr(exec_cfg, "enabled", False)
        )
        self._diag_on = bool(
            self._research_on and getattr(exec_cfg, "log_diagnostics", False)
        )
        retry_threshold = float(
            getattr(exec_cfg, "retry_gap_threshold", 0.0) or 0.0
        )
        self._retry_on = bool(
            self._research_on
            and getattr(exec_cfg, "retry_enabled", False)
            and retry_threshold > 0.0
        )
        self.last_retry_info = None
        self.last_diagnostics = None

    def _subgoal_index(self) -> int:
        """High-level subgoal index. Fixed to 1 unless the master switch is on."""
        if self._research_on:
            return int(getattr(self.exec_cfg, "subgoal_index", 1))
        return 1

    @property
    def task_id_offset(self) -> int:
        """Global task id offset for this chunk (shard offset + chunk offset)."""
        shard_offset = getattr(self.exec_cfg, "task_id_offset", 0) or 0
        return int(shard_offset) + self.chunk_offset

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = True):
        self.l2_planner.reset_targets(targets, repr_input=repr_input)

    def plan(
        self,
        current_state: torch.Tensor,
        plan_size: int,
        curr_proprio_pos: Optional[torch.Tensor] = None,
        curr_proprio_vel: Optional[torch.Tensor] = None,
        curr_locations: Optional[torch.Tensor] = None,
        repr_input: bool = False,
        mock_l1: bool = False,
        diff_loss_idx: Optional[torch.tensor] = None,
    ):
        batch_size = current_state.shape[0]
        self.last_retry_info = None
        self.last_diagnostics = None

        proprio_l1 = None
        if curr_proprio_pos is not None and curr_proprio_vel is not None:
            proprio_l1 = torch.cat([curr_proprio_pos, curr_proprio_vel], dim=-1).cuda()
        elif curr_proprio_pos is not None:
            proprio_l1 = curr_proprio_pos.cuda()
        elif curr_proprio_vel is not None:
            proprio_l1 = curr_proprio_vel.cuda()

        locations_cuda = curr_locations.cuda() if curr_locations is not None else None

        backbone_output = self.l1_planner.model.backbone(
            current_state.cuda(), proprio=proprio_l1, locations=locations_cuda
        )
        l2_result = self.l2_planner.plan(
            current_state=backbone_output,
            plan_size=plan_size,
            repr_input=True,
            # diff_loss_idx=diff_loss_idx.to(enc2.device),
        )

        if mock_l1:
            # ONLY MAKES SENSE FOR WALL DATASET
            actions_l1 = (
                l2_result.actions[:, 0]
                .detach()
                .unsqueeze(1)
                .repeat(1, self.l2_step_skip, 1)
                .to(encs2.device)
            )
            actions_l1 = actions_l1 / self.l2_step_skip * 2
            locations_l1 = torch.zeros(
                (self.l2_step_skip + 1, batch_size, 1, 2),
                device=encs2.device,
            )
        else:
            subgoal_index = self._subgoal_index()
            self.l1_planner.reset_targets(
                l2_result.pred_obs[subgoal_index].detach(), repr_input=True
            )

            l1_plan_kwargs = dict(
                current_state=backbone_output,
                plan_size=self.l2_step_skip,
                repr_input=True,
                curr_proprio_pos=curr_proprio_pos,
                curr_proprio_vel=curr_proprio_vel,
            )
            l1_result = self.l1_planner.plan(**l1_plan_kwargs)

            gap_metrics = None
            if self._diag_on or self._retry_on:
                objective = getattr(self.l1_planner, "objective", None)
                if objective is not None:
                    gap_metrics = cross_level_metrics(
                        l1_result.pred_obs,
                        l2_result.pred_obs,
                        objective,
                        subgoal_index=subgoal_index,
                    )

            if self._retry_on and gap_metrics is not None:
                l1_result = self._maybe_retry(
                    l1_result,
                    l1_plan_kwargs,
                    gap_metrics,
                    l2_result,
                    subgoal_index,
                )

            if self._diag_on and gap_metrics is not None:
                self.last_diagnostics = dict(gap_metrics)
                if self.last_retry_info is not None:
                    self.last_diagnostics[
                        "gap_selected"
                    ] = self.last_retry_info.gap_selected
                    if self.last_retry_info.path_cost_selected is not None:
                        self.last_diagnostics[
                            "path_cost_selected"
                        ] = self.last_retry_info.path_cost_selected

        return TwoLvlPlanningResult(level1=l1_result, level2=l2_result)

    def _maybe_retry(
        self,
        first_result: PlanningResult,
        plan_kwargs: dict,
        first_metrics: dict,
        l2_result: PlanningResult,
        subgoal_index: int,
    ) -> PlanningResult:
        """Step 3: give the low-level planner a conditional second attempt."""
        objective = getattr(self.l1_planner, "objective", None)
        if objective is None:
            return first_result

        def metrics_fn(result: PlanningResult) -> dict:
            return cross_level_metrics(
                result.pred_obs,
                l2_result.pred_obs,
                objective,
                subgoal_index=subgoal_index,
            )

        merged, info = plan_with_retry(
            plan_kwargs=plan_kwargs,
            planner=self.l1_planner,
            first_result=first_result,
            first_metrics=first_metrics,
            metrics_fn=metrics_fn,
            cfg=self.exec_cfg,
        )
        self.last_retry_info = info
        return merged
