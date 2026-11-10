from typing import Optional

from pldm.planning import objectives_v2
import torch
from pldm.models.hjepa import HJEPA
from pldm_envs.utils.normalizer import Normalizer
from pldm.planning.planners.enums import PlannerType
from pldm.planning.planners.mppi_planner import MPPIPlanner
from pldm.planning.planners.two_lvl_planner import TwoLvlPlanner
from pldm.planning.utils import normalize_actions
from abc import ABC
from pldm.planning.enums import MPCResult, PooledMPCResult
import numpy as np
from pldm.models.utils import flatten_conv_output
from tqdm import tqdm
from pldm.logger import Logger
from pldm.models.utils import flatten_ensemble_conv_output
from pldm.planning.diagnostics import (
    DiagnosticsRecorder,
    ExecutionOutcomeRecord,
    PlanTransitionRecord,
    execution_prediction_index,
    project_l1_encoding_to_l2,
)
from pldm.planning.exec_feedback import ExecutionFeedback
from pldm.planning.env_snapshot import (
    restore_env,
    snapshot_env,
    supports_snapshot,
)
from pldm.planning.level_retry import (
    restore_low_level_state,
    snapshot_low_level_state,
)
from pldm.planning.maze_metrics import graph_distance
from pldm.planning.progress_estimator import RouteHistory
from pldm.planning.warm_start import shift_low_level_warm_start


def _start_xy_from_infos(infos):
    """Real ``(B, 2)`` positions from a list of environment infos, or ``None``."""
    if not infos:
        return None
    try:
        return torch.from_numpy(
            np.stack(
                [
                    np.asarray(info["location"], dtype=np.float32).reshape(-1)[:2]
                    for info in infos
                ]
            )
        )
    except (KeyError, TypeError, ValueError):
        return None


class MPCEvaluator(ABC):
    def __init__(
        self,
        config,
        model: HJEPA,
        prober: torch.nn.Module,
        normalizer: Normalizer,
        quick_debug: bool = False,
        prefix: str = "",
        pixel_mapper=None,
        image_based=True,
        hierarchical=False,
        env_name=None,
        l2_use_latent_mean_std: bool = False,
        output_path: Optional[str] = None,
    ):
        self.config = config
        self.model = model
        self.prober = prober
        self.normalizer = normalizer
        self.quick_debug = quick_debug
        self.prefix = prefix
        self.pixel_mapper = pixel_mapper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_based = image_based
        self.hierarchical = hierarchical
        self.l2_use_latent_mean_std = l2_use_latent_mean_std
        self.output_path = output_path

    def close(self):
        pass

    def _infer_chunk_sizes(self):
        config = self.config

        if config.level1.planner_type == PlannerType.MPPI:
            # n_envs_batch_size = 500000 // (config.num_samples * config.max_plan_length)
            n_envs_batch_size = config.n_envs_batch_size
        else:
            n_envs_batch_size = config.n_envs

        chunk_sizes = [n_envs_batch_size] * (config.n_envs // n_envs_batch_size) + (
            [config.n_envs % n_envs_batch_size]
            if config.n_envs % n_envs_batch_size != 0
            else []
        )

        return chunk_sizes

    def _construct_h_planner(self, n_envs: int, chunk_offset: int = 0, exec_feedback=None):
        l1_planner = self._construct_planner(n_envs=n_envs, l1_to_l2=True)

        l2_planner = self._construct_planner(
            n_envs=n_envs,
            l2=True,
        )

        h_planner = TwoLvlPlanner(
            l1_planner=l1_planner,
            l2_planner=l2_planner,
            l2_step_skip=self.model.config.step_skip,
            # NOTE: hierarchical planners are rebuilt per chunk inside
            # _perform_mpc_in_chunks, so research state must live on the
            # planner (or config), never on the evaluator.
            exec_cfg=getattr(self.config, "exec_repair", None),
            chunk_offset=chunk_offset,
            replan_every=int(getattr(self.config, "replan_every", 4) or 4),
            exec_feedback=exec_feedback,
        )

        return h_planner

    # ------------------------------------------------------------ research cfg
    def _exec_cfg(self):
        return getattr(self.config, "exec_repair", None)

    def _plan_events_on(self) -> bool:
        """Whether ECCHP stage-1 plan events are recorded for this run."""
        cfg = self._exec_cfg()
        return bool(
            cfg is not None
            and getattr(cfg, "enabled", False)
            and (
                getattr(cfg, "log_plan_events", False)
                or getattr(cfg, "collect_candidates", False)
            )
        )

    def _candidate_collection_config(self) -> Optional[dict]:
        """Horizons / budget for stage-2 counterfactual replay, or ``None``."""
        cfg = self._exec_cfg()
        if (
            cfg is None
            or not getattr(cfg, "enabled", False)
            or not getattr(cfg, "collect_candidates", False)
        ):
            return None
        horizons = []
        for part in str(
            getattr(cfg, "candidate_horizons", "40,80") or ""
        ).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                value = int(float(part))
            except ValueError:
                continue
            if value > 0:
                horizons.append(value)
        if not horizons:
            return None
        return {
            "horizons": sorted(set(horizons)),
            "max_decisions": int(
                getattr(cfg, "candidate_max_decisions", 0) or 0
            ),
        }

    def _should_shift_l1_warm_start(self) -> bool:
        """Whether to refresh the low-level MPPI warm start before replanning."""
        cfg = self._exec_cfg()
        return bool(
            cfg is not None
            and getattr(cfg, "enabled", False)
            and getattr(cfg, "shift_l1_warm_start", False)
        )

    def _construct_planner(
        self,
        n_envs: int,
        l2: bool = False,
        l1_to_l2: bool = False,
    ):
        """
        Params:
            n_envs: how many environments to evaluate in parallel
            l2: whether to construct a level 2 planner
            l1_to_l2: whether to construct a level 1 planner for hierarchical planning
        """
        latent_actions = l2 and self.model.level2.config.predictor.z_dim > 0

        if l2:
            config = self.config.level2
            model = self.model.level2
            prober = self.prober_l2
            objective = objectives_v2.ReprTargetMPCObjective2(
                model=model,
                sum_all_diffs=config.sum_all_diffs,
                sum_last_n=config.sum_last_n,
            )
        else:
            config = self.config.level1
            model = self.model.level1
            prober = self.prober
            objective = objectives_v2.ReprTargetMPCObjective(
                model=model,
                proprio_cost=config.proprio_cost,
                sum_all_diffs=config.sum_all_diffs,
                loss_coeff_first=config.loss_coeff_first,
                loss_coeff_last=config.loss_coeff_last,
                pred_encoder=self.model.level2.backbone if l1_to_l2 else None,
            )

        dynamic_bounds = None
        if (
            l2
            and latent_actions
            and hasattr(self.normalizer, "l2_latent_min_bounds")
            and self.normalizer.l2_latent_min_bounds is not None
        ):
            dynamic_bounds = {
                "min_bounds": self.normalizer.l2_latent_min_bounds,
                "max_bounds": self.normalizer.l2_latent_max_bounds,
            }

        action_normalizer = lambda x: normalize_actions(
            x,
            min_norm=config.min_step,
            max_norm=config.max_step,
            clamp_actions=latent_actions or config.clamp_actions,
            env_name=self.config.env_name,
            dynamic_bounds=dynamic_bounds,
        )

        if config.planner_type == PlannerType.MPPI:
            planner = MPPIPlanner(
                config.mppi,
                model=model,
                normalizer=self.normalizer,
                objective=objective,  # [300, 13456]
                prober=prober,
                action_normalizer=action_normalizer,
                n_envs=n_envs,
                l2=l2,
                projected_cost=config.projected_cost,
                cost_entity=config.cost_entity,
                cost_dim_range=config.cost_dim_range,
                use_latent_mean_std=self.l2_use_latent_mean_std,
            )
        else:
            raise NotImplementedError(f"Unknown planner type {config.planner_type}")

        return planner

    def _make_diagnostics(self, chunk_offset: int):
        """Recorder for the current chunk, or None when diagnostics are off."""
        exec_cfg = getattr(self.config, "exec_repair", None)
        if exec_cfg is None:
            return None
        output_dir = getattr(exec_cfg, "output_dir", "") or self.output_path
        recorder = DiagnosticsRecorder(
            exec_cfg=exec_cfg,
            output_dir=output_dir,
            chunk_offset=chunk_offset,
        )
        if not recorder.enabled:
            return None
        if recorder.path is None:
            raise RuntimeError(
                "exec_repair diagnostics are enabled but no output directory is "
                "available; set eval_cfg.h_d4rl_planning.exec_repair.output_dir "
                "or make sure the evaluator has an output_path"
            )
        return recorder

    def _record_exec_error(
        self,
        *,
        diagnostics,
        pending,
        obs_t,
        curr_proprio_pos,
        curr_proprio_vel,
        curr_locations,
        last_infos,
        step: int,
        n_envs: int,
        objective=None,
    ):
        """Compare the stored low-level prediction with the real observation.

        The primary metric lives in the level-1 encoding space (the same space
        as the planner's prediction); XY is recorded as a secondary, prober
        dependent reference.

        When the execution context captured at plan time is available, an
        *execution progress gap* is also reported:
        ``pred_progress_h - real_progress_h``. It isolates "the plan did not move
        as far towards the subgoal as predicted" from "the plan was bad in the
        first place", which is the distinction the retry trigger needs.
        """
        proprio = None
        if curr_proprio_pos is not None and curr_proprio_vel is not None:
            proprio = torch.cat([curr_proprio_pos, curr_proprio_vel], dim=-1).cuda()
        elif curr_proprio_pos is not None:
            proprio = curr_proprio_pos.cuda()
        elif curr_proprio_vel is not None:
            proprio = curr_proprio_vel.cuda()

        with torch.no_grad():
            backbone_output = self.model.level1.backbone(
                obs_t.cuda(),
                proprio=proprio,
                locations=(
                    curr_locations.cuda() if curr_locations is not None else None
                ),
            )
            z_real = backbone_output.obs_component.detach().cpu()
            z_real_l2 = None
            if objective is not None:
                try:
                    z_real_l2 = (
                        project_l1_encoding_to_l2(
                            backbone_output.obs_component.detach(), objective
                        )
                        .detach()
                        .cpu()
                    )
                except Exception:  # pragma: no cover - diagnostics must not crash
                    z_real_l2 = None

        z_pred = pending["pred_obs"]
        if z_pred.shape != z_real.shape:
            raise RuntimeError(
                "execution diagnostic shape mismatch: predicted "
                f"{tuple(z_pred.shape)} vs real {tuple(z_real.shape)}"
            )

        flat_diff = (z_pred - z_real).flatten(start_dim=1)
        metrics = {
            "plan_step": pending["plan_step"],
            "horizon": pending["horizon"],
            "exec_mse_obs": flat_diff.pow(2).mean(dim=1),
            "exec_l2_obs": torch.linalg.vector_norm(flat_diff, dim=1),
        }

        # ------------------------------------------------ execution progress
        subgoal = pending.get("subgoal_l2")
        initial_dist = pending.get("initial_dist")
        if (
            z_real_l2 is not None
            and subgoal is not None
            and initial_dist is not None
            and z_real_l2.shape == subgoal.shape
        ):
            real_dist = (z_real_l2 - subgoal).pow(2).mean(dim=1)
            real_progress = (initial_dist - real_dist) / initial_dist.clamp_min(1e-8)
            metrics["initial_dist"] = initial_dist
            metrics["real_dist_h"] = real_dist
            metrics["real_progress_h"] = real_progress
            pred_progress = pending.get("pred_progress_h")
            if pred_progress is not None:
                metrics["pred_progress_h"] = pred_progress
                metrics["execution_progress_gap"] = pred_progress - real_progress

        pred_xy = pending.get("pred_xy")
        if pred_xy is not None and last_infos is not None:
            real_xy = torch.from_numpy(
                np.stack(
                    [
                        np.asarray(info["location"], dtype=np.float32).reshape(-1)[
                            :2
                        ]
                        for info in last_infos
                    ]
                )
            )
            if pred_xy.shape == real_xy.shape:
                diff_xy = pred_xy - real_xy
                metrics["pred_xy"] = pred_xy
                metrics["real_xy"] = real_xy
                metrics["exec_mse_xy"] = diff_xy.pow(2).mean(dim=-1)
                metrics["exec_l2_xy"] = torch.linalg.vector_norm(diff_xy, dim=-1)

                # Stuck window (review 4, A): the plan predicted motion but the
                # robot stayed put. Uses the real position at plan time, which is
                # why ``pending_exec`` stores it.
                start_xy = pending.get("start_xy")
                if start_xy is not None and start_xy.shape == real_xy.shape:
                    real_move = torch.linalg.vector_norm(real_xy - start_xy, dim=-1)
                    pred_move = torch.linalg.vector_norm(pred_xy - start_xy, dim=-1)
                    real_eps = float(
                        getattr(self._exec_cfg(), "stuck_real_eps", 0.15) or 0.0
                    )
                    pred_eps = float(
                        getattr(self._exec_cfg(), "stuck_pred_eps", 0.30) or 0.0
                    )
                    metrics["real_move"] = real_move
                    metrics["pred_move"] = pred_move
                    metrics["stuck_window"] = (
                        (real_move < real_eps) & (pred_move > pred_eps)
                    ).to(dtype=torch.float32)

        if diagnostics is not None:
            diagnostics.record(
                "exec_error",
                batch_size=n_envs,
                step=step,
                level="hier",
                metrics=metrics,
            )
        return metrics

    # ------------------------------------------------- ECCHP stage-1 events
    def _record_plan_events(
        self,
        *,
        diagnostics,
        decision,
        start_xy,
        step: int,
        n_envs: int,
    ) -> None:
        """Write one ``plan_decision`` and one ``plan_transition`` per env.

        The two events share ``decision_id``/``plan_id`` so a reader can join
        them with the later ``execution_outcome`` lines without parsing the
        whole file into one giant table.
        """
        records = (decision or {}).get("records") or []
        for record in records:
            index = int(getattr(record, "env_index", 0))
            if start_xy is not None and record.start_xy is None:
                if index < start_xy.shape[0]:
                    record.start_xy = (
                        start_xy[index].detach().cpu().reshape(-1)[:2].tolist()
                    )
            diagnostics.record_plan_decision(record, step=step)
            diagnostics.record_plan_transition(
                PlanTransitionRecord(
                    task_id=record.task_id,
                    env_index=index,
                    step=step,
                    decision_id=record.decision_id,
                    plan_id_before=record.previous_plan_id,
                    plan_id_after=record.plan_id,
                    switched=record.switched,
                    reason=record.switch_reason,
                    plan_age_replans=record.plan_age_replans,
                ),
                step=step,
            )

    def _record_plan_outcome(
        self,
        *,
        diagnostics,
        pending,
        exec_metrics,
        route_metrics,
        start_xy,
        step: int,
        n_envs: int,
    ) -> None:
        """Write one ``execution_outcome`` line per environment."""
        decision = pending.get("plan_decision") if pending else None
        records = (decision or {}).get("records") or []
        if not records:
            return

        def _at(value, index):
            if value is None:
                return None
            if torch.is_tensor(value):
                flat = value.detach().cpu().reshape(-1)
                if flat.numel() <= index:
                    return None
                return float(flat[index].double().item())
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _point(value, index):
            if value is None or not torch.is_tensor(value):
                return None
            if value.shape[0] <= index:
                return None
            return value[index].detach().cpu().reshape(-1)[:2].tolist()

        for record in records:
            index = int(getattr(record, "env_index", 0))
            if index >= n_envs:
                continue
            stuck = _at((exec_metrics or {}).get("stuck_window"), index)
            diagnostics.record_execution_outcome(
                ExecutionOutcomeRecord(
                    task_id=record.task_id,
                    env_index=index,
                    step=step,
                    decision_id=record.decision_id,
                    plan_id=record.plan_id,
                    start_xy=_point(pending.get("start_xy"), index),
                    end_xy=_point(start_xy, index),
                    predicted_xy=_point(pending.get("pred_xy"), index),
                    displacement=_at((exec_metrics or {}).get("real_move"), index),
                    predicted_displacement=_at(
                        (exec_metrics or {}).get("pred_move"), index
                    ),
                    subgoal_dist_before=_at(pending.get("initial_dist"), index),
                    subgoal_dist_after=_at(
                        (exec_metrics or {}).get("real_dist_h"), index
                    ),
                    real_progress_h=_at(
                        (exec_metrics or {}).get("real_progress_h"), index
                    ),
                    pred_progress_h=_at(
                        (exec_metrics or {}).get("pred_progress_h"), index
                    ),
                    execution_progress_gap=_at(
                        (exec_metrics or {}).get("execution_progress_gap"), index
                    ),
                    exec_mse_obs=_at(
                        (exec_metrics or {}).get("exec_mse_obs"), index
                    ),
                    exec_mse_xy=_at((exec_metrics or {}).get("exec_mse_xy"), index),
                    route_path_length=_at(
                        (route_metrics or {}).get("route_path_length"), index
                    ),
                    route_efficiency=_at(
                        (route_metrics or {}).get("route_efficiency"), index
                    ),
                    route_revisit_count=_at(
                        (route_metrics or {}).get("route_revisit_count"), index
                    ),
                    stuck=None if stuck is None else bool(stuck > 0.5),
                ),
                step=step,
            )

    # ------------------------------------- ECCHP stage-2 counterfactual replay
    @staticmethod
    def _env_xy(env):
        """Real XY of ``env`` (``info['location']``), or ``None``."""
        try:
            info = env.get_info()
            if "location" in info:
                return np.asarray(info["location"], dtype=np.float64).reshape(-1)[:2]
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            return np.asarray(
                env.get_pos(normalized=False), dtype=np.float64
            ).reshape(-1)[:2]
        except Exception:  # pragma: no cover - defensive
            return None

    def _stacked_obs_single(self, env) -> torch.Tensor:
        """Observation tensor for a single environment, stacked as in MPC."""
        obs = torch.from_numpy(np.stack([env.get_obs()])).float()
        if self.image_based:
            obs = torch.cat([obs] * self.config.stack_states, dim=1)
        return obs

    def _rollout_candidate_actions(
        self,
        env,
        planner,
        first_actions: torch.Tensor,
        horizon: int,
        *,
        max_plan_horizon_l2: int,
        step_skip: int,
    ) -> int:
        """Execute a candidate chunk, then continue with the normal controller.

        The first ``first_actions`` come from the candidate L1 plan. After that,
        **the same HWM controller** is used for every candidate, which is what
        makes the comparison a counterfactual on the *high-level plan* rather
        than on the low-level controller.

        ``first_actions`` are ``PlanningResult.actions`` (already un-normalised
        to the environment action space). They are deliberately *not* taken from
        ``ctrl.U``, which lives in the MPPI search space and would silently
        corrupt the replay.
        """
        executed = 0
        n_first = int(min(int(first_actions.shape[0]), int(horizon)))
        for t in range(n_first):
            action = first_actions[t].detach().cpu().numpy().astype(np.float32)
            env.step(action)
            executed += 1

        replan_every = int(self.config.replan_every)
        while executed < horizon:
            obs = self._stacked_obs_single(env)
            proprio_pos = None
            if self.model.level1.using_proprio_pos:
                proprio_pos = torch.from_numpy(
                    np.stack([env.get_proprio_pos(normalized=True)])
                ).float()
            proprio_vel = None
            if self.model.level1.using_proprio_vel:
                proprio_vel = torch.from_numpy(
                    np.stack([env.get_proprio_vel(normalized=True)])
                ).float()
            locations = None
            if self.model.level1.using_location:
                locations = torch.from_numpy(
                    np.stack([env.get_pos(normalized=True)])
                ).float()

            plan_size = (
                max_plan_horizon_l2 - executed + step_skip - 1
            ) // step_skip
            plan_size = max(plan_size, self.config.level2.min_plan_length)
            result = planner.plan(
                obs,
                plan_size=plan_size,
                repr_input=False,
                curr_proprio_pos=proprio_pos,
                curr_proprio_vel=proprio_vel,
                curr_locations=locations,
                decision_step=executed,
            )
            actions = result.level1.actions if hasattr(result, "level1") else result.actions
            n_exec = min(replan_every, horizon - executed)
            if actions.shape[1] < n_exec:
                n_exec = int(actions.shape[1])
            if n_exec <= 0:  # pragma: no cover - defensive
                break
            for t in range(n_exec):
                action = actions[0, t].detach().cpu().numpy().astype(np.float32)
                env.step(action)
                executed += 1
        return executed

    def _collect_counterfactual_returns(
        self,
        *,
        envs,
        planner,
        evidence,
        decision,
        step: int,
        diagnostics,
        collection_cfg,
        decisions_collected: int,
        repair_info=None,
    ) -> int:
        """Replay every candidate from the same real state and log returns.

        All simulator state, planner warm starts and RNG state are restored
        afterwards, so the collected data cannot change the factual trajectory
        of the run (see the stage-0 isolation tests).
        """
        if collection_cfg is None or not evidence:
            return 0
        max_decisions = int(collection_cfg.get("max_decisions", 0) or 0)
        if max_decisions and decisions_collected >= max_decisions:
            return 0

        unavailable = [i for i, env in enumerate(envs) if not supports_snapshot(env)]
        if unavailable:
            if diagnostics is not None:
                diagnostics.record(
                    "candidate_replay_unavailable",
                    batch_size=len(envs),
                    step=step,
                    level="hier",
                    metrics={
                        "unsupported_env_indices": unavailable,
                        "reason": "environment does not expose full state snapshots",
                    },
                )
            return 0

        records = (decision or {}).get("records") or []
        snapshots = [snapshot_env(env) for env in envs]
        planner_snapshot = {
            "l1": snapshot_low_level_state(planner.l1_planner),
            "l2": snapshot_low_level_state(planner.l2_planner),
        }
        rng_devices = (
            [torch.cuda.current_device()] if torch.cuda.is_available() else []
        )
        step_skip = int(self.model.config.step_skip)
        max_plan_horizon_l2 = int(self.config.level2.max_plan_length) * step_skip

        try:
            with torch.random.fork_rng(devices=rng_devices, enabled=True):
                triggered = getattr(repair_info, "triggered", None)
                if torch.is_tensor(triggered):
                    triggered = triggered.detach().cpu().reshape(-1)
                for env_index, env in enumerate(envs):
                    if triggered is not None and (
                        env_index >= triggered.numel()
                        or not bool(triggered[env_index])
                    ):
                        # Only environments whose repair branch fired have a
                        # comparable candidate set; the others only contain
                        # their own plan.
                        continue
                    record = (
                        records[env_index] if env_index < len(records) else None
                    )
                    start_xy = self._env_xy(env)
                    d_start = graph_distance(env, start_xy)
                    for candidate in evidence:
                        actions = candidate.get("actions")
                        if actions is None:
                            continue
                        per_env = (
                            actions[env_index]
                            if actions.dim() == 3
                            else actions
                        )
                        for horizon in collection_cfg["horizons"]:
                            restore_env(env, snapshots[env_index])
                            executed = self._rollout_candidate_actions(
                                env,
                                planner,
                                per_env,
                                int(horizon),
                                max_plan_horizon_l2=max_plan_horizon_l2,
                                step_skip=step_skip,
                            )
                            end_xy = self._env_xy(env)
                            d_end = graph_distance(env, end_xy)
                            real_return = (
                                float(d_start - d_end)
                                if d_start is not None and d_end is not None
                                else None
                            )
                            if diagnostics is not None:
                                diagnostics.record(
                                    "candidate_outcome",
                                    task_ids=[
                                        int(record.task_id)
                                        if record is not None
                                        else int(env_index)
                                    ],
                                    step=step,
                                    level="hier",
                                    metrics={
                                        "env_index": int(env_index),
                                        "task_id": (
                                            int(record.task_id)
                                            if record is not None
                                            else int(env_index)
                                        ),
                                        "decision_id": (
                                            int(record.decision_id)
                                            if record is not None
                                            else -1
                                        ),
                                        "plan_id": (
                                            int(record.plan_id)
                                            if record is not None
                                            else -1
                                        ),
                                        "candidate_id": int(
                                            candidate.get("candidate_id", -1)
                                        ),
                                        "candidate_source": str(
                                            candidate.get("source", "unknown")
                                        ),
                                        "selected": bool(
                                            torch.is_tensor(candidate.get("selected"))
                                            and bool(
                                                candidate["selected"][env_index].item()
                                            )
                                        ),
                                        "model_score": self._candidate_scalar(
                                            candidate.get("high_level_cost"),
                                            env_index,
                                        ),
                                        "cross_mse": self._candidate_scalar(
                                            candidate.get("cross_mse"), env_index
                                        ),
                                        "predicted_progress": self._candidate_scalar(
                                            candidate.get("pred_progress_h"),
                                            env_index,
                                        ),
                                        "horizon": int(horizon),
                                        "executed_steps": int(executed),
                                        "start_xy": None
                                        if start_xy is None
                                        else start_xy.tolist(),
                                        "end_xy": None
                                        if end_xy is None
                                        else end_xy.tolist(),
                                        "graph_distance_before": d_start,
                                        "graph_distance_after": d_end,
                                        "real_return": real_return,
                                        "replay_mode": "candidate_chunk+follow_up_hwm",
                                    },
                                )
        except Exception as exc:
            # Collection must never take down an evaluation run.
            if diagnostics is not None:
                diagnostics.record(
                    "candidate_replay_error",
                    batch_size=len(envs),
                    step=step,
                    level="hier",
                    metrics={"error": repr(exc)},
                )
            return 0
        finally:
            for env, snapshot in zip(envs, snapshots):
                try:
                    restore_env(env, snapshot)
                except Exception:  # pragma: no cover - defensive
                    pass
            restore_low_level_state(planner.l1_planner, planner_snapshot["l1"])
            restore_low_level_state(planner.l2_planner, planner_snapshot["l2"])
        return 1

    @staticmethod
    def _candidate_scalar(value, index: int):
        if value is None:
            return None
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1)
            if flat.numel() == 0:
                return None
            if flat.numel() == 1:
                return float(flat[0].double().item())
            if index < flat.numel():
                return float(flat[index].double().item())
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _perform_mpc_in_chunks(self):
        """
        Divide it up in chunks in order to prevent OOM
        """
        chunk_sizes = self._infer_chunk_sizes()

        mpc_data = PooledMPCResult()
        chunk_offset = 0

        for chunk_size in chunk_sizes:
            envs = self.envs[chunk_offset : chunk_offset + chunk_size]
            planner = self._construct_planner(n_envs=chunk_size)

            if self.hierarchical:
                exec_feedback = ExecutionFeedback.from_config(
                    self._exec_cfg(), n_envs=chunk_size
                )
                h_planner = self._construct_h_planner(
                    n_envs=chunk_size,
                    chunk_offset=chunk_offset,
                    exec_feedback=exec_feedback,
                )
                diagnostics = self._make_diagnostics(chunk_offset)
                try:
                    mpc_result = self._perform_h_mpc(
                        h_planner=h_planner,
                        planner=planner,
                        envs=envs,
                        diagnostics=diagnostics,
                    )
                finally:
                    if diagnostics is not None:
                        diagnostics.close()
            else:
                mpc_result = self._perform_mpc(
                    planner=planner,
                    envs=envs,
                )

            mpc_data.observations.append(mpc_result.observations)
            mpc_data.locations.append(mpc_result.locations)
            mpc_data.action_history.append(mpc_result.action_history)
            mpc_data.reward_history.append(mpc_result.reward_history)
            mpc_data.pred_locations.append(mpc_result.pred_locations)
            mpc_data.final_preds_dist.append(mpc_result.final_preds_dist)
            mpc_data.targets.append(mpc_result.targets)
            mpc_data.loss_history.append(mpc_result.loss_history)
            mpc_data.qpos_history.append(mpc_result.qpos_history)
            mpc_data.proprio_history.append(mpc_result.proprio_history)
            mpc_data.ensemble_var_history.append(mpc_result.ensemble_var_history)
            mpc_data.ensemble_obs_var_history.append(
                mpc_result.ensemble_obs_var_history
            )
            mpc_data.ensemble_proprio_var_history.append(
                mpc_result.ensemble_proprio_var_history
            )
            mpc_data.success_history.append(mpc_result.success_history)
            mpc_data.visual_observations.append(mpc_result.visual_observations)
            mpc_data.visual_targets.append(mpc_result.visual_targets)

            # for hierarchy
            if self.hierarchical:
                mpc_data.pred_locations_l2.append(mpc_result.pred_locations_l2)
                mpc_data.loss_history_l2.append(mpc_result.loss_history_l2)

            chunk_offset += chunk_size

        mpc_data.concatenate_chunks()

        return mpc_data

    def _perform_h_mpc(self, h_planner, planner, envs, diagnostics=None):
        """
        Hierarchical planning.
        Two stages:
        1. Perform MPC with the hierarchical planner
        2. Perform MPC with the level 1 planner for last part of the trip.
        Rationale: Hierarchical MPC might have some issue with fine-grained control.
            Could be problematic for the final part of the job.
        """
        # Stage 1
        mpc_result_1 = self._perform_mpc(
            planner=h_planner,
            envs=envs,
            bilevel_planning=True,
            diagnostics=diagnostics,
        )

        # Stage 2: Use flat L1 planning for the final steps near the goal
        if self.config.final_trans_steps:
            mpc_result_2 = self._perform_mpc(
                planner=planner,
                envs=envs,
                max_steps_override=self.config.final_trans_steps,
            )
            # combine results
            mpc_result = MPCResult(
                observations=mpc_result_1.observations
                + mpc_result_2.observations[1:],  #
                locations=mpc_result_1.locations + mpc_result_2.locations[1:],  #
                action_history=mpc_result_1.action_history
                + mpc_result_2.action_history,  #
                reward_history=mpc_result_1.reward_history
                + mpc_result_2.reward_history,  #
                pred_locations=mpc_result_1.pred_locations
                + mpc_result_2.pred_locations,  #
                final_preds_dist=mpc_result_1.final_preds_dist,
                targets=mpc_result_1.targets,
                loss_history=mpc_result_1.loss_history,
                ensemble_var_history=mpc_result_1.ensemble_var_history,
                ensemble_obs_var_history=mpc_result_1.ensemble_obs_var_history,
                ensemble_proprio_var_history=mpc_result_1.ensemble_proprio_var_history,
                qpos_history=mpc_result_1.qpos_history + mpc_result_2.qpos_history[1:],
                proprio_history=mpc_result_1.proprio_history
                + mpc_result_2.proprio_history[1:],
                # for hierarchy
                pred_locations_l2=mpc_result_1.pred_locations_l2,
                loss_history_l2=mpc_result_1.loss_history_l2,
                success_history=mpc_result_1.success_history
                + mpc_result_2.success_history[1:],
                visual_observations=mpc_result_1.visual_observations
                + mpc_result_2.visual_observations[1:],
                visual_targets=mpc_result_1.visual_targets,
            )
        else:
            mpc_result = mpc_result_1

        return mpc_result

    def _encode_targets(self, envs, bilevel_planning=False, squeeze_target=True):
        targets_obs = torch.stack([e.get_target_obs() for e in envs]).to(self.device)

        # encode target obs
        if self.model.level1.backbone.using_proprio:
            proprio_states = torch.stack([e.get_target_proprio() for e in envs]).to(
                self.device
            )
        else:
            proprio_states = None

        if self.model.level1.backbone.using_location:
            locations = torch.stack([e.get_goal_xy(normalized=True) for e in envs]).to(
                self.device
            )
        else:
            locations = None

        l1_output = self.model.level1.backbone(
            targets_obs, proprio=proprio_states, locations=locations
        )

        target_obs = l1_output.obs_component.detach()
        target_proprio = l1_output.proprio_component
        target_locations = l1_output.location_component
        target_raw_locations = l1_output.raw_locations

        if target_proprio is not None:
            target_proprio = target_proprio.detach()

        if not bilevel_planning:
            if self.config.level1.cost_entity == "obs_component":
                if len(target_obs.shape) == 3:
                    target_obs = target_obs.unsqueeze(0)
                target_obs_flat = flatten_conv_output(target_obs)
                if target_obs_flat.shape[0] == 1 and squeeze_target:
                    target_obs_flat = target_obs_flat.squeeze(0)
                return target_obs_flat
            elif self.config.level1.cost_entity == "proprio_component":
                return target_proprio
            elif self.config.level1.cost_entity == "location_component":
                return target_locations
            elif self.config.level1.cost_entity == "raw_locations":
                return target_raw_locations
            else:
                raise NotImplementedError(
                    f"Unknown cost entity {self.config.level1.cost_entity}"
                )

        l2_output = self.model.level2.backbone(target_obs, proprio=target_proprio)

        target_obs = l2_output.obs_component.detach()
        target_proprio = l2_output.proprio_component
        if target_proprio is not None:
            target_proprio = target_proprio.detach()

        if self.config.level2.cost_entity == "obs_component":
            if len(target_obs.shape) == 3:
                target_obs = target_obs.unsqueeze(0)
            target_obs_flat = flatten_conv_output(target_obs)
            if target_obs_flat.shape[0] == 1 and squeeze_target:
                target_obs_flat = target_obs_flat.squeeze(0)
            return target_obs_flat
        elif self.config.level2.cost_entity == "proprio_component":
            return target_proprio
        else:
            raise NotImplementedError(
                f"Unknown cost entity {self.config.level2.cost_entity}"
            )

    def _perform_mpc(
        self,
        planner,
        envs,
        bilevel_planning: bool = False,
        max_steps_override: int = None,
        diagnostics=None,
    ):
        """
        Parameters:
            starts: (bs, 4)
            targets: (bs, 4)
            max_steps_override: if provided, use this as the max number of steps
                instead of computing from config. Used for Stage 2 of hierarchical
                planning where we only want to run for final_trans_steps.
        Outputs:
            observations: list of a_T (bs, 3, 64, 64) or (bs, 2)
            locations: list of a_T (bs, 2)
            action_history: list of a_T (bs, p_T, 2)
            reward_history: list of a_T (bs,)
            pred_locations: list of a_T (p_T, bs, 1, 2)
            targets: (bs, 4)
            loss_history: list of a_T (n_iters,)
        """
        orig_training_state = self.model.training

        targets = [e.get_target() for e in envs]
        targets = torch.from_numpy(np.stack(targets))

        if not self.image_based and envs[0]._target_visual is not None:
            visual_targets = [e.get_target_visual() for e in envs]
            visual_targets = torch.from_numpy(np.stack(visual_targets))
        else:
            visual_targets = None

        targets_t = self._encode_targets(envs, bilevel_planning=bilevel_planning, squeeze_target=False)
        planner.reset_targets(targets_t, repr_input=True)

        observation_history = [torch.stack([e.get_obs() for e in envs])]

        obs_t = observation_history[0]
        if self.image_based:
            obs_t = torch.cat([obs_t] * self.config.stack_states, dim=1)  # VERIFY

        action_history = []
        reward_history = []
        location_history = []
        qpos_history = []
        proprio_history = []
        success_history = []

        visual_observations = []
        pred_locations_history = []
        loss_history = []
        final_preds_dist_history = []
        ensemble_var_history = []
        ensemble_obs_var_history = []
        ensemble_proprio_var_history = []

        # for hierarchy
        pred_locations_l2_history = []
        loss_history_l2 = []

        # Step 2 diagnostics: prediction awaiting ``replan_every`` real steps.
        pending_exec = None
        last_infos = None
        # Step 6: online execution feedback owned by the hierarchical planner.
        exec_feedback = getattr(planner, "exec_feedback", None)
        # ECCHP stage 1: real-XY route indicators and plan-event recording.
        plan_events_on = bool(bilevel_planning and self._plan_events_on())
        exec_cfg = self._exec_cfg()
        route_history = (
            RouteHistory(
                len(envs),
                window=int(getattr(exec_cfg, "route_history_window", 8) or 8),
                revisit_radius=float(
                    getattr(exec_cfg, "revisit_radius", 0.25) or 0.25
                ),
            )
            if plan_events_on
            else None
        )
        # ECCHP stage 2: counterfactual candidate replay (opt-in, expensive).
        collection_cfg = (
            self._candidate_collection_config() if bilevel_planning else None
        )
        candidates_collected = 0

        init_infos = [e.get_info() for e in envs]
        if "location" in init_infos[0]:
            location_history.append(np.array([info["location"] for info in init_infos]))

        if "qpos" in init_infos[0]:
            qpos_history.append(np.array([info["qpos"] for info in init_infos]))

        if "proprio" in init_infos[0]:
            proprio_history.append(np.array([info["proprio"] for info in init_infos]))

        if "visual_obs" in init_infos[0]:
            visual_observations.append(
                np.array([info["visual_obs"] for info in init_infos])
            )

        if max_steps_override is not None:
            # Used for Stage 2 of hierarchical planning (final_trans_steps)
            max_steps = max_steps_override
        elif bilevel_planning:
            step_skip = self.model.config.step_skip
            max_plan_horizon_l2 = self.config.level2.max_plan_length * step_skip
            # For hierarchical planning, reserve final_trans_steps for Stage 2 (flat L1)
            # so that total steps = Stage 1 + Stage 2 = n_steps
            final_trans_steps = getattr(self.config, "final_trans_steps", 0)
            n_steps_stage1 = self.config.n_steps - final_trans_steps
            max_steps = min(n_steps_stage1, max_plan_horizon_l2)
        else:
            max_steps = self.config.n_steps

        for i in tqdm(range(max_steps), desc="Planning steps"):
            if i % self.config.replan_every == 0:
                # ECCHP stage 1: refresh the route indicators with the real
                # position *before* the new decision, so the window recorded
                # below covers exactly the steps that were executed since the
                # previous decision.
                route_metrics = None
                current_xy = None
                plan_decision = None
                if route_history is not None:
                    current_xy = _start_xy_from_infos(
                        last_infos if last_infos is not None else init_infos
                    )
                    route_metrics = route_history.update(current_xy)

                if self.model.level1.using_proprio_pos:
                    curr_proprio_pos = [
                        e.get_proprio_pos(normalized=True) for e in envs
                    ]
                    curr_proprio_pos = torch.from_numpy(
                        np.stack(curr_proprio_pos)
                    ).float()
                else:
                    curr_proprio_pos = None

                if self.model.level1.using_proprio_vel:
                    curr_proprio_vel = [
                        e.get_proprio_vel(normalized=True) for e in envs
                    ]
                    curr_proprio_vel = torch.from_numpy(
                        np.stack(curr_proprio_vel)
                    ).float()
                else:
                    curr_proprio_vel = None

                if self.model.level1.using_location:
                    curr_locations = [e.get_pos(normalized=True) for e in envs]
                    curr_locations = torch.from_numpy(np.stack(curr_locations)).float()
                else:
                    curr_locations = None

                if pending_exec is not None and (
                    diagnostics is not None or exec_feedback is not None
                ):
                    exec_metrics = self._record_exec_error(
                        diagnostics=diagnostics,
                        pending=pending_exec,
                        obs_t=obs_t,
                        curr_proprio_pos=curr_proprio_pos,
                        curr_proprio_vel=curr_proprio_vel,
                        curr_locations=curr_locations,
                        last_infos=last_infos,
                        step=i,
                        n_envs=len(envs),
                        objective=getattr(
                            getattr(planner, "l1_planner", planner),
                            "objective",
                            None,
                        ),
                    )
                    # Step 6: feed the realised error back into the planner.
                    # Only windows with a meaningful relative progress ratio are
                    # used (review 3, step 1-2), so a tiny ``initial_dist``
                    # cannot inject an extreme error estimate into the EMA.
                    if exec_feedback is not None and exec_metrics is not None:
                        errors = exec_metrics.get("exec_mse_obs")
                        if errors is not None:
                            min_dist = float(
                                getattr(
                                    self._exec_cfg(),
                                    "feedback_min_initial_dist",
                                    0.0,
                                )
                                or 0.0
                            )
                            valid = torch.isfinite(errors)
                            initial_dist = exec_metrics.get("initial_dist")
                            if initial_dist is not None and min_dist > 0.0:
                                valid = valid & (initial_dist >= min_dist)
                            exec_feedback.update(errors, valid=valid)
                        stuck = exec_metrics.get("stuck_window")
                        if stuck is not None:
                            exec_feedback.update_stuck(stuck > 0.5)
                    # ECCHP stage 1: attribute the window to the plan that
                    # produced it.
                    if diagnostics is not None and plan_events_on:
                        self._record_plan_outcome(
                            diagnostics=diagnostics,
                            pending=pending_exec,
                            exec_metrics=exec_metrics,
                            route_metrics=route_metrics,
                            start_xy=current_xy,
                            step=i,
                            n_envs=len(envs),
                        )
                    pending_exec = None

                if bilevel_planning:
                    plan_size = (max_plan_horizon_l2 - i + step_skip - 1) // step_skip
                    plan_size = max(plan_size, self.config.level2.min_plan_length)
                else:
                    plan_size = min(
                        self.config.n_steps - i, self.config.level1.max_plan_length
                    )

                # P0: the low-level MPPI warm start is only shifted by
                # ``MPPIPlanner`` when the plan length shrinks, which never
                # happens with the constant L1 plan length used here. Advance it
                # manually so index 0 again corresponds to the next executed
                # primitive instead of a stale one.
                if (
                    bilevel_planning
                    and i > 0
                    and self._should_shift_l1_warm_start()
                ):
                    shift_low_level_warm_start(
                        getattr(planner, "l1_planner", None),
                        int(self.config.replan_every),
                    )

                plan_kwargs = dict(
                    curr_proprio_pos=curr_proprio_pos,
                    curr_proprio_vel=curr_proprio_vel,
                    curr_locations=curr_locations,
                    plan_size=plan_size,
                    repr_input=False,
                )
                if isinstance(planner, TwoLvlPlanner):
                    # ECCHP stage 1: the planner does not know the global step,
                    # but plan-level records need it for the replay index.
                    plan_kwargs["decision_step"] = i
                planning_result = planner.plan(obs_t, **plan_kwargs)

                if bilevel_planning:
                    planning_result_l2 = planning_result.level2
                    planning_result = planning_result.level1
                    # ECCHP stage 1: the decision record produced by
                    # ``TwoLvlPlanner`` (None unless ``log_plan_events`` is on).
                    plan_decision = getattr(planner, "last_plan_decision", None)
                    repair_info_obj = getattr(planner, "last_repair_info", None)

                    # Step 3: log the retry decision (no-op unless enabled).
                    if diagnostics is not None:
                        retry_info = getattr(planner, "last_retry_info", None)
                        if retry_info is not None:
                            diagnostics.record(
                                "retry",
                                batch_size=len(envs),
                                step=i,
                                level="hier",
                                metrics=retry_info.as_dict(),
                            )
                            planner.last_retry_info = None

                    # Step 4/5: log the high-level re-search decision.
                    if diagnostics is not None:
                        repair_info = getattr(planner, "last_repair_info", None)
                        if repair_info is not None:
                            diagnostics.record(
                                "repair",
                                batch_size=len(envs),
                                step=i,
                                level="hier",
                                metrics=repair_info.as_dict(),
                            )
                            planner.last_repair_info = None
                        # ECCHP stage 0: a failed candidate search is recorded
                        # instead of disappearing into the fallback path.
                        repair_error = getattr(planner, "last_repair_error", None)
                        if repair_error:
                            diagnostics.record(
                                "repair_error",
                                batch_size=len(envs),
                                step=i,
                                level="hier",
                                metrics={"error": str(repair_error)},
                            )
                            planner.last_repair_error = None

                    # Step 1: cross-level gap of the first L1 plan.
                    if diagnostics is not None:
                        gap_metrics = getattr(planner, "last_diagnostics", None)
                        if gap_metrics is not None:
                            diagnostics.record(
                                "plan_gap",
                                batch_size=len(envs),
                                step=i,
                                level="hier",
                                metrics=gap_metrics,
                            )
                            planner.last_diagnostics = None

                    # ECCHP stage 1: plan-level decision and transition events.
                    if diagnostics is not None and plan_decision is not None:
                        self._record_plan_events(
                            diagnostics=diagnostics,
                            decision=plan_decision,
                            start_xy=current_xy,
                            step=i,
                            n_envs=len(envs),
                        )
                        planner.last_plan_decision = None

                    # ECCHP stage 2: replay every candidate from the same real
                    # state and log its realised return. All simulator and
                    # planner state is restored afterwards, so enabling this
                    # cannot change the executed trajectory.
                    if collection_cfg is not None:
                        evidence = getattr(planner, "last_candidate_evidence", None)
                        if evidence:
                            planner.last_candidate_evidence = None
                            candidates_collected += (
                                self._collect_counterfactual_returns(
                                    envs=envs,
                                    planner=planner,
                                    evidence=evidence,
                                    decision=plan_decision,
                                    step=i,
                                    diagnostics=diagnostics,
                                    collection_cfg=collection_cfg,
                                    decisions_collected=candidates_collected,
                                    repair_info=repair_info_obj,
                                )
                            )

                    # Step 2: store the prediction that matches replan_every
                    # executed steps; it is compared at the next replan.
                    if diagnostics is not None or exec_feedback is not None:
                        pred_obs = getattr(planning_result, "pred_obs", None)
                        if pred_obs is not None:
                            pred_index = execution_prediction_index(
                                self.config.replan_every, pred_obs.shape[0]
                            )
                            if pred_index is not None:
                                pred_xy = None
                                locations = getattr(
                                    planning_result, "locations", None
                                )
                                if (
                                    locations is not None
                                    and locations.shape[0] > pred_index
                                ):
                                    loc = locations[pred_index].detach().cpu()
                                    pred_xy = loc.reshape(loc.shape[0], -1)[:, :2]
                                # Execution context captured *after* the retry
                                # decision, so the progress metrics describe the
                                # plan that is actually executed.
                                exec_context = (
                                    getattr(planner, "last_exec_context", None) or {}
                                )

                                def _cpu(value):
                                    if torch.is_tensor(value):
                                        return value.detach().cpu()
                                    return value

                                pending_exec = {
                                    "pred_obs": pred_obs[pred_index].detach().cpu(),
                                    "pred_xy": pred_xy,
                                    "plan_step": i,
                                    "horizon": pred_index,
                                    # Real position at plan time (last_infos is the
                                    # state the current observation belongs to), so
                                    # the realised/predicted displacements of this
                                    # window use the same origin.
                                    "start_xy": _start_xy_from_infos(last_infos),
                                    "subgoal_l2": _cpu(
                                        exec_context.get("subgoal_l2")
                                    ),
                                    "initial_dist": _cpu(
                                        exec_context.get("initial_dist")
                                    ),
                                    "pred_progress_h": _cpu(
                                        exec_context.get("pred_progress_h")
                                    ),
                                    # ECCHP stage 1: keep the plan identity so
                                    # the next window can be attributed to it.
                                    "plan_decision": plan_decision,
                                }

            if bilevel_planning:
                pred = self._get_relevant_pred(
                    planning_result_l2,
                    self.config.level2.cost_entity,
                )
            else:
                pred = self._get_relevant_pred(
                    planning_result,
                    self.config.level1.cost_entity,
                )
            pred_dist = torch.norm(pred - targets_t.unsqueeze(0), dim=2).cpu()
            final_preds_dist_history.append(pred_dist)

            # calculate ensemble variance
            ensemble_predictions = planning_result.ensemble_predictions
            if ensemble_predictions.shape[1] > 1:
                ensemble_var = self._calculate_ensemble_var(ensemble_predictions.cpu())
                ensemble_var_history.append(ensemble_var)

            ensemble_obs_component = planning_result.ensemble_obs_component
            if ensemble_obs_component.shape[1] > 1:
                ensemble_obs_var = self._calculate_ensemble_var(
                    ensemble_obs_component.cpu()
                )
                ensemble_obs_var_history.append(ensemble_obs_var)

            ensemble_proprio_component = planning_result.ensemble_proprio_component
            if (
                ensemble_proprio_component is not None
                and ensemble_proprio_component.shape[1] > 1
            ):
                ensemble_proprio_var = self._calculate_ensemble_var(
                    ensemble_proprio_component.cpu()
                )
                ensemble_proprio_var_history.append(ensemble_proprio_var)

            planned_actions = (
                planning_result.actions[:, i % self.config.replan_every :]
                .detach()
                .cpu()
            )

            if self.config.random_actions:
                results = [
                    envs[j].step(envs[0].action_space.sample())
                    for j in range(len(envs))
                ]
            else:
                results = [
                    envs[j].step(
                        planned_actions[j, 0].detach().cpu().contiguous().numpy()
                    )
                    for j in range(len(envs))
                ]

            assert len(results[0]) == 5
            current_obs = torch.from_numpy(np.stack([r[0] for r in results])).float()
            rewards_t = torch.from_numpy(np.stack([r[1] for r in results])).float()
            infos = [r[4] for r in results]
            last_infos = infos

            action_history.append(planned_actions.detach().cpu())
            observation_history.append(current_obs)
            reward_history.append(rewards_t)

            if "location" in infos[0]:
                location_history.append(np.array([info["location"] for info in infos]))

            if "qpos" in infos[0]:
                qpos_history.append(np.array([info["qpos"] for info in infos]))

            if "proprio" in infos[0]:
                proprio_history.append(np.array([info["proprio"] for info in infos]))

            if "success" in infos[0]:
                success_history.append(np.array([info["success"] for info in infos]))

            if "visual_obs" in infos[0] and visual_observations is not None:
                visual_observations.append(
                    np.array([info["visual_obs"] for info in infos])
                )

            if planning_result.locations is not None:
                pred_locations = planning_result.locations.detach().cpu()
                pred_locations = pred_locations.squeeze(2)
                pred_locations_history.append(pred_locations)

            # stack states if necessary for next iteration
            if self.config.stack_states == 1:
                obs_t = current_obs
            else:
                obs_t = torch.cat(
                    [obs_t[:, current_obs.shape[1] :], current_obs], dim=1
                )

            loss_history.append(planning_result.losses)

            if bilevel_planning:
                if planning_result_l2.locations is not None:
                    pred_locations_l2 = planning_result_l2.locations.detach().cpu()
                    if pred_locations_l2.dim() > 2 and pred_locations_l2.shape[2] == 1:
                        pred_locations_l2 = pred_locations_l2.squeeze(2)
                    pred_locations_l2_history.append(pred_locations_l2)
                loss_history_l2.append(planning_result_l2.losses)

        observation_history = [
            self.normalizer.unnormalize_state(o) for o in observation_history
        ]

        self.model.train(orig_training_state)

        # put everything on cpu
        observation_history = [o.cpu() for o in observation_history]

        return MPCResult(
            observations=observation_history,
            locations=[torch.from_numpy(x) for x in location_history],
            action_history=action_history,
            reward_history=reward_history,
            pred_locations=pred_locations_history,
            final_preds_dist=final_preds_dist_history,
            targets=targets,
            loss_history=loss_history,
            ensemble_var_history=ensemble_var_history,
            ensemble_obs_var_history=ensemble_obs_var_history,
            ensemble_proprio_var_history=ensemble_proprio_var_history,
            qpos_history=[torch.from_numpy(x) for x in qpos_history],
            proprio_history=[torch.from_numpy(x) for x in proprio_history],
            # for hierarchy
            pred_locations_l2=pred_locations_l2_history,
            loss_history_l2=loss_history_l2,
            success_history=[torch.from_numpy(x) for x in success_history],
            visual_observations=[torch.from_numpy(x) for x in visual_observations],
            visual_targets=visual_targets,
        )

    def _get_relevant_pred(self, planning_result, cost_entity):
        """
        Get the relevant prediction to diff against the target
        """
        if cost_entity == "obs_component":
            pred = planning_result.pred_obs
            pred = flatten_conv_output(pred)
        elif cost_entity == "proprio_component":
            pred = planning_result.pred_proprio
        elif cost_entity == "location_component":
            pred = planning_result.pred_location
        elif cost_entity == "raw_locations":
            pred = planning_result.raw_locations
        else:
            raise NotImplementedError(f"Unknown cost entity {cost_entity}")
        return pred

    def _calculate_ensemble_var(self, ensemble_states):
        ensemble_states = flatten_ensemble_conv_output(ensemble_states)
        if ensemble_states.shape[1] > 1:
            ensemble_var = torch.var(ensemble_states, dim=1).sum(-1)  # (pred_T, bs)
        return ensemble_var

    def _log_ensemble_var(self, ensemble_var_history, plot_prefix="", var_type="state"):
        """
        Plot ensemble variance across predicted timesteps for different action steps
        Action step 0
        Action step N * 1/3
        Action step N * 2/3
        """

        steps = [0, len(ensemble_var_history) // 3, len(ensemble_var_history) * 2 // 3]

        for step in steps:
            var = ensemble_var_history[step]
            # average across batch
            var = var.mean(dim=1)

            Logger.run().log_line_plot(
                data=[[i, x.item()] for i, x in enumerate(var)],
                plot_name=f"{plot_prefix}_planning_step_{step}_ensemble_{var_type}_var",
            )

        # then plot the aggregate variance across the steps
        last_step = steps[-1]
        # make sure all the ensemble variances have the same length
        last_step_pred_len = ensemble_var_history[last_step].shape[0]
        aggregate_vars = [
            var[:last_step_pred_len].mean() for var in ensemble_var_history[:last_step]
        ]
        Logger.run().log_line_plot(
            data=[[i, x.item()] for i, x in enumerate(aggregate_vars)],
            plot_name=f"{plot_prefix}_planning_aggregate_ensemble_var",
        )
