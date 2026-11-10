from dataclasses import dataclass, field
from typing import NamedTuple
import torch
from pldm.configs import ConfigBase
from pldm.planning.enums import MPCConfig
from pldm.planning.planners.enums import PlannerConfig


@dataclass
class D4RLMPCConfig(MPCConfig):
    position_only: bool = True
    subgoal_planning: bool = False
    set_full_states: bool = False
    unique_shortest_path: bool = (
        False  # gen trials where there's a unique shortest path
    )


@dataclass
class ExecRepairConfig(ConfigBase):
    """Switches for the execution-aware HWM research line.

    Everything is disabled by default so that the vanilla HWM baseline can be
    reproduced bit-for-bit: when ``enabled`` is False no extra computation runs
    and the original planning path is used unchanged.

    The switches are intentionally staged (diagnose -> retry -> repair ->
    feedback) so each research step can be ablated independently.
    """

    # Master switch. While False, no research code path may be entered.
    enabled: bool = False

    # Step 1/2: record cross-level disagreement and execution error diagnostics.
    log_diagnostics: bool = False

    # ---- ECCHP stage 1: plan-level diagnostics ----
    # Record one ``plan_decision`` / ``execution_outcome`` / ``plan_transition``
    # line per environment and replanning window. This is purely observational:
    # it must not change the executed actions, but it does cost extra L1->L2
    # projections, so it has its own switch.
    log_plan_events: bool = False
    # Number of replanning windows used for the route indicators
    # (``route_path_length`` / ``route_efficiency`` / ``route_revisit_count``).
    route_history_window: int = 8
    # A position counts as a revisit when the agent returns within this radius
    # (metres) of an earlier position inside the window.
    revisit_radius: float = 0.25

    # ---- ECCHP stage 2: counterfactual candidate collection ----
    # When on, every decision that runs the reselection branch also exposes its
    # candidate set. The evaluator then replays every candidate from the *same*
    # environment snapshot, continues with the normal HWM controller and records
    # the realised graph-distance return (``candidate_outcome`` events).
    # This is an expensive, offline-only measurement: it is never used online.
    collect_candidates: bool = False
    # Observation horizons for the counterfactual returns, in primitive steps.
    candidate_horizons: str = "40,80"
    # Cap on the number of decisions per chunk that are replayed (0 = no cap).
    candidate_max_decisions: int = 0

    # Step 3: give the low-level MPPI planner a conditional second attempt.
    retry_enabled: bool = False

    # Step 4: project the high-level subgoal onto low-level reachable states.
    repair_enabled: bool = False
    # Which repair branch to run when triggered:
    #   "reselection" -> re-search the L2 plan and re-rank candidates (step 5)
    #   "projection"  -> project the subgoal onto L1 reachable states (step 4)
    repair_mode: str = "reselection"
    # Step 4: how many cheapest MPPI samples to roll out as reachable candidates.
    repair_top_k: int = 8
    # Step 4: weight of the normalised MPPI sample cost inside the candidate
    # score (0 = pure distance to the original subgoal).
    repair_risk_weight: float = 0.0
    # Step 4: a candidate must reduce the distance to the original subgoal by
    # more than this (absolute, in the L2 mean-squared units used everywhere).
    repair_min_progress: float = 0.0
    # How many extra L2 plans to sample when the repair branch fires. Each
    # candidate costs one extra L2 plan plus one extra L1 plan.
    high_resample_count: int = 2
    # Step 5: weight lambda_0 of the low-level reachability term in
    #   score = J_H_norm + lambda * C_L_norm
    high_level_cost_weight: float = 0.5
    # How the two terms are put on a common scale:
    #   "baseline" -> relative to candidate 0 (score = J/J0 + lambda * C/C0)
    #   "minmax"   -> min-max across the (few) candidates of this decision
    repair_normalize_mode: str = "baseline"
    # Re-sample every high-level candidate from the *same* pre-repair warm start
    # so the candidates are independent instead of being refined on top of each
    # other ("sequential" keeps the old behaviour as an ablation).
    high_resample_independent: bool = True
    # Feasibility gate for step 5 (review 3): a candidate is admissible only if
    # its predicted progress towards its own subgoal is at least this, and its
    # cross-level gap is at most ``repair_max_cross_mse`` (<=0 disables).
    repair_min_pred_progress: float = 0.0
    repair_max_cross_mse: float = 0.0
    # Accept a repaired macro action only if it lowers the cross-level gap by
    # more than this amount.
    repair_min_improvement: float = 0.0
    # ECCHP stage 0: keep the legacy "the terminal cross-level gap must
    # improve" acceptance rule. Switching it off leaves only the feasibility
    # gate and the combined score (J_H + lambda * C_L), which is what
    # execution-calibrated switching needs: a detour that is temporarily
    # further from the subgoal but long-term better must not be vetoed by the
    # old rule. Default True preserves the historical behaviour.
    repair_require_gap_improvement: bool = True
    # Cross-level gap above which the repair branch is allowed to fire. Kept
    # separate from ``retry_gap_threshold`` so E2 (retry) and E3/E4 (repair) use
    # their own budgets; 0 falls back to ``retry_gap_threshold``.
    repair_gap_threshold: float = 0.0

    # ---- Review 4: repair only after a confirmed execution failure ----
    # "gap"   -> fire whenever cross_mse > repair_gap_threshold (E3 behaviour)
    # "stuck" -> fire only when the low level predicted motion and the robot did
    #            not move for ``stuck_windows`` consecutive windows
    repair_trigger_mode: str = "gap"
    # A window is "stuck" when the realised displacement is below
    # ``stuck_real_eps`` while the predicted displacement exceeded
    # ``stuck_pred_eps`` (both in metres per replan window).
    stuck_real_eps: float = 0.15
    stuck_pred_eps: float = 0.30
    stuck_windows: int = 2
    # Never fire twice within this many replans, and at most this many times per
    # episode. Both limit the "repair oscillation" seen on the regressed tasks.
    repair_cooldown_replans: int = 0
    repair_max_per_episode: int = 0
    # Only accept a repair when recent prediction has been unreliable; a
    # reliable high-level plan is not overridden for a marginal gap gain.
    repair_require_exec_failure: bool = False
    repair_reliable_trust: float = 0.05
    repair_reliable_min_improvement: float = 0.02
    # Debug: compute repairs but drop them (must reproduce the baseline exactly;
    # used to verify RNG/warm-start isolation).
    repair_force_reject: bool = False
    # Restore the CPU/CUDA RNG state when *no* environment accepted a repair, so
    # that a rejected repair leaves later randomness untouched (stage 0).
    repair_restore_rng_on_reject: bool = True
    # ... and does not make the cross-level gap worse by more than this.
    repair_gap_tolerance: float = 0.0
    # Re-raise exceptions from the experimental repair branch (debugging only).
    repair_raise_errors: bool = False

    # ---- Step 6: online execution feedback ----
    # Adapt lambda_t = clip(lambda_0 * (1 + kappa * EMA(exec_error))) per env.
    exec_feedback_enabled: bool = False
    feedback_beta: float = 0.1
    feedback_kappa: float = 1.0
    feedback_lambda_min: float = 0.1
    feedback_lambda_max: float = 10.0
    # Review 3: scaling the weight of a *model prediction* by how unreliable the
    # model has been is backwards. Off by default; keep it only as an ablation.
    feedback_lambda_scale: bool = False
    # Whether recent model unreliability gates/enables high-level repair.
    repair_trigger_from_feedback: bool = False
    # Threshold on ``U_exec = ema / (1 + ema)``. On hard mazes the execution
    # error is O(0.1), i.e. ``U_exec`` ~ 0.05-0.1, so this default is low on
    # purpose; calibrate it on a development split, never on the test set.
    feedback_trigger_threshold: float = 0.05
    # Execution windows with a smaller initial distance are ignored: the
    # relative progress ratio is numerically meaningless there.
    feedback_min_initial_dist: float = 1e-4

    # Step 5/6: re-rank high-level candidates with low-level feedback.
    high_feedback_enabled: bool = False

    # ---- Step 3: conditional low-level retry ----
    # Trigger a retry only when the cross-level gap exceeds this value.
    # A non-positive value disables the retry path even if retry_enabled=True.
    retry_gap_threshold: float = 0.0
    # Which quantity decides *whether* to retry:
    #   "cross_mse"     -> absolute terminal gap > retry_gap_threshold
    #   "pred_progress" -> predicted progress < retry_progress_threshold
    # Predicting "this plan closes less than X% of the subgoal distance in the
    # steps we will actually execute" is closer to a failure signal than the
    # absolute gap, which mostly tracks how far the subgoal happens to be.
    retry_trigger_metric: str = "cross_mse"
    retry_progress_threshold: float = 0.0
    # Accept a retry only if it lowers the gap by more than these amounts.
    retry_min_improvement: float = 0.0
    retry_min_improvement_ratio: float = 0.0
    # Restore CPU/CUDA RNG state when no environment accepts the retry, so a
    # fully rejected retry leaves the episode's randomness untouched.
    retry_restore_rng_on_reject: bool = True
    # Additionally require the retried plan to be no worse on the path-level
    # cross-level cost (guards against terminal-only improvements).
    retry_require_path_cost_improvement: bool = True
    # Soft alternative to the hard path-cost guard: accept when
    # ``d_T + weight * J_path`` improves. A positive value overrides
    # ``retry_require_path_cost_improvement`` (review-2 section 5 ablation).
    retry_path_cost_weight: float = 0.0

    # Index into pred_obs used as the high-level subgoal. pred_obs[0] is the
    # current state, so index 1 is one macro step (= step_skip primitives).
    subgoal_index: int = 1

    # ---- Warm start hygiene (P0) ----
    # The low-level MPPI warm start is only refreshed when ``plan_size`` changes,
    # which never happens for the constant-length L1 plans HWM uses. With this
    # switch the nominal trajectory is shifted forward by ``replan_every``
    # primitives before every replan so that index 0 again corresponds to the
    # next executed step. Pure planning hygiene: it does not add any new
    # mechanism, it only stops the warm start from being stale.
    shift_l1_warm_start: bool = False

    # ---- RNG isolation (P1) ----
    # Run the retried low-level plan inside ``torch.random.fork_rng`` so a retry
    # cannot perturb the episode's global random stream (and therefore cannot
    # silently change later disturbance / evaluation randomness).
    retry_isolate_rng: bool = True

    # Global task id = task_id_offset + chunk_offset + env_index.
    # Sharded runs pass ``TASKS_PER_SHARD * shard_index`` here.
    task_id_offset: int = 0

    # Where ``DiagnosticsRecorder`` writes its JSONL file. Empty means
    # "use the evaluator output directory" (wired up in a later step).
    output_dir: str = ""
    diagnostics_filename: str = "diagnostics.jsonl"


@dataclass
class HierarchicalD4RLMPCConfig(D4RLMPCConfig):
    final_trans_norm_cutoff: float = 4
    final_trans_steps: int = 15
    error_threshold: float = 1.0
    mock_l1: bool = False
    level2: PlannerConfig = PlannerConfig()
    exec_repair: ExecRepairConfig = field(default_factory=ExecRepairConfig)


class MPCReport(NamedTuple):
    success_rate: float
    success: torch.Tensor
    avg_steps_to_goal: float
    median_steps_to_goal: float
    terminations: list
    one_turn_success_rate: float
    two_turn_success_rate: float
    three_turn_success_rate: float
    num_one_turns: int
    num_two_turns: int
    num_three_turns: int
    num_turns: list
    block_dists: list
    ood_report: dict

    def build_log_dict(self, prefix=""):
        log_dict = {
            f"{prefix}_planning_success_rate": self.success_rate,
            f"{prefix}_avg_steps_to_goal": self.avg_steps_to_goal,
            f"{prefix}_median_steps_to_goal": self.median_steps_to_goal,
            f"{prefix}_one_turn_success_rate": self.one_turn_success_rate,
            f"{prefix}_two_turn_success_rate": self.two_turn_success_rate,
            f"{prefix}_three_turn_success_rate": self.three_turn_success_rate,
            f"{prefix}_num_one_turns": self.num_one_turns,
            f"{prefix}_num_two_turns": self.num_two_turns,
            f"{prefix}_num_three_turns": self.num_three_turns,
            # f"{prefix}_block_dist_p": self.block_dist_p,
            # f"{prefix}_num_turns_p": self.num_turns_p,
        }

        log_dict.update(self.ood_report)
        return log_dict
