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

    # Step 3: give the low-level MPPI planner a conditional second attempt.
    retry_enabled: bool = False

    # Step 4: project the high-level subgoal onto low-level reachable states.
    repair_enabled: bool = False

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
