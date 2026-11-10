#!/usr/bin/env bash
# Task-4 parameter sweep (review 4 / task-level screening).
#
# Every group runs the SAME single environment: global task 4 of the 40-task
# hard benchmark (the task that failed under the baseline HWM and was rescued by
# the E3 projection), 500 steps, seed 42, warm-start shift ON.
#
#   bash scripts/run_task4_sweep.sh              # all 8 groups
#   SWEEP_GROUPS="G0 G3 G7" bash scripts/run_task4_sweep.sh
#   MAXJOBS=4 bash scripts/run_task4_sweep.sh
#
# The task file is sliced from the 40-task file by index 4 (NOT the first entry
# of a new file), so the maze / start / target are identical to the baseline.
set -u

ROOT_DIR=${ROOT_DIR:-/root/autodl-tmp/.autodl}
cd "${ROOT_DIR}/pldm"
source "${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pldm_test}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PROBE=${PROBE:-${ROOT_DIR}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe}
TRIALS40=${PROBE}/starts_targets_13_16.pt
TRIALS1=${PROBE}/starts_targets_task4.pt
LOGDIR=${LOGDIR:-${ROOT_DIR}/logs}
STATUS=${STATUS:-${LOGDIR}/queue_status.log}
CONFIG=${CONFIG:-configs/diverse_maze/icml/hard_hwm_exec.yaml}

N_STEPS=${N_STEPS:-500}
SEED=${SEED:-42}
EVAL_SEED=${EVAL_SEED:-42}
MAXJOBS=${MAXJOBS:-8}
# NOTE: do not call this variable GROUPS - bash reserves GROUPS for the user's
# group ids, so ${GROUPS:-default} silently expands to "0".
SWEEP_GROUPS=${SWEEP_GROUPS:-"G0 G1 G2 G3 G4 G5 G6 G7"}

mkdir -p "${LOGDIR}"

if [ ! -f "${TRIALS1}" ]; then
    python - "${TRIALS40}" "${TRIALS1}" <<'PY'
import sys
import torch

src, dst = sys.argv[1], sys.argv[2]
trials = torch.load(src, weights_only=False)
keys = {"starts", "targets", "map_layouts", "ood_dist", "mode", "block_dists", "turns"}
single = {k: (v[4:5] if k in keys else v) for k, v in trials.items()}
torch.save(single, dst)
print(f"wrote {dst}")
PY
fi

# Shared overrides: warm-start only, everything else off unless a group says so.
COMMON="eval_cfg.h_d4rl_planning.exec_repair.enabled=true \
eval_cfg.h_d4rl_planning.exec_repair.log_diagnostics=true \
eval_cfg.h_d4rl_planning.exec_repair.shift_l1_warm_start=true \
eval_cfg.h_d4rl_planning.exec_repair.retry_enabled=false \
eval_cfg.h_d4rl_planning.exec_repair.retry_isolate_rng=true \
eval_cfg.h_d4rl_planning.exec_repair.repair_enabled=false \
eval_cfg.h_d4rl_planning.exec_repair.repair_mode=projection \
eval_cfg.h_d4rl_planning.exec_repair.repair_gap_threshold=0.06 \
eval_cfg.h_d4rl_planning.exec_repair.repair_top_k=8 \
eval_cfg.h_d4rl_planning.exec_repair.repair_risk_weight=0.0 \
eval_cfg.h_d4rl_planning.exec_repair.repair_min_progress=0.0 \
eval_cfg.h_d4rl_planning.exec_repair.repair_min_improvement=0.0 \
eval_cfg.h_d4rl_planning.exec_repair.repair_trigger_mode=gap \
eval_cfg.h_d4rl_planning.exec_repair.repair_cooldown_replans=0 \
eval_cfg.h_d4rl_planning.exec_repair.repair_max_per_episode=0 \
eval_cfg.h_d4rl_planning.exec_repair.exec_feedback_enabled=false \
eval_cfg.h_d4rl_planning.exec_repair.high_feedback_enabled=false \
eval_cfg.h_d4rl_planning.exec_repair.repair_trigger_from_feedback=false \
eval_cfg.h_d4rl_planning.exec_repair.feedback_lambda_scale=false \
eval_cfg.h_d4rl_planning.exec_repair.repair_raise_errors=true \
eval_cfg.h_d4rl_planning.exec_repair.task_id_offset=4"

group_overrides() {
    case "$1" in
        G0) echo "" ;;
        G1) echo "eval_cfg.h_d4rl_planning.exec_repair.repair_enabled=true" ;;
        G2) echo "eval_cfg.h_d4rl_planning.exec_repair.repair_enabled=true \
eval_cfg.h_d4rl_planning.exec_repair.repair_gap_threshold=0.04" ;;
        G3) echo "eval_cfg.h_d4rl_planning.exec_repair.repair_enabled=true \
eval_cfg.h_d4rl_planning.exec_repair.repair_gap_threshold=0.09" ;;
        G4) echo "eval_cfg.h_d4rl_planning.exec_repair.repair_enabled=true \
eval_cfg.h_d4rl_planning.exec_repair.repair_top_k=16" ;;
        G5) echo "eval_cfg.h_d4rl_planning.exec_repair.repair_enabled=true \
eval_cfg.h_d4rl_planning.exec_repair.repair_min_improvement=0.005" ;;
        G6) echo "eval_cfg.h_d4rl_planning.exec_repair.repair_enabled=true \
eval_cfg.h_d4rl_planning.exec_repair.repair_gap_threshold=0.02 \
eval_cfg.h_d4rl_planning.exec_repair.repair_trigger_mode=stuck \
eval_cfg.h_d4rl_planning.exec_repair.exec_feedback_enabled=true \
eval_cfg.h_d4rl_planning.exec_repair.stuck_windows=1 \
eval_cfg.h_d4rl_planning.exec_repair.stuck_real_eps=0.15 \
eval_cfg.h_d4rl_planning.exec_repair.stuck_pred_eps=0.30" ;;
        G7) echo "eval_cfg.h_d4rl_planning.exec_repair.repair_enabled=true \
eval_cfg.h_d4rl_planning.exec_repair.repair_gap_threshold=0.02 \
eval_cfg.h_d4rl_planning.exec_repair.repair_trigger_mode=stuck \
eval_cfg.h_d4rl_planning.exec_repair.exec_feedback_enabled=true \
eval_cfg.h_d4rl_planning.exec_repair.stuck_windows=2 \
eval_cfg.h_d4rl_planning.exec_repair.stuck_real_eps=0.15 \
eval_cfg.h_d4rl_planning.exec_repair.stuck_pred_eps=0.30" ;;
        *) echo "" ;;
    esac
}

run_one() {
    local group=$1
    local tag="task4_${group}"
    local extra
    extra=$(group_overrides "${group}")
    echo "$(date '+%F %T') ${tag} start" >> "${STATUS}"
    python train.py --configs "${CONFIG}" --values \
        eval_only=true \
        load_l1_only=false \
        wandb=false \
        data.num_workers=0 \
        data.d4rl_config.num_workers=0 \
        eval_cfg.l2_latent_bounds_percentile=0 \
        eval_cfg.probing.max_samples=2000 \
        eval_cfg.h_d4rl_planning.n_envs_batch_size=1 \
        eval_cfg.h_d4rl_planning.hard.n_envs=1 \
        eval_cfg.h_d4rl_planning.hard.n_steps="${N_STEPS}" \
        eval_cfg.h_d4rl_planning.hard.set_start_target_path="${TRIALS1}" \
        output_dir="${tag}" \
        seed="${SEED}" \
        eval_cfg.h_d4rl_planning.seed="${EVAL_SEED}" \
        ${COMMON} \
        ${extra} \
        >> "${LOGDIR}/${tag}.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') ${tag} exit ${rc}" >> "${STATUS}"
}

pids=()
for g in ${SWEEP_GROUPS}; do
    run_one "${g}" &
    pids+=($!)
    while [ "${#pids[@]}" -ge "${MAXJOBS}" ]; do
        sleep 10
        alive=()
        for p in "${pids[@]}"; do
            if kill -0 "${p}" >/dev/null 2>&1; then
                alive+=("${p}")
            fi
        done
        pids=("${alive[@]}")
    done
done

wait
echo "$(date '+%F %T') task4 sweep done groups=[${SWEEP_GROUPS}]" >> "${STATUS}"
