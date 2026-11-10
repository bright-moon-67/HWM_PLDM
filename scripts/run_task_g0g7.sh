#!/usr/bin/env bash
# G0 vs G7 on a list of individual Hard tasks (single environment, 500 steps).
#
#   TASK_IDS="0 19 23 24" MAXJOBS=3 bash scripts/run_task_g0g7.sh
#
# G0 = warm-start shift only (no repair)
# G7 = warm-start + stuck-gated reachable projection (stuck_windows=2)
#
# Every task is sliced out of the 40-task trial file at its *global* index, so
# "task 19" is the same start/target/map as in the 40-task benchmark; the
# EnvsGenerator reads a fixed trial file from index 0, so n_envs=1 alone would
# always evaluate task 0.
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
LOGDIR=${LOGDIR:-${ROOT_DIR}/logs}
STATUS=${STATUS:-${LOGDIR}/queue_status.log}
CONFIG=${CONFIG:-configs/diverse_maze/icml/hard_hwm_exec.yaml}

TASK_IDS=${TASK_IDS:-"0 19 23 24"}
GROUP_LIST=${SWEEP_GROUPS:-"G0 G7"}
N_STEPS=${N_STEPS:-500}
SEED=${SEED:-42}
EVAL_SEED=${EVAL_SEED:-42}
MAXJOBS=${MAXJOBS:-3}

mkdir -p "${LOGDIR}"

# One single-task trial file per global task id.
python - "${TRIALS40}" "${PROBE}" ${TASK_IDS} <<'PY'
import sys
import torch
from pathlib import Path

src, probe_dir = sys.argv[1], Path(sys.argv[2])
task_ids = [int(x) for x in sys.argv[3:]]
trials = torch.load(src, weights_only=False)
for tid in task_ids:
    single = {
        k: (v[tid : tid + 1] if hasattr(v, "__len__") and len(v) == 40 else v)
        for k, v in trials.items()
    }
    dst = probe_dir / f"starts_targets_task{tid}.pt"
    torch.save(single, dst)
    print(f"wrote {dst} (global task {tid})")
PY

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
eval_cfg.h_d4rl_planning.exec_repair.repair_raise_errors=true"

group_overrides() {
    case "$1" in
        G0) echo "" ;;
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
    local tid=$1 group=$2
    local tag="task${tid}_${group}"
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
        eval_cfg.h_d4rl_planning.hard.set_start_target_path="${PROBE}/starts_targets_task${tid}.pt" \
        output_dir="${tag}" \
        seed="${SEED}" \
        eval_cfg.h_d4rl_planning.seed="${EVAL_SEED}" \
        eval_cfg.h_d4rl_planning.exec_repair.task_id_offset="${tid}" \
        ${COMMON} \
        ${extra} \
        >> "${LOGDIR}/${tag}.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') ${tag} exit ${rc}" >> "${STATUS}"
}

pids=()
for tid in ${TASK_IDS}; do
    for g in ${GROUP_LIST}; do
        run_one "${tid}" "${g}" &
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
done

wait
echo "$(date '+%F %T') g0g7 sweep done tasks=[${TASK_IDS}]" >> "${STATUS}"
