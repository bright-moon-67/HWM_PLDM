#!/usr/bin/env bash
# Run ONE Hard-maze task with diagnostics (short smoke tests / case studies).
#
#   TASK_ID=4 N_STEPS=200 RUN_TAG=single_task4 bash scripts/run_one_task.sh
#   TASK_ID=4 N_STEPS=200 RUN_TAG=retry_task4 RETRY=1 THRESHOLD=0.06 \
#       bash scripts/run_one_task.sh
#
# The script slices the official shard trial file so the task keeps its original
# map / start / target. Seeds follow the original shard mapping.
set -u

ROOT_DIR=${ROOT_DIR:-/root/autodl-tmp/.autodl}
cd "${ROOT_DIR}"
source "${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pldm_test}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1

TASK_ID=${TASK_ID:?set TASK_ID (global 0..39)}
TASKS_PER_SHARD=${TASKS_PER_SHARD:-5}
SHARD=${SHARD:-$((TASK_ID / TASKS_PER_SHARD))}
LOCAL_IDX=${LOCAL_IDX:-$((TASK_ID % TASKS_PER_SHARD))}
N_STEPS=${N_STEPS:-200}
RUN_TAG=${RUN_TAG:-single_task${TASK_ID}}
RETRY=${RETRY:-0}
THRESHOLD=${THRESHOLD:-0.0}
LOGDIR=${LOGDIR:-${ROOT_DIR}/logs}
CONFIG=${CONFIG:-configs/diverse_maze/icml/hard_hwm_exec.yaml}

if [ "${RETRY}" = "1" ]; then
    RETRY_BOOL=true
else
    RETRY_BOOL=false
fi

PROBE=${ROOT_DIR}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe
TRIALS=${PROBE}/starts_targets_13_16_shard${SHARD}.pt
TRIALS_SINGLE=${PROBE}/starts_targets_single_task${TASK_ID}.pt

mkdir -p "${LOGDIR}"

python - "${TRIALS}" "${LOCAL_IDX}" "${TRIALS_SINGLE}" <<'PY'
import sys
import torch

src, idx, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
trials = torch.load(src, weights_only=False)
single = {
    key: ([value[idx]] if isinstance(value, list) else value)
    for key, value in trials.items()
}
torch.save(single, dst)
print(f"wrote {dst} (task local index {idx})")
PY

echo "TASK_ID=${TASK_ID} SHARD=${SHARD} LOCAL_IDX=${LOCAL_IDX} N_STEPS=${N_STEPS}" \
     "RETRY=${RETRY_BOOL} THRESHOLD=${THRESHOLD} RUN_TAG=${RUN_TAG}"

cd "${ROOT_DIR}/pldm"
python train.py --configs "${CONFIG}" --values \
    eval_only=true \
    load_l1_only=false \
    wandb=false \
    data.num_workers=0 \
    data.d4rl_config.num_workers=0 \
    eval_cfg.l2_latent_bounds_percentile=0 \
    eval_cfg.probing.max_samples=2000 \
    eval_cfg.h_d4rl_planning.hard.n_envs=1 \
    eval_cfg.h_d4rl_planning.n_envs_batch_size=1 \
    eval_cfg.h_d4rl_planning.hard.n_steps="${N_STEPS}" \
    eval_cfg.h_d4rl_planning.hard.set_start_target_path="${TRIALS_SINGLE}" \
    output_dir="${RUN_TAG}" \
    seed=$((246 + SHARD)) \
    eval_cfg.h_d4rl_planning.seed=$((42 + SHARD)) \
    eval_cfg.h_d4rl_planning.exec_repair.task_id_offset="${TASK_ID}" \
    eval_cfg.h_d4rl_planning.exec_repair.enabled=true \
    eval_cfg.h_d4rl_planning.exec_repair.log_diagnostics=true \
    eval_cfg.h_d4rl_planning.exec_repair.retry_enabled="${RETRY_BOOL}" \
    eval_cfg.h_d4rl_planning.exec_repair.retry_gap_threshold="${THRESHOLD}" \
    > "${LOGDIR}/${RUN_TAG}.log" 2>&1
