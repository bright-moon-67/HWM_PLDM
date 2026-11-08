#!/usr/bin/env bash
# Hard-only HWM 两级规划：分片队列（2 路并发，避免 24GB 显存 OOM）
#
# 常用环境变量：
#   CONFIG=configs/diverse_maze/icml/hard_hwm_exec.yaml   # 实验配置
#   RUN_TAG=step1_diag          # 产物目录前缀：${RUN_TAG}_shard{i}
#   SHARDS="0 1 2 3 4 5 6 7"   # 要跑的分片
#   MAXJOBS=2                   # 并发数
#   SEED_BASE=246 EVAL_SEED_BASE=42
#   TASKS_PER_SHARD=5
#   WAIT_FOR=""                 # 可选：先等这些旧分片结束，如 "0 3"
#   EXTRA_VALUES="a.b=1 c.d=2"  # 额外的 --values 覆盖项
set -u

ROOT_DIR=${ROOT_DIR:-/root/autodl-tmp/.autodl}
cd "${ROOT_DIR}/pldm"
source "${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pldm_test}"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROBE=${PROBE:-${ROOT_DIR}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe}
LOGDIR=${LOGDIR:-${ROOT_DIR}/logs}
STATUS=${STATUS:-${LOGDIR}/queue_status.log}

CONFIG=${CONFIG:-configs/diverse_maze/icml/hard_hwm_parallel.yaml}
RUN_TAG=${RUN_TAG:-hard_hwm}
MAXJOBS=${MAXJOBS:-2}
SHARDS=${SHARDS:-"0 1 2 3 4 5 6 7"}
TASKS_PER_SHARD=${TASKS_PER_SHARD:-5}
SEED_BASE=${SEED_BASE:-246}
EVAL_SEED_BASE=${EVAL_SEED_BASE:-42}
WAIT_FOR=${WAIT_FOR:-}
EXTRA_VALUES=${EXTRA_VALUES:-}

mkdir -p "${LOGDIR}"
echo "$(date '+%F %T') queue start tag=${RUN_TAG} config=${CONFIG} shards=[${SHARDS}] maxjobs=${MAXJOBS}" >> "${STATUS}"

run_one() {
    local i=$1
    local tag="${RUN_TAG}_shard${i}"
    echo "$(date '+%F %T') shard ${i} -> ${tag} start (pid $$)" >> "${STATUS}"
    python train.py --configs "${CONFIG}" --values \
        eval_only=true \
        load_l1_only=false \
        wandb=false \
        data.num_workers=0 \
        data.d4rl_config.num_workers=0 \
        eval_cfg.l2_latent_bounds_percentile=0 \
        eval_cfg.probing.max_samples=2000 \
        eval_cfg.h_d4rl_planning.hard.n_envs="${TASKS_PER_SHARD}" \
        eval_cfg.h_d4rl_planning.n_envs_batch_size="${TASKS_PER_SHARD}" \
        eval_cfg.h_d4rl_planning.hard.set_start_target_path="${PROBE}/starts_targets_13_16_shard${i}.pt" \
        output_dir="${tag}" \
        seed=$((SEED_BASE + i)) \
        eval_cfg.h_d4rl_planning.seed=$((EVAL_SEED_BASE + i)) \
        eval_cfg.h_d4rl_planning.exec_repair.task_id_offset=$((TASKS_PER_SHARD * i)) \
        ${EXTRA_VALUES} \
        >> "${LOGDIR}/${tag}.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') shard ${i} exit ${rc}" >> "${STATUS}"
}

# 可选：先等待指定的旧分片结束（例如上一轮残留的 0/3 号）
if [ -n "${WAIT_FOR}" ]; then
    for w in ${WAIT_FOR}; do
        while pgrep -f "output_dir=${RUN_TAG}_shard${w}" >/dev/null 2>&1; do
            sleep 20
        done
        echo "$(date '+%F %T') old shard ${w} finished" >> "${STATUS}"
    done
fi

pids=()
for i in ${SHARDS}; do
    run_one "${i}" &
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
echo "$(date '+%F %T') ALL DONE" >> "${STATUS}"
