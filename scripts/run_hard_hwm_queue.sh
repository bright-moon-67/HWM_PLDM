#!/usr/bin/env bash
# Hard-only HWM 两级规划：分片队列（2 路并发，避免 24GB 显存 OOM）
set -u

cd /root/autodl-tmp/.autodl/pldm
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pldm_test
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROBE=/root/autodl-tmp/.autodl/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe
LOGDIR=/root/autodl-tmp/.autodl/logs
STATUS=${LOGDIR}/queue_status.log
MAXJOBS=${MAXJOBS:-2}
# 已经在跑的分片：0 3；需要补跑的分片：1 2 4 5 6 7
SHARDS=${SHARDS:-"1 2 4 5 6 7"}

mkdir -p "${LOGDIR}"
echo "$(date '+%F %T') queue start, shards=[${SHARDS}] maxjobs=${MAXJOBS}" >> "${STATUS}"

run_one() {
    local i=$1
    echo "$(date '+%F %T') shard ${i} start (pid $$)" >> "${STATUS}"
    python train.py --configs configs/diverse_maze/icml/hard_hwm_parallel.yaml --values \
        eval_only=true \
        load_l1_only=false \
        wandb=false \
        data.num_workers=0 \
        data.d4rl_config.num_workers=0 \
        eval_cfg.l2_latent_bounds_percentile=0 \
        eval_cfg.probing.max_samples=2000 \
        eval_cfg.h_d4rl_planning.hard.n_envs=5 \
        eval_cfg.h_d4rl_planning.n_envs_batch_size=5 \
        eval_cfg.h_d4rl_planning.hard.set_start_target_path="${PROBE}/starts_targets_13_16_shard${i}.pt" \
        output_dir="hard_hwm_shard${i}" \
        seed=$((246 + i)) \
        eval_cfg.h_d4rl_planning.seed=$((42 + i)) \
        >> "${LOGDIR}/hard_hwm_shard${i}.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') shard ${i} exit ${rc}" >> "${STATUS}"
}

# 等待已经在跑的分片 0 / 3 结束
while pgrep -f "output_dir=hard_hwm_shard0" >/dev/null 2>&1 || pgrep -f "output_dir=hard_hwm_shard3" >/dev/null 2>&1; do
    sleep 20
done
echo "$(date '+%F %T') running shards 0/3 finished" >> "${STATUS}"

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
