#!/usr/bin/env bash
# E3 / E4 ablation queue (review-3 experiment matrix).
#
#   E3 = projection only      (retry off, reselection off, feedback off)
#   E4 = reselection only     (retry off, projection off, feedback off)
#
# Both arms run with the warm-start shift OFF, so they are directly comparable to
# the frozen 40-task baseline (E0) and to each other.
#
#   ARM=e3 SHARDS="0 1 2" MAXJOBS=3 bash scripts/run_e3_e4_queue.sh
#   ARM=e4 SHARDS="0 1"   MAXJOBS=2 bash scripts/run_e3_e4_queue.sh
#
# Why E4 gets fewer slots: one extra high-level candidate costs a full L2 MPPI
# plan (4000 samples x ~47 macro steps), i.e. tens of seconds. With three
# processes on one 24 GB GPU the reselection arm degrades to ~2x the per-step
# time even when it does not trigger, so it is scheduled with MAXJOBS=2 and a
# single extra candidate.
#
# Tunables:
#   ARM                   e3 | e4 | both   (default both)
#   MAXJOBS               concurrent train.py processes (default 3)
#   SHARDS                shards to run (default "0 1 2 3 4 5 6 7")
#   REPAIR_GAP_THRESHOLD  gap above which repair fires (default 0.06 = the 75th
#                         percentile of the diagnostics-only cross_mse
#                         distribution, i.e. a fixed ~25% trigger budget)
#   PROJECTION_TOP_K      E3 reachable candidates per decision (default 8)
#   RESAMPLE_COUNT        E4 extra high-level candidates (default 1)
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
LOGDIR=${LOGDIR:-${ROOT_DIR}/logs}
STATUS=${STATUS:-${LOGDIR}/queue_status.log}
CONFIG=${CONFIG:-configs/diverse_maze/icml/hard_hwm_exec.yaml}

ARM=${ARM:-both}
SHARDS=${SHARDS:-"0 1 2 3 4 5 6 7"}
MAXJOBS=${MAXJOBS:-3}
TASKS_PER_SHARD=${TASKS_PER_SHARD:-5}
SEED_BASE=${SEED_BASE:-246}
EVAL_SEED_BASE=${EVAL_SEED_BASE:-42}
REPAIR_GAP_THRESHOLD=${REPAIR_GAP_THRESHOLD:-0.06}
PROJECTION_TOP_K=${PROJECTION_TOP_K:-8}
RESAMPLE_COUNT=${RESAMPLE_COUNT:-1}
RUN_TAG_E3=${RUN_TAG_E3:-e3_projection}
RUN_TAG_E4=${RUN_TAG_E4:-e4_reselection}

case "${ARM}" in
    e3) ARMS="${RUN_TAG_E3}" ;;
    e4) ARMS="${RUN_TAG_E4}" ;;
    *)  ARMS="${RUN_TAG_E3} ${RUN_TAG_E4}" ;;
esac

mkdir -p "${LOGDIR}"
echo "$(date '+%F %T') E3/E4 queue start arm=[${ARMS}] shards=[${SHARDS}] maxjobs=${MAXJOBS} gap=${REPAIR_GAP_THRESHOLD} resample=${RESAMPLE_COUNT}" >> "${STATUS}"

run_one() {
    local arm=$1 i=$2
    local tag="${arm}_shard${i}"
    local extra
    if [ "${arm}" = "${RUN_TAG_E3}" ]; then
        extra="eval_cfg.h_d4rl_planning.exec_repair.repair_mode=projection \
eval_cfg.h_d4rl_planning.exec_repair.repair_top_k=${PROJECTION_TOP_K} \
eval_cfg.h_d4rl_planning.exec_repair.high_resample_count=0"
    else
        extra="eval_cfg.h_d4rl_planning.exec_repair.repair_mode=reselection \
eval_cfg.h_d4rl_planning.exec_repair.high_resample_count=${RESAMPLE_COUNT} \
eval_cfg.h_d4rl_planning.exec_repair.repair_top_k=0"
    fi
    echo "$(date '+%F %T') ${tag} start" >> "${STATUS}"
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
        eval_cfg.h_d4rl_planning.exec_repair.enabled=true \
        eval_cfg.h_d4rl_planning.exec_repair.log_diagnostics=true \
        eval_cfg.h_d4rl_planning.exec_repair.retry_enabled=false \
        eval_cfg.h_d4rl_planning.exec_repair.repair_enabled=true \
        eval_cfg.h_d4rl_planning.exec_repair.repair_gap_threshold="${REPAIR_GAP_THRESHOLD}" \
        eval_cfg.h_d4rl_planning.exec_repair.shift_l1_warm_start=false \
        eval_cfg.h_d4rl_planning.exec_repair.exec_feedback_enabled=false \
        eval_cfg.h_d4rl_planning.exec_repair.high_feedback_enabled=false \
        ${extra} \
        >> "${LOGDIR}/${tag}.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') ${tag} exit ${rc}" >> "${STATUS}"
}

pids=()
for i in ${SHARDS}; do
    for arm in ${ARMS}; do
        run_one "${arm}" "${i}" &
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
echo "$(date '+%F %T') E3/E4 queue done arm=[${ARMS}]" >> "${STATUS}"
