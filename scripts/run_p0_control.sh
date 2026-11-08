#!/usr/bin/env bash
# Matched control for the P0 warm-start ablation.
#
# Waits for the running ``p0_shift`` shards to finish, then runs the *same*
# shards / seeds / step budget with ``shift_l1_warm_start=false`` but the same
# diagnostics enabled. Only the warm-start flag differs between the two runs, so
# the pair is directly comparable.
set -u

ROOT_DIR=${ROOT_DIR:-/root/autodl-tmp/.autodl}
cd "${ROOT_DIR}"

STATUS=${STATUS:-${ROOT_DIR}/logs/queue_status.log}

echo "$(date '+%F %T') p0 control: waiting for p0_shift to finish" >> "${STATUS}"
while pgrep -f "output_dir=p0_shift_shard" >/dev/null 2>&1; do
    sleep 60
done
echo "$(date '+%F %T') p0 control: p0_shift finished, starting p0_noshift" >> "${STATUS}"

CONFIG=configs/diverse_maze/icml/hard_hwm_exec.yaml \
RUN_TAG=p0_noshift \
SHARDS="1 6" \
MAXJOBS=2 \
EXTRA_VALUES="eval_cfg.h_d4rl_planning.exec_repair.enabled=true eval_cfg.h_d4rl_planning.exec_repair.log_diagnostics=true eval_cfg.h_d4rl_planning.exec_repair.retry_enabled=false eval_cfg.h_d4rl_planning.exec_repair.shift_l1_warm_start=false" \
bash scripts/run_hard_hwm_queue.sh

echo "$(date '+%F %T') p0 control: done" >> "${STATUS}"
