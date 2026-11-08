#!/usr/bin/env bash
# Orchestrates the E3/E4 ablation in two phases so the heavy reselection arm does
# not compete with the light projection arm for GPU memory:
#
#   phase 1: finish E3 (projection) on all 8 shards, 3 concurrent
#   phase 2: E4 (reselection, 1 extra candidate) on all 8 shards, 2 concurrent
#
# Shards 0 and 1 of E3 may already be running when this starts; it waits for them
# so they are not duplicated.
set -u

ROOT_DIR=${ROOT_DIR:-/root/autodl-tmp/.autodl}
cd "${ROOT_DIR}"

STATUS=${STATUS:-${ROOT_DIR}/logs/queue_status.log}

for w in ${ALREADY_RUNNING:-0 1}; do
    while pgrep -f "output_dir=e3_projection_shard${w}" >/dev/null 2>&1; do
        sleep 30
    done
done
echo "$(date '+%F %T') E3 shards ${ALREADY_RUNNING:-0 1} finished" >> "${STATUS}"

# Remaining E3 shards, three at a time.
ARM=e3 SHARDS=${E3_SHARDS:-"2 3 4 5 6 7"} MAXJOBS=${E3_MAXJOBS:-3} \
    bash scripts/run_e3_e4_queue.sh

# Reselection arm: one extra candidate, two slots.
ARM=e4 SHARDS=${E4_SHARDS:-"0 1 2 3 4 5 6 7"} MAXJOBS=${E4_MAXJOBS:-2} \
    RESAMPLE_COUNT=${RESAMPLE_COUNT:-1} \
    bash scripts/run_e3_e4_queue.sh

echo "$(date '+%F %T') phased E3/E4 orchestration done" >> "${STATUS}"
