#!/usr/bin/env bash
# Chained Task-4 sweep: wait for the currently running groups, then run the rest
# three at a time.
#
# Why 3 and not 4+: a single-environment process grows its PyTorch caching
# allocator to ~6.5-7 GB when the GPU has room, and does not give it back. Three
# such processes already use ~20 of 24.5 GB, so a fourth allocation peak (the
# MPPI rollout wants ~0.9 GB at once) fails with
# "CUDA out of memory. Tried to allocate 904.00 MiB".
set -u

ROOT_DIR=${ROOT_DIR:-/root/autodl-tmp/.autodl}
cd "${ROOT_DIR}"

STATUS=${STATUS:-${ROOT_DIR}/logs/queue_status.log}
WAIT_TAGS=${WAIT_TAGS:-"G0 G1 G4"}
RUN_GROUPS=${RUN_GROUPS:-"G2 G3 G5 G6 G7"}
MAXJOBS=${MAXJOBS:-3}

for g in ${WAIT_TAGS}; do
    while pgrep -f "output_dir=task4_${g}" >/dev/null 2>&1; do
        sleep 20
    done
done
echo "$(date '+%F %T') task4 chain: previous groups finished, starting [${RUN_GROUPS}]" >> "${STATUS}"

MAXJOBS=${MAXJOBS} SWEEP_GROUPS="${RUN_GROUPS}" bash scripts/run_task4_sweep.sh

echo "$(date '+%F %T') task4 chain: done" >> "${STATUS}"
