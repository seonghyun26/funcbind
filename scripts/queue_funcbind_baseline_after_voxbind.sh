#!/usr/bin/env bash
# Wait for the active VoxBind training chain, then launch the MCP holo-density
# FuncBind baseline on all four GPUs.
set -uo pipefail

REPO="${REPO:-${FUNCBIND_ROOT:-/home1/irteam/funcbind}}"
VOXBIND_PID="${VOXBIND_PID:-1273140}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PY="${PY:-$REPO/.repro-env/bin/python}"
QUEUE_LOG="${QUEUE_LOG:-$REPO/exps/funcbind/_queues/funcbind_mcpp_holo_after_voxbind.log}"

mkdir -p "$(dirname "$QUEUE_LOG")"
exec > >(tee -a "$QUEUE_LOG") 2>&1

say() {
    echo "[$(date --iso-8601=seconds)] [funcbind-baseline-queue] $*"
}

say "armed: VoxBind supervisor pid=$VOXBIND_PID -> FuncBind train_fb_mcpp_holo_density on GPUs 0,1,2,3"

checks=0
while kill -0 "$VOXBIND_PID" 2>/dev/null; do
    cmdline="$(tr '\0' ' ' < "/proc/$VOXBIND_PID/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" != *"68_train_champion_receptor_ed_after_sampling.sh"* ]]; then
        say "pid $VOXBIND_PID no longer belongs to the expected VoxBind supervisor; switching to GPU-idle checks"
        break
    fi
    if (( checks % 10 == 0 )); then
        say "waiting for the active VoxBind training chain"
    fi
    sleep "$POLL_SECONDS"
    checks=$((checks + 1))
done

say "VoxBind supervisor is gone; waiting for all four GPUs to remain idle"

if pgrep -af 'train_fb.py --config-name train_fb_mcpp_holo_density' >/dev/null 2>&1; then
    say "a matching FuncBind baseline process already exists; refusing a duplicate launch"
    exit 3
fi

"$REPO/scripts/wait_for_gpus.sh" \
    --gpus 0,1,2,3 \
    --poll-seconds "$POLL_SECONDS" \
    --stable-checks 3 \
    --timeout-seconds 0 \
    -- env \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        PYTHONUNBUFFERED=1 \
        "$PY" "$REPO/funcbind/train_fb.py" \
        --config-name train_fb_mcpp_holo_density
rc=$?

say "FuncBind baseline exited with code $rc"
exit "$rc"
