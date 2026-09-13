#!/usr/bin/env bash
# Bring the FuncBind fine-tune back once some OTHER GPU job finishes.
#
#   WAIT_PATTERN='[s]ample\.py.*260908_fusion_default_cv2_scratch_8gpu' \
#   BASE_NAME=20260910_fb_mcpp_default_atomblob7_zeroinit \
#   CONFIG=train_fb_mcpp_holo_density_default \
#     setsid nohup bash scripts/chain_resume_funcbind_after_gpu_job.sh &
#
# The sibling chain (chain_resume_funcbind_after_mcp_run.sh) keys on OUR OWN sampling
# round's exit_code files. This one keys on an arbitrary process instead, for the common
# case where funcbind was stopped to hand the GPUs to a VoxBind run.
#
# Relaunching the watchdog IS the resume: with no trainer running it picks the
# furthest-along checkpoint by acc_iter -- never by epoch, which restarts at 0 on every
# resume -- and continues into the next free _rN.
#
# ROOT_SRC POINTS AT NOTHING ON PURPOSE. best_source() ranks candidates by acc_iter, and
# the density-free base (exps/funcbind/fb_unified) carries acc_iter=87,214,080 from its own
# pretraining. Leaving the base in the ranking would make it permanently "furthest along",
# so every restart would silently begin the fine-tune again from zero.
set -uo pipefail

REPO="${REPO:-/home1/irteam/funcbind}"
: "${BASE_NAME:?set BASE_NAME (the funcbind run to bring back)}"
CONFIG="${CONFIG:-train_fb_mcpp_holo_density_default}"
WAIT_PATTERN="${WAIT_PATTERN:?set WAIT_PATTERN, a pgrep -f pattern for the job to wait on}"
ROOT_SRC="${ROOT_SRC:-/nonexistent/no-root-src}"
BATCH_SIZE="${BATCH_SIZE:-5}"
ACCUM_STEPS="${ACCUM_STEPS:-38}"
NUM_WORKERS="${NUM_WORKERS:-6}"
GPUS="${GPUS:-0,1,2,3}"
POLL="${POLL:-120}"
TIMEOUT="${TIMEOUT:-172800}"        # 48 h
FREE_GIB="${FREE_GIB:-100}"         # a rank needs ~138 GiB; wait until the cards are really clear
LOG="${LOG:-$REPO/exps/funcbind/chain_resume_after_gpu_job_${BASE_NAME}.log}"

say() { echo "[$(date --iso-8601=seconds)] [chain-gpu] $*" | tee -a "$LOG"; }

say "waiting for: $WAIT_PATTERN"
waited=0
while pgrep -f "$WAIT_PATTERN" >/dev/null 2>&1; do
    if (( waited >= TIMEOUT )); then say "TIMEOUT after ${waited}s — the job is still running; NOT resuming"; exit 2; fi
    sleep "$POLL"; waited=$((waited + POLL))
done
say "the job is gone after ${waited}s"

# A process that has exited its Python frame can still hold GPU memory while CUDA tears
# down, and the trainer needs almost the whole card on every rank.
say "waiting for the GPUs to come back (>= ${FREE_GIB} GiB free on every card)"
for _ in $(seq 1 120); do
    min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    (( min_free / 1024 >= FREE_GIB )) && break
    sleep 15
done
say "min free across cards: $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1) / 1024 )) GiB"

cd "$REPO" || exit 1
if pgrep -f "[t]rain_fb\.py --config-name $CONFIG" >/dev/null; then
    say "a trainer for $CONFIG is already up; leaving it alone"
elif pgrep -f "[b]ash scripts/watchdog_funcbind_receptor_ed.sh" >/dev/null; then
    say "a watchdog is already running; leaving it alone"
else
    say "relaunching the watchdog for $BASE_NAME (config=$CONFIG)"
    BASE_NAME="$BASE_NAME" CONFIG="$CONFIG" ROOT_SRC="$ROOT_SRC" \
    BATCH_SIZE="$BATCH_SIZE" ACCUM_STEPS="$ACCUM_STEPS" NUM_WORKERS="$NUM_WORKERS" GPUS="$GPUS" \
        setsid nohup bash scripts/watchdog_funcbind_receptor_ed.sh \
        >>"$REPO/exps/funcbind/watchdog_launch_${BASE_NAME}.out" 2>&1 &
    sleep 45
fi

if pgrep -f "[b]ash scripts/crashguard_funcbind_receptor_ed.sh" >/dev/null; then
    say "crashguard already running"
else
    say "relaunching the crashguard"
    BASE_NAME="$BASE_NAME" CONFIG="$CONFIG" \
        setsid nohup bash scripts/crashguard_funcbind_receptor_ed.sh >/dev/null 2>&1 &
fi

sleep 60
say "trainer ranks now: $(pgrep -cf "[t]rain_fb\.py --config-name $CONFIG")"
say "done"
