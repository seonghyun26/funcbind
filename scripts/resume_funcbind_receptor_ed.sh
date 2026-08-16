#!/usr/bin/env bash
# Resume the receptor-ED FuncBind run from its own checkpoint instead of
# restarting the 5.24B-parameter model from the density-free baseline.
#
# The 2026-08-15 run died at epoch 5 / optimizer step 62 when the container was
# restarted (PID 1 postdates the last log line by ~80 s) -- not from anything in
# the model. Its epoch-4 checkpoint is intact and verified, so the training can
# pick up from there rather than repeat ~17 h of work.
#
#   source     : exps/funcbind/20260815_..._val1_ckpt1/checkpoint.pth.tar
#                epoch 5, global_step 1235, acc_iter 932540, best_res 3831954.39
#   resumes    : weights + EMA + AdamW moments + LR-schedule position + best_res
#   writes to  : a NEW exp dir, so the source checkpoint stays a working fallback
#
# This differs from a cold start in one line of config: fb_pretrained_path points
# at the run's own output instead of fb_unified, and resume_optimizer is true so
# the optimizer state and step counter carry over rather than resetting.
#
# Usage:
#   scripts/resume_funcbind_receptor_ed.sh                 # wait for idle GPUs, then resume
#   SKIP_WAIT=1 scripts/resume_funcbind_receptor_ed.sh     # resume now
#   SRC=<dir> EXP_NAME=<name> scripts/resume_funcbind_receptor_ed.sh
#   FRESH_OPTIMIZER=1 scripts/resume_funcbind_receptor_ed.sh   # keep weights, reset optimizer
set -uo pipefail

REPO="${REPO:-/home1/irteam/funcbind}"
PY="${PY:-$REPO/.repro-env/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-train_fb_mcpp_holo_density}"

# Source checkpoint: the crashed run's own output.
SRC="${SRC:-$REPO/exps/funcbind/20260815_fb_mcpp_champion_receptor_ed_zeroinit_ga27_pf4_ooo_val1_ckpt1}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
EXP_NAME="${EXP_NAME:-${RUN_DATE}_fb_mcpp_champion_receptor_ed_zeroinit_resumed}"

# Must match the source run, or the state_dict will not load onto the model.
BATCH_SIZE="${BATCH_SIZE:-7}"
NUM_WORKERS="${NUM_WORKERS:-6}"
ACCUM_STEPS="${ACCUM_STEPS:-27}"
POLL_SECONDS="${POLL_SECONDS:-60}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export WANDB_ENTITY="${WANDB_ENTITY:-eddy26}"
export WANDB_PROJECT="${WANDB_PROJECT:-voxbind}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"

OUT="$REPO/exps/funcbind/$EXP_NAME"
LOG="${LOG:-$OUT/run.log}"
mkdir -p "$OUT"

say() { echo "[$(date --iso-8601=seconds)] [funcbind-resume] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1

if [ ! -f "$SRC/checkpoint.pth.tar" ]; then
    say "no checkpoint at $SRC/checkpoint.pth.tar -- nothing to resume from"
    exit 2
fi

if pgrep -af "train_fb.py --config-name $CONFIG" >/dev/null 2>&1; then
    say "a matching FuncBind density run already exists; refusing a duplicate launch"
    exit 3
fi

# Fail before burning a GPU reservation if the archive is damaged: a truncated
# checkpoint only surfaces deep inside fabric.load, minutes into startup.
say "verifying source checkpoint"
if ! "$PY" "$REPO/scripts/verify_checkpoint_val.py" "$SRC" >>"$LOG" 2>&1; then
    say "source checkpoint failed verification; see $LOG"
    exit 4
fi

resume_args=(resume_optimizer=true)
if [ -n "${FRESH_OPTIMIZER:-}" ]; then
    resume_args=(resume_optimizer=false)
    say "FRESH_OPTIMIZER set: loading weights only, optimizer/step/best_res reset"
fi

run_args=(
    "$PY" "$REPO/funcbind/train_fb.py"
    --config-name "$CONFIG"
    exp_name="$EXP_NAME"
    fb_pretrained_path="$SRC"
    "${resume_args[@]}"
    dset.batch_size="$BATCH_SIZE"
    dset.num_workers="$NUM_WORKERS"
    accum_steps="$ACCUM_STEPS"
)

say "resuming from $SRC"
say "exp=$OUT gpus=$GPUS bsz=$BATCH_SIZE accum=$ACCUM_STEPS workers=$NUM_WORKERS"
# Each of the 4 ranks deserializes the full 77 GiB archive, ~309 GiB against the
# 768 GiB cgroup cap. That is the peak host-RAM moment of the whole run.
say "cgroup: $("$PY" -c "
m=int(open('/sys/fs/cgroup/memory.max').read()); c=int(open('/sys/fs/cgroup/memory.current').read())
print(f'{c/1024**3:.1f} GiB used of {m/1024**3:.0f} GiB cap')")"

if [ -n "${SKIP_WAIT:-}" ]; then
    "${run_args[@]}" >>"$LOG" 2>&1
else
    "$REPO/scripts/wait_for_gpus.sh" \
        --gpus "$GPUS" \
        --poll-seconds "$POLL_SECONDS" \
        --stable-checks 3 \
        --timeout-seconds 0 \
        -- "${run_args[@]}" >>"$LOG" 2>&1
fi
rc=$?

say "training exited with code $rc"
if [ "$rc" -ne 0 ]; then
    say "last 30 lines:"
    tail -30 "$LOG"
fi
exit "$rc"
