#!/usr/bin/env bash
# Smoke test for the epoch-boundary checkpoint + validation path.
#
# Exercises exactly the code that changed in the receptor-ED run — the split of
# validation and checkpointing into two independent cadences, the atomic
# tmp->replace write, and the best-checkpoint hard link — but on a ~30M-parameter
# denoiser and a 16-sample split, so a full epoch costs seconds instead of hours
# and the checkpoint is megabytes instead of 77 GiB.
#
# Usage:
#   scripts/smoke_test_checkpoint_val.sh              # 3 epochs, GPU 0
#   EPOCHS=2 GPUS=1 scripts/smoke_test_checkpoint_val.sh
set -uo pipefail

REPO="${REPO:-${FUNCBIND_ROOT:-/home1/irteam/funcbind}}"
PY="${PY:-$REPO/.repro-env/bin/python}"
GPUS="${GPUS:-0}"
CONFIG="${CONFIG:-train_fb_mcpp_holo_density}"
EXP_NAME="${EXP_NAME:-smoke_ckpt_val}"
EXP_DIR="${EXP_DIR:-exps/smoke}"
EPOCHS="${EPOCHS:-3}"
N_SAMPLES="${N_SAMPLES:-16}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"

OUT="$REPO/$EXP_DIR/$EXP_NAME"
LOG="$OUT/smoke.log"
rm -rf "$OUT"
mkdir -p "$OUT"

cd "$REPO" || exit 1

echo "[smoke] exp=$OUT gpus=$GPUS epochs=$EPOCHS n_samples=$N_SAMPLES" | tee -a "$LOG"

# fb_pretrained_path=null forces a fresh small denoiser: loading the real 77 GiB
# checkpoint would both dwarf the runtime and fail on shape mismatch.
"$PY" "$REPO/funcbind/train_fb.py" \
    --config-name "$CONFIG" \
    exp_dir="$EXP_DIR" \
    exp_name="$EXP_NAME" \
    debug=true \
    wandb=false \
    n_samples="$N_SAMPLES" \
    num_epochs="$EPOCHS" \
    val_every=1 \
    checkpoint_every=1 \
    sample_every=100000 \
    accum_steps="$ACCUM_STEPS" \
    log_every_steps=1 \
    fb_pretrained_path=null \
    dset.batch_size="$BATCH_SIZE" \
    dset.val_batch_size=4 \
    dset.num_workers=2 \
    dset.persistent_workers=false \
    dset.prefetch_factor=2 \
    denoiser.model_channels=32 \
    denoiser.n_blocks=1 \
    'sampler.val_sigmas=[0.5,2,10]' \
    >>"$LOG" 2>&1
rc=$?

echo "[smoke] training exited with code $rc" | tee -a "$LOG"
if [ "$rc" -ne 0 ]; then
    echo "[smoke] FAILED during training; last 40 lines:" | tee -a "$LOG"
    tail -40 "$LOG"
    exit "$rc"
fi

"$PY" "$REPO/scripts/verify_checkpoint_val.py" "$OUT" --epochs "$EPOCHS" | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
