#!/usr/bin/env bash
# Second half of the checkpoint smoke test: prove the file written by
# smoke_test_checkpoint_val.sh can actually be read back by the resume path.
#
# A checkpoint that serializes cleanly but will not reload is still a lost run,
# and load_funcbind reads keys (optimizer, code_stats, global_step) that the
# save path has to have written for the resume to continue rather than restart.
#
# Run scripts/smoke_test_checkpoint_val.sh first, then:
#   scripts/smoke_test_resume.sh
set -uo pipefail

REPO="${REPO:-${FUNCBIND_ROOT:-/home1/irteam/funcbind}}"
PY="${PY:-$REPO/.repro-env/bin/python}"
GPUS="${GPUS:-0}"
CONFIG="${CONFIG:-train_fb_mcpp_holo_density}"
SRC_NAME="${SRC_NAME:-smoke_ckpt_val}"
EXP_NAME="${EXP_NAME:-smoke_ckpt_resume}"
EXP_DIR="${EXP_DIR:-exps/smoke}"
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

SRC="$REPO/$EXP_DIR/$SRC_NAME"
OUT="$REPO/$EXP_DIR/$EXP_NAME"
LOG="$OUT/resume.log"

if [ ! -f "$SRC/checkpoint.pth.tar" ]; then
    echo "[resume] no checkpoint at $SRC; run scripts/smoke_test_checkpoint_val.sh first"
    exit 2
fi

rm -rf "$OUT"
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "[resume] resuming from $SRC into $OUT" | tee -a "$LOG"

# The model geometry must match what the smoke run wrote, or the state_dict load
# fails on shape rather than on anything to do with the checkpoint itself.
"$PY" "$REPO/funcbind/train_fb.py" \
    --config-name "$CONFIG" \
    exp_dir="$EXP_DIR" \
    exp_name="$EXP_NAME" \
    debug=true \
    wandb=false \
    n_samples="$N_SAMPLES" \
    num_epochs=1 \
    val_every=1 \
    checkpoint_every=1 \
    sample_every=100000 \
    accum_steps="$ACCUM_STEPS" \
    log_every_steps=1 \
    fb_pretrained_path="$SRC" \
    resume_optimizer=true \
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

echo "[resume] training exited with code $rc" | tee -a "$LOG"
if [ "$rc" -ne 0 ]; then
    echo "[resume] FAILED; last 40 lines:" | tee -a "$LOG"
    tail -40 "$LOG"
    exit "$rc"
fi

echo "[resume] checking the resume actually continued rather than restarted" | tee -a "$LOG"
grep -E "loading checkpoint from|loading optimizer state" "$LOG" | tee -a "$LOG"

# global_step must pick up past the source run's final step, not restart at 1.
src_steps=$("$PY" - "$SRC/checkpoint.pth.tar" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="meta", weights_only=False, mmap=True)
print(int(ck["global_step"]))
EOF
)
first_step=$(grep -oE "global_step=[0-9]+" "$LOG" | head -1 | cut -d= -f2)
echo "[resume] source checkpoint global_step=$src_steps ; first step after resume=$first_step" | tee -a "$LOG"
if [ -n "$first_step" ] && [ "$first_step" -gt "$src_steps" ]; then
    echo "[resume] PASS  global_step continued from the checkpoint" | tee -a "$LOG"
else
    echo "[resume] FAIL  global_step restarted (expected > $src_steps, got $first_step)" | tee -a "$LOG"
    exit 1
fi

"$PY" "$REPO/scripts/verify_checkpoint_val.py" "$OUT" | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
