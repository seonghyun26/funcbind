#!/usr/bin/env bash
# FuncBind MCP training conditioned on the receptor's deposited 2Fo-Fc electron
# density, fused zero-init — the recipe VoxBind's receptor-ED arm uses:
#
#     receptor_latent <- receptor_latent + zero_conv(frozen_CDG_encoder(rho))
#
#   * density source : funcbind/dataset/data/mcpp_holo_xray_v1 — one deposited
#                      receptor-level holo map per MCP target (557/643 available),
#                      cropped and augmented in lockstep with the conformer's atoms.
#   * encoder        : frozen VoxBind 13-channel ChannelViT [7 zero ligand ch |
#                      4 receptor atom blobs | rho | ||grad rho||]. The ligand channels
#                      stay zero: the molecule is the generation target.
#   * fusion         : ControlNet zero convolution, so step 0 is exactly the
#                      density-free baseline and unavailable maps stay an exact no-op.
#   * start point    : exps/funcbind/fb_unified (density-free MCP FuncBind), fresh
#                      optimizer.
#
# Usage:
#   scripts/train_funcbind_receptor_ed_zeroinit.sh              # wait for idle GPUs, then run
#   GPUS=0,1 EXP_NAME=my_run scripts/train_funcbind_receptor_ed_zeroinit.sh
#   SKIP_WAIT=1 scripts/train_funcbind_receptor_ed_zeroinit.sh  # run now
set -uo pipefail

REPO="${REPO:-${FUNCBIND_ROOT:-/home1/irteam/funcbind}}"
PY="${PY:-$REPO/.repro-env/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-train_fb_mcpp_holo_density}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
EXP_NAME="${EXP_NAME:-${RUN_DATE}_fb_mcpp_champion_receptor_ed_zeroinit}"
BATCH_SIZE="${BATCH_SIZE:-7}"
NUM_WORKERS="${NUM_WORKERS:-6}"
POLL_SECONDS="${POLL_SECONDS:-60}"
# The Vina docking evaluations on this box run ~20 single-core workers against a
# 32-core container quota, so the loaders are deliberately modest.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# The POSIX user here is not a wandb entity; leaving this unset makes wandb fall
# back to the API key's own default entity.
export WANDB_ENTITY="${WANDB_ENTITY:-eddy26}"
export WANDB_PROJECT="${WANDB_PROJECT:-voxbind}"
# Whether to compile follows the CONFIG, not this launcher -- see scripts/lib/dynamo_env.sh.
# This used to be a hardcoded =1 on the belief that the container has no C compiler. It has
# none, but the venv ships its own toolchain, inductor builds fine with it, and compiling is
# 1.20x faster here (13.66 -> 16.42 samples/s, measured). Hardcoding also made a run train
# differently before and after a watchdog resume.
source "$REPO/scripts/lib/dynamo_env.sh"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"

OUT="$REPO/exps/funcbind/$EXP_NAME"
LOG="${LOG:-$OUT/run.log}"
mkdir -p "$OUT"

say() { echo "[$(date --iso-8601=seconds)] [funcbind-receptor-ed] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1

if pgrep -af "train_fb.py --config-name $CONFIG" >/dev/null 2>&1; then
    say "a matching FuncBind density run already exists; refusing a duplicate launch"
    exit 3
fi

# exp_name must be pinned here: the config default embeds ${now:...}, which every
# Fabric worker re-resolves, scattering one run across four timestamped directories.
run_args=(
    "$PY" "$REPO/funcbind/train_fb.py"
    --config-name "$CONFIG"
    exp_name="$EXP_NAME"
    dset.batch_size="$BATCH_SIZE"
    dset.num_workers="$NUM_WORKERS"
)

say "exp=$OUT gpus=$GPUS bsz=$BATCH_SIZE workers=$NUM_WORKERS config=$CONFIG"

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
exit "$rc"
