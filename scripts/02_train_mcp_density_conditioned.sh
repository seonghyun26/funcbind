#!/usr/bin/env bash
# Step 2 of the MCP density-conditioning recipe: the conditioned fine-tune.
#
#   bash scripts/02_train_mcp_density_conditioned.sh
#   GPUS=0,1,2,3,4,5,6,7 BATCH_SIZE=1 bash scripts/02_train_mcp_density_conditioned.sh
#   FORCE=1 ... # launch even if the preflight says the memory does not fit
#
# The H100 config preserves bf16-mixed, batch 1, eager execution, and
# non-foreach AdamW, ZeRO-1, CPU EMA, and block activation checkpointing.
# Static state is ~43.1 GiB/rank on 8 GPUs; full-model peaks still need validation.
#
# WHAT THIS TRAINS
#   FuncBind's MCP denoiser, fine-tuned from the density-free base (exps/funcbind/fb_unified)
#   with the receptor's deposited 2Fo-Fc density as an extra condition:
#
#       receptor_latent <- receptor_latent + zero_conv(proj([receptor + ligand, encoder(rho)]))
#
#   * encoder : FROZEN 13-channel CDG v2 epoch 25 (dim 640 / depth 18 /
#               groups [7,4,2]). 7 ligand channels are zero -- the molecule is the target.
#   * fusion  : `default`, i.e. the projection is conditioned on the receptor AND the noisy
#               ligand. This is the variant that beat the density-free baseline on the
#               78-pocket CrossDocked benchmark (-8.45 vs -8.05). The earlier MCP arm
#               conditioned on density alone and never closed its gap over 26.1M samples.
#   * init    : ControlNet zero convolution, so step 0 is EXACTLY the density-free model and
#               a target with no map stays an exact no-op.
#
# EFFECTIVE BATCH is what the LR schedule is written against (decay onset =
# ref_batches x effective_batch, measured in acc_iter), so it is held fixed at 760 and the
# accumulation is derived from the batch size and GPU count rather than set by hand. Change
# BATCH_SIZE freely; change EFFECTIVE_BATCH only if you mean to change the experiment.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$REPO/.repro-env/bin/python}"
CONFIG="${CONFIG:-train_fb_mcpp_holo_density_h100}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$BATCH_SIZE}"
NUM_WORKERS="${NUM_WORKERS:-6}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-760}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
EXP_NAME="${EXP_NAME:-${RUN_DATE}_fb_mcpp_default_cdg_v2_e0025_zeroinit}"
N_SAMPLES="${N_SAMPLES:-}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
WANDB_ENABLED="${WANDB_ENABLED:-}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-}"

export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-funcbind}"
export FUNCBIND_ROOT="${FUNCBIND_ROOT:-$REPO}"
export VOXBIND_PYTHON_ROOT="${VOXBIND_PYTHON_ROOT:-$(dirname "$REPO")}"
export FUNCBIND_DENSITY_ENCODER="${FUNCBIND_DENSITY_ENCODER:-$VOXBIND_PYTHON_ROOT/voxbind/exps/260806_cdg_100m_v2_ep100/checkpoint_e0025.pth.tar}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Whether to compile follows the CONFIG -- see scripts/lib/dynamo_env.sh. On the H200 box
# compiling is 1.20x faster (13.66 -> 16.42 samples/s), and inductor builds against the
# venv's own conda toolchain even though the container ships no cc. On a box without either,
# set TORCHDYNAMO_DISABLE=1 in the environment and this defers to it.
source "$REPO/scripts/lib/dynamo_env.sh"
# Loader threads are deliberately modest: oversubscribing the CPU quota starves the loaders
# and the GPUs idle (measured: 14.2 -> 6.6 samples/s when a 52-core job shared 32 cores).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

say() { echo "[$(date --iso-8601=seconds)] [02-train] $*"; }
die() { say "ABORT: $*"; exit 1; }

N_GPUS=$(awk -F, '{print NF}' <<<"$GPUS")
ACCUM=$(( EFFECTIVE_BATCH / (BATCH_SIZE * N_GPUS) ))
[ "$ACCUM" -ge 1 ] || die "batch_size x gpus ($((BATCH_SIZE * N_GPUS))) exceeds the effective batch ($EFFECTIVE_BATCH)"
REAL=$(( BATCH_SIZE * N_GPUS * ACCUM ))
say "gpus=$N_GPUS bsz=$BATCH_SIZE accum=$ACCUM -> effective $REAL (target $EFFECTIVE_BATCH)"
[ "$REAL" = "$EFFECTIVE_BATCH" ] || say "NOTE: not an exact divisor; effective batch is $REAL"

say "preflight"
if ! "$PY" "$REPO/scripts/preflight_mcp_density.py" \
    --config "$CONFIG" --expected-gpus "$N_GPUS" --batch-size "$BATCH_SIZE"; then
    [ -n "${FORCE:-}" ] || die "preflight failed (set FORCE=1 to launch anyway, e.g. if you have
     already measured this recipe's full-model peak on the target hardware)"
    say "preflight failed but FORCE=1 was set — continuing"
fi

OUT="$REPO/exps/funcbind/$EXP_NAME"
LOG="$OUT/run.log"
train_overrides=(
    "exp_name=$EXP_NAME"
    "dset.batch_size=$BATCH_SIZE"
    "dset.val_batch_size=$VAL_BATCH_SIZE"
    "dset.num_workers=$NUM_WORKERS"
    "accum_steps=$ACCUM"
)
[ -n "$N_SAMPLES" ] && train_overrides+=("n_samples=$N_SAMPLES")
[ -n "$NUM_EPOCHS" ] && train_overrides+=("num_epochs=$NUM_EPOCHS")
[ -n "$WANDB_ENABLED" ] && train_overrides+=("wandb=$WANDB_ENABLED")
[ -n "$SAVE_CHECKPOINTS" ] && train_overrides+=("save_checkpoints=$SAVE_CHECKPOINTS")

if [ -n "${DRY_RUN:-}" ]; then
    say "DRY_RUN set — everything checked, launching nothing"
    printf '[%s] [02-train] would run: train_fb.py --config-name %q' \
        "$(date --iso-8601=seconds)" "$CONFIG"
    printf ' %q' "${train_overrides[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$OUT" || die "cannot create output directory: $OUT"
cd "$REPO" || die "cannot enter repository: $REPO"

# exp_name must be pinned on the command line: the config default embeds ${now:...}, which
# every Fabric worker re-resolves, scattering one run across N timestamped directories.
if pgrep -af "train_fb.py --config-name $CONFIG" >/dev/null 2>&1; then
    die "a run with this config is already alive; refusing a duplicate launch"
fi

say "exp=$OUT config=$CONFIG"
"$PY" "$REPO/funcbind/train_fb.py" \
    --config-name "$CONFIG" \
    "${train_overrides[@]}" \
    >>"$LOG" 2>&1
rc=$?
say "training exited with code $rc  (log: $LOG)"
exit "$rc"
