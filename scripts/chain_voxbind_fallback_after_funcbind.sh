#!/usr/bin/env bash
# Fall back to the VoxBind from-scratch zero-init fusion recipe if the funcbind
# receptor-ED run fails.
#
# WHAT COUNTS AS FAILURE
#   1. the watchdog's own verdict: .watchdog_gave_up_<BASE_NAME>, written after
#      MAX_NO_PROGRESS cycles that advanced no checkpoint. This is the designed
#      signal and the one to trust.
#   2. the watchdog process is gone AND no trainer is running, sustained for
#      GONE_CHECKS polls. This exists because on 2026-08-18 a relaunch lost the
#      flock race, printed "refusing to start a second", and left NOTHING
#      supervising -- the run was down two days before anyone noticed. A silent
#      disappearance is a failure even though no marker is ever written.
#
# WHAT IS NOT FAILURE
#   funcbind sitting in wait_for_gpus while another job holds the GPUs, and a
#   clean finish ("trainer exited cleanly" in the watchdog log), which would
#   otherwise look identical to case 2.
#
# The fallback is scripts/67_train_fusion_champion_reference_4gpu.sh driven with
# the argument set 68_train_champion_receptor_ed_after_sampling.sh uses for its
# primary arm -- WARM_START="" (from scratch; the empty string must override the
# config default or it silently warm-starts), receptor-ED crops, sigma 0.9 -- but
# with FUSION=v4 instead of default (user, 2026-08-26).
#
# v4 merges the frozen encoder's TOKENS rather than its unpatchified voxel output,
# so it skips _pool_groups (which mean-pools the channel groups) and decoder_proj
# (a frozen Linear trained as an intermediate of the MAE recon head). It is still
# a zero-init residual, so step 0 is the density-free model exactly as before.
# v4 has never run on a GPU, so this chain smoke-tests it at the real batch size
# and refuses to launch if that fails -- see test/v4_gpu_smoke.py.
set -uo pipefail

FB_REPO="${FUNCBIND_ROOT:-${FUNCBIND_ROOT:-/home1/irteam/funcbind}}"
VB_ROOT="${VOXBIND_PYTHON_ROOT:-${VOXBIND_PYTHON_ROOT:-/home1/irteam/VoxBind}}/voxbind"
BASE_NAME="${BASE_NAME:-20260816_fb_mcpp_champion_receptor_ed_zeroinit_resumed_bsz6_ga32}"
GAVE_UP="$FB_REPO/exps/funcbind/.watchdog_gave_up_${BASE_NAME}"
WD_LOG="$FB_REPO/exps/funcbind/watchdog_${BASE_NAME}.log"

EXP_NAME="${EXP_NAME:-voxbind_fusion_v4_newenc_receptor_ed_sig0.9}"
ZOO="${ZOO:-${VOXBIND_PYTHON_ROOT:-/home1/irteam/VoxBind}/voxbind/model_zoo}"
# Newest encoder checkpoint present on 2026-08-23 (efficient_60m, 2026-07-29 13:35).
# Anything newer is the encoder the user said they would upload; the dropbox zoo
# watcher pulls it within its poll interval.
ENCODER_BASELINE_TS="${ENCODER_BASELINE_TS:-1785299709}"
ENCODER_WAIT="${ENCODER_WAIT:-43200}"      # 12 h
# Density-residual dropout, new for this run. The frozen encoder is held in eval()
# with train() overridden, so density_vit.dropout never fires and the density path
# had no regularization at all -- frozen_v3 showed train loss falling 628->529 while
# val sat at 412-423 for its last hundred epochs. 0.1 matches the cfg_dropout the
# funcbind denoiser already uses for its class/receptor condition.
COND_DROPOUT="${COND_DROPOUT:-0.1}"
CROPS_DIR="${CROPS_DIR:-$VB_ROOT/dataset/data/xray_crops_receptor_ed_v5}"
NUM_EPOCHS="${NUM_EPOCHS:-350}"
SMOKE_BSZ="${SMOKE_BSZ:-32}"        # must match bsz= in 67_train_fusion_champion_reference_4gpu.sh
POLL="${POLL:-300}"
GONE_CHECKS="${GONE_CHECKS:-6}"        # 6 x 300 s = 30 min of nothing running
LOG="${LOG:-$FB_REPO/exps/funcbind/chain_voxbind_fallback.log}"

say() { echo "[$(date --iso-8601=seconds)] [chain] $*" | tee -a "$LOG"; }

watchdog_up() { pgrep -f "[b]ash scripts/watchdog_funcbind_receptor_ed.sh" >/dev/null 2>&1; }
trainer_up()  { pgrep -f "[t]rain_fb\.py --config-name train_fb_mcpp_holo_density" >/dev/null 2>&1; }
gpus_busy()   { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }

say "armed: watching $BASE_NAME; fallback exp=$EXP_NAME"

gone=0
while true; do
    if [ -f "$GAVE_UP" ]; then
        say "FAILURE: watchdog gave up -- $(tr '\n' ' ' <"$GAVE_UP")"
        break
    fi

    if watchdog_up || trainer_up; then
        gone=0
    else
        if tail -5 "$WD_LOG" 2>/dev/null | grep -q "trainer exited cleanly"; then
            say "funcbind finished cleanly; nothing to fall back to"
            exit 0
        fi
        gone=$((gone + 1))
        say "no watchdog and no trainer ($gone/$GONE_CHECKS)"
        [ "$gone" -ge "$GONE_CHECKS" ] && { say "FAILURE: funcbind vanished with no clean-exit record"; break; }
    fi
    sleep "$POLL"
done

# A dying watchdog would otherwise resume funcbind onto the GPUs this run is about
# to claim, and the two would thrash each other.
if watchdog_up; then
    say "stopping the funcbind watchdog before claiming the GPUs"
    pkill -TERM -f "[b]ash scripts/watchdog_funcbind_receptor_ed.sh"
    sleep 10
    pkill -KILL -f "[b]ash scripts/watchdog_funcbind_receptor_ed.sh" 2>/dev/null
fi
if trainer_up; then
    say "stopping leftover funcbind ranks"
    pkill -TERM -f "[t]rain_fb\.py --config-name train_fb_mcpp_holo_density"
    sleep 60
    pkill -KILL -f "[t]rain_fb\.py --config-name train_fb_mcpp_holo_density" 2>/dev/null
fi

# funcbind failing says nothing about whether the GPUs are free: another job (the
# frozen_v3 VoxBind run) may still hold them.
while gpus_busy; do
    say "GPUs still busy; waiting"
    sleep "$POLL"
done

VB_PY="${VB_PY:-/opt/conda/envs/voxbind/bin/python}"
TRAIN_AVAILABLE=$("$VB_PY" -c \
    "import json;print(json.load(open('$CROPS_DIR/.complete'))['train']['available'])" 2>/dev/null) \
  || TRAIN_AVAILABLE=""
[ -n "$TRAIN_AVAILABLE" ] || { say "ABORT: cannot read $CROPS_DIR/.complete"; exit 1; }
SUBSET_N=$((TRAIN_AVAILABLE - 100))

# The run is defined by the encoder, so it waits for one rather than quietly
# substituting the old champion -- that would be a different experiment wearing this
# experiment's name. A folder counts only if it holds checkpoint_e*.pth.tar, which is
# what separates an encoder from a plain VoxBind checkpoint like
# voxbind_sig0.9_crossdocked. -L resolves the champion/coords symlinks.
newest_encoder() {
    find -L "$ZOO" -maxdepth 2 -name "checkpoint_e*.pth.tar" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1
}

say "waiting for a new encoder in $ZOO (newer than $(date -d @$ENCODER_BASELINE_TS '+%Y-%m-%d %H:%M'))"
enc_deadline=$(( $(date +%s) + ENCODER_WAIT ))
ENC=""
while :; do
    line="$(newest_encoder)"
    ts="${line%% *}"; ts="${ts%%.*}"; cand="${line#* }"
    if [ -n "$ts" ] && [ "$ts" -gt "$ENCODER_BASELINE_TS" ]; then
        ENC="$cand"
        say "new encoder: $ENC ($(date -d @$ts '+%Y-%m-%d %H:%M'))"
        break
    fi
    if [ "$(date +%s)" -ge "$enc_deadline" ]; then
        say "PROBLEM: no new encoder within ${ENCODER_WAIT}s -- NOT launching, because"
        say "         falling back to the old champion would silently run a different experiment"
        exit 1
    fi
    sleep "$POLL"
done

# Geometry comes from the encoder's own cfg.yaml: model_zoo entries are not
# interchangeable at fixed dims (champion 640/10 vs efficient_60m 512/8).
ENC_DIR="$(dirname "$ENC")"
read -r VIT_PATCH VIT_DIM VIT_DEPTH VIT_HEADS VIT_MLP VIT_DROP VIT_NCH VIT_GROUPS <<<"$(
    "$VB_PY" - "$ENC_DIR/cfg.yaml" <<'PY' 2>/dev/null
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
m = c.get("model", c)
g = m.get("channel_groups") or [7, 4, 2]
print(m.get("patch_size", 8), m.get("dim", 640), m.get("depth", 18), m.get("heads", 10),
      m.get("mlp_ratio", 4), m.get("dropout", 0.1), m.get("n_in_channels", 13),
      "[" + ",".join(str(int(x)) for x in g) + "]")
PY
)"
[ -n "${VIT_DIM:-}" ] || { say "ABORT: cannot read encoder geometry from $ENC_DIR/cfg.yaml"; exit 1; }
say "encoder geometry: patch=$VIT_PATCH dim=$VIT_DIM depth=$VIT_DEPTH heads=$VIT_HEADS groups=$VIT_GROUPS n_in=$VIT_NCH"

# One fwd+bwd at the real per-rank batch size, with this exact encoder loaded
# strict. Catches a geometry mismatch, a CUDA-only shape bug, and the extra ~3 GiB
# of 64^3 activations v4 adds -- all of which would otherwise show up as a dead
# 350-epoch launch hours from now. The GPUs are already free at this point.
say "v4 GPU smoke (bsz=$SMOKE_BSZ) with $ENC"
if ! ( cd "$VB_ROOT" && "$VB_PY" test/v4_gpu_smoke.py --encoder "$ENC" --bsz "$SMOKE_BSZ" ) >>"$LOG" 2>&1; then
    say "ABORT: v4 GPU smoke failed -- not launching; see $LOG"
    exit 1
fi
say "v4 GPU smoke passed"

say "launching $EXP_NAME (from scratch, v4 token fusion, sigma=0.9, ${NUM_EPOCHS}ep, subset=$SUBSET_N+100, cond_dropout=$COND_DROPOUT)"
cd "$VB_ROOT" || exit 1
WARM_START="" EXP_NAME="$EXP_NAME" NUM_EPOCHS="$NUM_EPOCHS" SIGMA=0.9 \
SEES_LIGAND=true MASK_LIGAND=false FUSION=v4 \
COND_DROPOUT="$COND_DROPOUT" \
ENCODER="$ENC" \
VIT_PATCH="$VIT_PATCH" VIT_DIM="$VIT_DIM" VIT_DEPTH="$VIT_DEPTH" VIT_HEADS="$VIT_HEADS" \
VIT_MLP_RATIO="$VIT_MLP" VIT_DROPOUT="$VIT_DROP" VIT_NCH="$VIT_NCH" VIT_GROUPS="$VIT_GROUPS" \
CROPS_DIR="$CROPS_DIR" SUBSET_N="$SUBSET_N" SUBSET_VAL_N=100 \
WANDB_TAGS="[voxbind,fusion,v4,token_fusion,new_encoder,reference_ligand,receptor_ed,full_holo_ed,zero_init,sigma0.9,cond_dropout0.1]" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash scripts/67_train_fusion_champion_reference_4gpu.sh >>"$LOG" 2>&1
say "fallback training exited with code $?"
