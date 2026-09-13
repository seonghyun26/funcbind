#!/usr/bin/env bash
# Sample macrocyclic peptides (MCP) from the pretrained FuncBind checkpoint.
#
# The run is detached with setsid so it survives the launching shell (the
# Jul-29 paper_run attempt died within a minute of launch with an empty log,
# which is what an un-detached child does when its parent goes away).
#
#   smoke test : NAME=smoke GPU=2 IDS=[0] NTARGETS=1 NPR=8 NCHAINS=64 NATT=1 \
#                  scripts/run_mcpp_sampling.sh
#   paper chunk: NAME=paper_run/gpu0 GPU=0 IDS=[0,1,...,24] NTARGETS=25 \
#                  scripts/run_mcpp_sampling.sh
#
# Writes artifacts/reproduction/mcpp/$NAME/{run.log,exit_code} and the samples
# under .../$NAME/samples/target_XX/.
set -uo pipefail

REPO=/home1/irteam/funcbind
PY="$REPO/.repro-env/bin/python"

: "${NAME:?set NAME (run label under artifacts/reproduction/mcpp/)}"
: "${IDS:?set IDS, e.g. [0] or [0,1,2] (no spaces)}"
: "${NTARGETS:?set NTARGETS (must match the length of IDS)}"
GPU=${GPU:-0}
NPR=${NPR:-100}       # sampling.n_samples_per_receptor
NCHAINS=${NCHAINS:-756}
NATT=${NATT:-10}
SEED=${SEED:-1}
# Rendering chunk sizes. Both are pure batching knobs (decoder.py splits along
# dim 0 / dim 1 and concatenates), so they change speed and memory, not results.
# batch_size_render_codes costs ~1 GiB of GPU memory per code: the b200 preset of
# 110 wants ~110 GiB, which does not fit an H200 that is shared with another job.
BSRC=${BSRC:-16}      # sampling.batch_size_render_codes
BSR=${BSR:-1000}      # sampling.batch_size_render
# Which model to sample. Default = the density-free MCP FuncBind ("vanilla").
# The receptor-ED ControlNet fine-tune needs BOTH overrides together: its
# architecture comes from the checkpoint's own config, but the sampling config
# supplies the holo-density dataset wiring (mcpp_holo_density_dir, voxbind root).
#   CONFIG=sample_fb_mcpp_holo_density \
#   FB_PATH=$REPO/exps/funcbind/20260816_..._resumed_bsz6_ga32_r3
CONFIG=${CONFIG:-sample_fb_mcpp}
FB_PATH=${FB_PATH:-$REPO/exps/funcbind/fb_unified}

OUT="$REPO/artifacts/reproduction/mcpp/$NAME"
export NAME IDS NTARGETS GPU NPR NCHAINS NATT SEED BSRC BSR CONFIG FB_PATH

if [[ ${MCPP_RUNNER:-0} == 1 ]]; then
    cd "$REPO/funcbind" || exit 1
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" sample_fb.py \
        --config-name "$CONFIG" \
        wandb=false \
        +n_devs=1 \
        +nf_pretrained_path="$REPO/exps/neural_field/nf_unified" \
        fb_pretrained_path="$FB_PATH" \
        dirname="$OUT" \
        exp_name="mcpp_${NAME//\//_}" \
        seed="$SEED" \
        gpu_type=b200 \
        sampling.n_targets="$NTARGETS" \
        sampling.n_samples_per_receptor="$NPR" \
        sampling.n_chains="$NCHAINS" \
        sampling.n_attempts="$NATT" \
        sampling.receptor_ids="$IDS" \
        sampling.batch_size_render_codes="$BSRC" \
        sampling.batch_size_render="$BSR" \
        >"$OUT/run.log" 2>&1
    echo $? >"$OUT/exit_code"
    exit 0
fi

mkdir -p "$OUT"
rm -f "$OUT/exit_code"
MCPP_RUNNER=1 setsid nohup "$0" </dev/null >"$OUT/launch.log" 2>&1 &
echo "launched pid $! : gpu=$GPU targets=$NTARGETS ids=$IDS npr=$NPR chains=$NCHAINS attempts=$NATT"
echo "log: $OUT/run.log"
