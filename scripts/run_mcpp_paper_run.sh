#!/usr/bin/env bash
# Full MCP paper reproduction: 100 test targets x 100 samples on 8 H100 GPUs.
#
#   scripts/run_mcpp_paper_run.sh            # all 8 chunks
#   CHUNKS=0,1 scripts/run_mcpp_paper_run.sh # only gpu0 and gpu1
#   DRY_RUN=1 scripts/run_mcpp_paper_run.sh  # print assignments only
#
# Each chunk is launched through run_mcpp_sampling.sh (detached, own run.log and
# exit_code) once its GPU has MINFREE MiB free. Waiting is bounded by MAXWAIT so
# this never hangs forever behind another job.
set -uo pipefail

REPO="${FUNCBIND_ROOT:-/home1/irteam/funcbind}"
HERE="$(cd "$(dirname "$0")" && pwd)"

MINFREE=${MINFREE:-70000}          # MiB of free GPU memory required per H100
MAXWAIT=${MAXWAIT:-7200}           # seconds to wait per GPU before giving up
STABLE=${STABLE:-3}                # consecutive OK checks before launching
INTERVAL=${INTERVAL:-60}
GPU_COUNT=${GPU_COUNT:-8}
TOTAL_TARGETS=${TOTAL_TARGETS:-100}
CHUNKS=${CHUNKS:-0,1,2,3,4,5,6,7}
DRY_RUN=${DRY_RUN:-}
RUN=${RUN:-paper_run}
NPR=${NPR:-100}
CONFIG=${CONFIG:-sample_fb_mcpp_holo_density}
FB_PATH=${FB_PATH:-$REPO/exps/funcbind/fb_mcpp_holo_density}
MCP_AUTO_FETCH_MODEL=${MCP_AUTO_FETCH_MODEL:-1}
export CONFIG FB_PATH MCP_AUTO_FETCH_MODEL

LOG="$REPO/artifacts/reproduction/mcpp/$RUN/launcher.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

if ! [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    say "GPU_COUNT must be a positive integer (got: $GPU_COUNT)"
    exit 2
fi
if ! [[ "$TOTAL_TARGETS" =~ ^[1-9][0-9]*$ ]]; then
    say "TOTAL_TARGETS must be a positive integer (got: $TOTAL_TARGETS)"
    exit 2
fi

base_count=$((TOTAL_TARGETS / GPU_COUNT))
remainder=$((TOTAL_TARGETS % GPU_COUNT))

say "MCP reproduction: $TOTAL_TARGETS targets on $GPU_COUNT H100 GPUs, chunks=$CHUNKS, minfree=${MINFREE}MiB, npr=$NPR"
say "config=$CONFIG, checkpoint=$FB_PATH"

if [ -z "$DRY_RUN" ]; then
    for required_file in \
        "$REPO/funcbind/dataset/data/mcpp_dataset/test_data.pt" \
        "$REPO/exps/neural_field/nf_unified/model.pt"; do
        if [ ! -f "$required_file" ]; then
            say "required input missing: $required_file"
            exit 2
        fi
    done

    if [ "$CONFIG" = sample_fb_mcpp_holo_density ]; then
        density_dir="$REPO/funcbind/dataset/data/mcpp_holo_xray_v1"
        if [ ! -d "$density_dir" ]; then
            say "MCP holo-density data missing: $density_dir"
            exit 2
        fi
        if [ ! -s "$FB_PATH/checkpoint.pth.tar" ]; then
            if [ "$MCP_AUTO_FETCH_MODEL" = 1 ]; then
                MCP_MODEL_DIR="$FB_PATH" "$HERE/pull_mcpp_density_model.sh"
            else
                say "density checkpoint missing: $FB_PATH/checkpoint.pth.tar"
                exit 2
            fi
        fi
    fi
fi

for g in ${CHUNKS//,/ }; do
    if ! [[ "$g" =~ ^[0-9]+$ ]] || (( g >= GPU_COUNT )); then
        say "invalid chunk '$g'; expected a GPU index from 0 to $((GPU_COUNT - 1))"
        exit 2
    fi

    count=$base_count
    if (( g < remainder )); then
        count=$((count + 1))
        lo=$((g * count))
    else
        lo=$((remainder * (base_count + 1) + (g - remainder) * base_count))
    fi
    if (( count == 0 )); then
        say "gpu$g has no assigned targets - SKIPPING"
        continue
    fi

    hi=$((lo + count - 1))
    ids="[$(seq -s, $lo $hi)]"

    if [ -n "$DRY_RUN" ]; then
        say "DRY_RUN gpu$g: targets $lo-$hi ($count)"
        continue
    fi

    ok=0; waited=0
    while (( ok < STABLE )); do
        # Free-memory polling alone is not enough to tell "the other job finished"
        # from "the other job is between work items": a VoxBind eval rank swings
        # between ~8 and ~55 GiB. WAIT_UNTIL_GONE gates on the process itself.
        if [ -n "${WAIT_UNTIL_GONE:-}" ] && pgrep -f "$WAIT_UNTIL_GONE" >/dev/null 2>&1; then
            say "waiting on gpu$g: '$WAIT_UNTIL_GONE' still running"
            ok=0; sleep "$INTERVAL"; waited=$((waited + INTERVAL))
            (( waited >= MAXWAIT )) && { say "gpu$g wait timed out - SKIPPING chunk $lo-$hi"; break; }
            continue
        fi
        free=$(nvidia-smi --id="$g" --query-gpu=memory.free --format=csv,noheader,nounits)
        if (( free >= MINFREE )); then
            ok=$((ok + 1))
            # Stability must be measured over time, not from back-to-back reads of
            # the same instant, so sleep between consecutive OK checks.
            (( ok < STABLE )) && sleep "$INTERVAL"
        else
            (( ok > 0 )) && say "gpu$g dipped back (${free} MiB) - resetting stability counter"
            ok=0
            if (( waited >= MAXWAIT )); then
                say "gpu$g still short (${free} MiB) after ${waited}s - SKIPPING chunk $lo-$hi"
                break
            fi
            say "waiting on gpu$g (${free} MiB free, need $MINFREE)"
            sleep "$INTERVAL"; waited=$((waited + INTERVAL))
        fi
    done
    (( ok < STABLE )) && continue

    say "launching gpu$g targets $lo-$hi ($count)"
    NAME="$RUN/gpu$g" GPU="$g" IDS="$ids" NTARGETS="$count" NPR="$NPR" \
        "$HERE/run_mcpp_sampling.sh" 2>&1 | tee -a "$LOG"

    # confirm the chunk actually got past model load before moving to the next GPU
    for _ in $(seq 1 30); do
        sleep 20
        if grep -q "start sampling" "$REPO/artifacts/reproduction/mcpp/$RUN/gpu$g/run.log" 2>/dev/null; then
            say "gpu$g is sampling"; break
        fi
        if [ -f "$REPO/artifacts/reproduction/mcpp/$RUN/gpu$g/exit_code" ]; then
            say "gpu$g DIED early (exit $(cat "$REPO/artifacts/reproduction/mcpp/$RUN/gpu$g/exit_code"))"; break
        fi
    done
done

say "launcher done"
