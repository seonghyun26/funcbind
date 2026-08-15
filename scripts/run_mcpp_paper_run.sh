#!/usr/bin/env bash
# Full MCP paper reproduction: 100 test targets x 100 samples, 25 targets per GPU.
#
#   scripts/run_mcpp_paper_run.sh            # all 4 chunks
#   CHUNKS=0,1 scripts/run_mcpp_paper_run.sh # only gpu0 and gpu1
#
# Each chunk is launched through run_mcpp_sampling.sh (detached, own run.log and
# exit_code) once its GPU has MINFREE MiB free. Waiting is bounded by MAXWAIT so
# this never hangs forever behind another job.
set -uo pipefail

REPO=/home1/irteam/funcbind
HERE="$(cd "$(dirname "$0")" && pwd)"

MINFREE=${MINFREE:-90000}          # MiB of free GPU memory required per chunk
MAXWAIT=${MAXWAIT:-7200}           # seconds to wait per GPU before giving up
STABLE=${STABLE:-3}                # consecutive OK checks before launching
INTERVAL=${INTERVAL:-60}
CHUNKS=${CHUNKS:-0,1,2,3}
RUN=${RUN:-paper_run}
NPR=${NPR:-100}

LOG="$REPO/artifacts/reproduction/mcpp/$RUN/launcher.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

say "MCP reproduction: 100 targets, chunks=$CHUNKS, minfree=${MINFREE}MiB, npr=$NPR"

for g in ${CHUNKS//,/ }; do
    lo=$((g * 25)); hi=$((lo + 24))
    ids="[$(seq -s, $lo $hi)]"

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

    say "launching gpu$g targets $lo-$hi (25)"
    NAME="$RUN/gpu$g" GPU="$g" IDS="$ids" NTARGETS=25 NPR="$NPR" \
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
