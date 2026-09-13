#!/usr/bin/env bash
# crashguard_funcbind_receptor_ed.sh — cut the 90-minute recovery lag after a rank dies.
#
# WHY THIS EXISTS, given the watchdog already restarts the run:
#   When rank 0 raises (CUDA OOM is the recurring one here), it parks in teardown while
#   ranks 1-3 spin at 100% GPU inside an NCCL collective that will never complete. The
#   watchdog's own stall guard does catch this, but only via STALL_SECONDS=5400 -- so on
#   2026-08-30 the trainer died at 14:58 and did not resume until 16:31. The traceback is
#   in run.log the moment it happens; reading it turns 93 minutes of four idle H200s into
#   about two.
#
# It does NOT restart anything. It kills the wedged ranks and lets the existing watchdog
# do what it already does when a trainer dies. Deliberately separate from that script:
# bash reads a running script incrementally, so editing the live watchdog risks corrupting
# it mid-execution.
#
# Safety -- all four must hold before it will kill:
#   1. a crash signature is present in the active run.log
#   2. no optimizer_step line was written after that signature (not a caught/retried error)
#   3. the log has been silent for >= QUIET_SECONDS (the traceback has finished printing)
#   4. trainer processes actually exist
# Anything short of that and it waits. Worst case it does nothing and the stall guard
# still fires at 5400s, exactly as today.
set -uo pipefail

REPO=/home1/irteam/funcbind
EXPS="$REPO/exps/funcbind"
BASE_NAME="${BASE_NAME:-20260816_fb_mcpp_champion_receptor_ed_zeroinit_resumed_bsz6_ga32}"
CONFIG="${CONFIG:-train_fb_mcpp_holo_density}"
STATE="$EXPS/watchdog_${BASE_NAME}.state"
LOG="$EXPS/crashguard_${BASE_NAME}.log"
LOCK="$EXPS/.crashguard_${BASE_NAME}.lock"

POLL_SECONDS="${POLL_SECONDS:-30}"
QUIET_SECONDS="${QUIET_SECONDS:-180}"
GRACE="${GRACE:-60}"
SIG='torch\.OutOfMemoryError|torch\.cuda\.OutOfMemoryError|Error executing job with overrides|RuntimeError: CUDA error|NCCL error'

say() { echo "[$(date --iso-8601=seconds)] [crashguard] $*" | tee -a "$LOG"; }

exec 9>"$LOCK"
flock -n 9 || { say "another crashguard holds $LOCK; exiting"; exit 3; }
echo $$ >&9

trainer_pids() { pgrep -f "[t]rain_fb\.py --config-name $CONFIG" 2>/dev/null; }
active_log() {
    local name=""
    [ -f "$STATE" ] && name="$(sed -n 's/^active_exp=//p' "$STATE" | tail -1)"
    echo "$EXPS/${name:-$BASE_NAME}/run.log"
}

say "armed for $BASE_NAME (poll ${POLL_SECONDS}s, quiet ${QUIET_SECONDS}s, grace ${GRACE}s)"
say "signatures: $SIG"

while true; do
    sleep "$POLL_SECONDS"
    pids="$(trainer_pids)"; [ -n "$pids" ] || continue
    log="$(active_log)"; [ -f "$log" ] || continue

    # Only the tail matters: an old traceback from a previous epoch is not news, and
    # condition 2 below is what actually rules those out.
    tail_txt="$(tail -400 "$log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')"
    echo "$tail_txt" | grep -qE "$SIG" || continue

    # 2. did training carry on past the error? then it was handled -- leave it alone.
    # Anchor on the progress-line FORMAT. A bare /optimizer_step/ also matches the
    # traceback's own Lightning stack frames (fabric/.../optimizer_step), which made
    # a real OOM look like a recovered one.
    after="$(echo "$tail_txt" | awk -v sig="$SIG" '
        $0 ~ sig {seen=1; next}
        seen && /^>> epoch / && /optimizer_step/ {n++}
        END {print n+0}')"
    [ "$after" -eq 0 ] || continue

    # 3. let the traceback finish printing before judging the log dead.
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    [ "$age" -ge "$QUIET_SECONDS" ] || continue

    say "CRASH: $log carries a fatal signature, no progress after it, silent ${age}s"
    say "CRASH: $(echo "$tail_txt" | grep -oE "$SIG" | tail -1) -- terminating ranks: $(echo $pids | tr '\n' ' ')"
    kill -TERM $pids 2>/dev/null
    sleep "$GRACE"
    pids="$(trainer_pids)"
    if [ -n "$pids" ]; then
        say "CRASH: ranks survived SIGTERM; sending SIGKILL"
        kill -KILL $pids 2>/dev/null
    fi
    say "CRASH: ranks down; the watchdog resume loop takes it from here"
    sleep 300      # let the watchdog relaunch before looking again
done
