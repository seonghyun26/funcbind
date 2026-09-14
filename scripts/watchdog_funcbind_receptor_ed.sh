#!/usr/bin/env bash
# Keep the receptor-ED FuncBind run alive across process-level deaths.
#
# The 2026-08-15 run was lost because nothing noticed it had stopped: the
# container bounced, the trainer died, and four idle H200s sat there until a
# human looked. This supervises the run instead -- when the trainer exits, it
# picks the furthest-along checkpoint in the chain and resumes into a fresh exp
# dir, so a crash costs at most the epoch in flight rather than the whole run.
#
# WHAT IT COVERS
#   trainer process death: CUDA OOM, NCCL collapse, an uncaught exception, the
#   OOM killer, a rank wedging and taking the job down.
#
# WHAT IT CANNOT COVER
#   a container restart. /etc/services.d and /etc/cont-init.d are on the
#   container overlay, not the /dev/md127 PVC, so nothing added there survives a
#   bounce -- and this watchdog is a plain process, so it dies with the container
#   exactly like the trainer does. After a bounce, relaunch by hand:
#       scripts/watchdog_funcbind_receptor_ed.sh
#   It is idempotent: with no trainer running it resumes from the furthest-along
#   checkpoint, which is precisely the post-bounce recovery step.
#
# GIVING UP
#   Restarting forever is worse than stopping: a config that OOMs on step 1 would
#   burn the GPUs indefinitely and bury the real error. A cycle that fails to
#   advance the checkpoint epoch counts as no progress, and MAX_NO_PROGRESS of
#   those in a row stops the watchdog and leaves a .watchdog_gave_up marker.
#
# Usage:
#   scripts/watchdog_funcbind_receptor_ed.sh              # attach + supervise
#   MAX_NO_PROGRESS=5 scripts/watchdog_funcbind_receptor_ed.sh
#   DRY_RUN=1 scripts/watchdog_funcbind_receptor_ed.sh    # report state, launch nothing
set -uo pipefail

REPO="${REPO:-${FUNCBIND_ROOT:-/home1/irteam/funcbind}}"
PY="${PY:-$REPO/.repro-env/bin/python}"
CONFIG="${CONFIG:-train_fb_mcpp_holo_density}"
EXPS="$REPO/exps/funcbind"

# BASE_NAME must match the live run, or a restart silently changes the experiment.
#
# bsz 6 -> 5 as of 2026-08-23. At bsz=6 a rank held 137.85 GiB of a 139.80 GiB card
# -- 98.6% -- and r4 died of CUDA OOM at epoch 5 when an unrelated 1.90 GiB process
# landed on GPU 2. The run needs headroom for a neighbour, not just for itself.
#
# accum_steps 32 -> 38 keeps the optimizer step comparable across that change:
# 6*32*4 = 768 samples became 5*38*4 = 760, within 1%.
#
# Both values feed the LR schedule, which is ON (use_lr_schedule: true): utils_fb.py
# computes lr = lr0 / sqrt(acc_iter / (ref_batches * bsz * world_size * accum_steps)),
# clamped at lr0 until the ratio passes 1. With ref_batches=20040 the knee sits at
# ~15.2M samples and acc_iter was 4.1M at this change, so the run is still on flat
# lr0=1e-3 and the 1% batch shift moves the knee by the same 1%.
BASE_NAME="${BASE_NAME:-20260816_fb_mcpp_champion_receptor_ed_zeroinit_resumed_bsz6_ga32}"
BATCH_SIZE="${BATCH_SIZE:-5}"
ACCUM_STEPS="${ACCUM_STEPS:-38}"
NUM_WORKERS="${NUM_WORKERS:-6}"
GPUS="${GPUS:-0,1,2,3}"

# Root of the resume chain: used only until the new run writes its first
# checkpoint, after which its own output is always further along.
ROOT_SRC="${ROOT_SRC:-$EXPS/20260815_fb_mcpp_champion_receptor_ed_zeroinit_ga27_pf4_ooo_val1_ckpt1}"

POLL_SECONDS="${POLL_SECONDS:-60}"
RETRY_DELAY="${RETRY_DELAY:-120}"
MAX_NO_PROGRESS="${MAX_NO_PROGRESS:-3}"

# A wedged rank -- an NCCL collective that never returns, a hung loader -- keeps
# the processes alive, so a liveness check alone waits on it forever. Treat a
# silent run.log as death instead. The margin is deliberately wide: the longest
# legitimate quiet stretch is a checkpoint write (measured 73-89 s) plus
# validation and the every-10-epochs sampling round, and a false positive costs
# the epoch in flight. Set STALL_SECONDS=0 to disable.
STALL_SECONDS="${STALL_SECONDS:-5400}"
STALL_GRACE="${STALL_GRACE:-60}"

# The stall guard only fires once the run is already dead in the water for 90 min,
# with four H200s idle the whole time. Anonymous memory is the signal that gets
# there first. Do NOT watch memory.current: page cache expands to fill the cgroup,
# so it reads 100% on a perfectly healthy run and alarms constantly. anon is the
# part that cannot be reclaimed -- when it crowds out the last of the page cache
# the trainer does not crash, it thrashes, and a single optimizer step stretches
# from 48 s to 38-91 min. That is what killed 2026-08-17 and 2026-08-18. Set
# MEM_POLL_SECONDS=0 to disable.
MEM_POLL_SECONDS="${MEM_POLL_SECONDS:-600}"
MEM_WARN_PCT="${MEM_WARN_PCT:-70}"
MEM_CRIT_PCT="${MEM_CRIT_PCT:-85}"

LOG="${LOG:-$EXPS/watchdog_${BASE_NAME}.log}"
STATE="${STATE:-$EXPS/watchdog_${BASE_NAME}.state}"
GAVE_UP="$EXPS/.watchdog_gave_up_${BASE_NAME}"
LOCK="${LOCK:-$EXPS/.watchdog_${BASE_NAME}.lock}"

mkdir -p "$EXPS"
say() { echo "[$(date --iso-8601=seconds)] [watchdog] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1

# One watchdog per run. Two would race to relaunch and start duplicate trainers
# on the same four GPUs.
exec 9>"$LOCK"
if ! flock -n 9; then
    say "another watchdog already holds $LOCK; refusing to start a second"
    exit 3
fi
echo $$ >&9

# The [t] keeps the pattern from matching a shell that merely carries it on its
# own command line -- the classic way a liveness check convinces itself the job
# is up when only the checker is.
trainer_pids() { pgrep -f "[t]rain_fb\.py --config-name $CONFIG" 2>/dev/null; }
training_active() { [ -n "$(trainer_pids)" ]; }

# "<acc_iter> <global_step> <epoch>" for a run dir's checkpoint, empty if unreadable.
# mmap + the meta device keeps this off the 77 GiB of tensor payload.
#
# acc_iter is the ordering key, NOT epoch. train_fb.py runs
# `for epoch in range(0, num_epochs)`, so the epoch counter restarts at 0 on every
# resume and a fresh restart writes epoch=1 while its own source says epoch=5.
# Ranking by epoch would resume from the stale source and silently discard the
# newer work. acc_iter (samples seen) and global_step both carry across resumes;
# acc_iter is preferred because it is invariant to batch size and accum_steps.
ckpt_progress() {
    local dir="$1"
    [ -f "$dir/checkpoint.pth.tar" ] || return 1
    "$PY" - "$dir/checkpoint.pth.tar" <<'EOF' 2>/dev/null
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="meta", weights_only=False, mmap=True)
    print(int(ck.get("acc_iter", 0)), int(ck.get("global_step", 0)), int(ck.get("epoch", 0)))
except Exception:
    sys.exit(1)
EOF
}

# The furthest-along checkpoint across the whole resume chain. Ordering by samples
# seen rather than mtime matters too: a restart that dies during its first save
# would otherwise look newer than the good checkpoint it was resumed from.
best_source() {
    local best_dir="" best_acc=-1 dir prog acc
    for dir in "$ROOT_SRC" "$EXPS/$BASE_NAME" "$EXPS/${BASE_NAME}_r"*; do
        [ -d "$dir" ] || continue
        prog="$(ckpt_progress "$dir")" || continue
        [ -n "$prog" ] || continue
        acc="${prog%% *}"
        if [ "$acc" -gt "$best_acc" ]; then
            best_acc="$acc"; best_dir="$dir $prog"
        fi
    done
    [ -n "$best_dir" ] || return 1
    echo "$best_dir"
}

next_exp_name() {
    local n=1
    while [ -d "$EXPS/${BASE_NAME}_r${n}" ]; do n=$((n + 1)); done
    echo "${BASE_NAME}_r${n}"
}

# The exp dir the trainer is currently writing to: whatever the last launch
# recorded, falling back to the run this watchdog was armed for.
active_exp_dir() {
    local name=""
    [ -f "$STATE" ] && name="$(sed -n 's/^active_exp=//p' "$STATE" | tail -1)"
    echo "$EXPS/${name:-$BASE_NAME}"
}

# Runs for the watchdog's lifetime. The main loop blocks inside the resume script
# while a trainer is up, so it cannot poll for a stall itself -- this can.
stall_guard() {
    local log age pids
    while true; do
        sleep "$POLL_SECONDS"
        [ "$STALL_SECONDS" -gt 0 ] || continue
        training_active || continue
        log="$(active_exp_dir)/run.log"
        [ -f "$log" ] || continue
        age=$(( $(date +%s) - $(stat -c %Y "$log") ))
        [ "$age" -ge "$STALL_SECONDS" ] || continue

        say "STALL: $log silent for ${age}s (limit ${STALL_SECONDS}s); terminating the trainer"
        pids="$(trainer_pids)"
        [ -n "$pids" ] || continue
        kill -TERM $pids 2>/dev/null
        sleep "$STALL_GRACE"
        pids="$(trainer_pids)"
        if [ -n "$pids" ]; then
            say "STALL: ranks survived SIGTERM; sending SIGKILL"
            kill -KILL $pids 2>/dev/null
        fi
    done
}

# Anonymous bytes in the cgroup, and the cap. Empty/failure when this is not
# cgroup v2 or the cgroup is uncapped, in which case the guard just idles.
cgroup_anon() { awk '/^anon /{print $2; exit}' /sys/fs/cgroup/memory.stat 2>/dev/null; }
cgroup_cap() {
    local m
    m="$(cat /sys/fs/cgroup/memory.max 2>/dev/null)" || return 1
    [ -n "$m" ] && [ "$m" != "max" ] || return 1
    echo "$m"
}

# Logs only while over the warn threshold, so a healthy run stays silent and a
# sick one leaves a trend -- level, growth rate, and how long until the cap.
mem_guard() {
    local cap anon now pct prev_anon=0 prev_t=0 msg
    while true; do
        sleep "$MEM_POLL_SECONDS"
        # A restart resets the footprint; carrying the old sample across one would
        # report a huge negative rate.
        training_active || { prev_anon=0; prev_t=0; continue; }
        cap="$(cgroup_cap)" || continue
        anon="$(cgroup_anon)"
        [ -n "$anon" ] || continue
        now="$(date +%s)"
        pct=$(( anon * 100 / cap ))

        if [ "$pct" -ge "$MEM_WARN_PCT" ]; then
            msg="$(awk -v a="$anon" -v c="$cap" -v pa="$prev_anon" -v dt="$(( now - prev_t ))" 'BEGIN {
                g = 1073741824
                printf "anon %.0f GiB of %.0f GiB cap (%d%%)", a/g, c/g, a*100/c
                if (pa > 0 && dt > 0) {
                    r = (a - pa) / g / (dt / 3600.0)
                    printf ", %+.1f GiB/h", r
                    if (r > 0) printf ", ~%.1f h of headroom left", (c - a) / g / r
                }
            }')"
            if [ "$pct" -ge "$MEM_CRIT_PCT" ]; then
                say "MEMORY CRITICAL: $msg -- expect step times to collapse and the stall guard to fire"
            else
                say "MEMORY WARN: $msg"
            fi
        fi

        prev_anon="$anon"
        prev_t="$now"
    done
}

if [ -f "$GAVE_UP" ]; then
    say "$GAVE_UP exists: a previous watchdog stopped after repeated no-progress"
    say "investigate, then remove that file to re-arm"
    exit 4
fi

say "armed for $BASE_NAME (bsz=$BATCH_SIZE accum=$ACCUM_STEPS gpus=$GPUS)"
say "restart policy: stop after $MAX_NO_PROGRESS consecutive cycles without a newer checkpoint"

if src_info="$(best_source)"; then
    read -r d a s e <<<"$src_info"
    say "furthest checkpoint: $d (acc_iter=$a global_step=$s epoch=$e)"
else
    say "no readable checkpoint anywhere in the chain yet"
fi

if [ -n "${DRY_RUN:-}" ]; then
    if training_active; then
        say "DRY_RUN: trainer live (pids $(trainer_pids | tr '\n' ' ')); would supervise"
    else
        say "DRY_RUN: no trainer; would resume into $(next_exp_name)"
    fi
    exit 0
fi

GUARD_PIDS=""
if [ "$STALL_SECONDS" -gt 0 ]; then
    stall_guard &
    STALL_PID=$!
    GUARD_PIDS="$GUARD_PIDS $STALL_PID"
    say "stall guard armed (pid $STALL_PID): kill the trainer if run.log goes quiet for ${STALL_SECONDS}s"
else
    say "stall guard disabled"
fi

if [ "$MEM_POLL_SECONDS" -gt 0 ]; then
    mem_guard &
    MEM_PID=$!
    GUARD_PIDS="$GUARD_PIDS $MEM_PID"
    say "memory guard armed (pid $MEM_PID): report anon memory past ${MEM_WARN_PCT}% of the cgroup cap, critical at ${MEM_CRIT_PCT}%"
else
    say "memory guard disabled"
fi

if [ -n "$GUARD_PIDS" ]; then
    # Killing the guard shells is not enough: each is parked in `sleep`, and that
    # sleep is a separate child that inherits fd 9 and therefore the flock. It
    # outlives its parent and keeps the lock for the rest of its interval -- up to
    # MEM_POLL_SECONDS. That is what made the 2026-08-18 and 2026-08-24 relaunches
    # exit 3 with "another watchdog already holds", leaving nothing supervising.
    # Reap the sleep first, then the guard.
    stop_guards() {
        local g
        for g in $GUARD_PIDS; do
            pkill -P "$g" 2>/dev/null
            kill "$g" 2>/dev/null
        done
    }
    trap stop_guards EXIT INT TERM
fi

# Attach to whatever is already running rather than starting a rival trainer.
if training_active; then
    say "attaching to live trainer (pids $(trainer_pids | tr '\n' ' '))"
    while training_active; do sleep "$POLL_SECONDS"; done
    say "supervised trainer exited"
fi

no_progress=0
while true; do
    src_info="$(best_source)" || {
        say "no valid checkpoint to resume from; stopping"
        exit 5
    }
    read -r src acc_before step_before epoch_before <<<"$src_info"

    exp_name="$(next_exp_name)"
    say "resuming from $src (acc_iter=$acc_before global_step=$step_before) into $exp_name"
    {
        echo "base_name=$BASE_NAME"
        echo "active_exp=$exp_name"
        echo "resumed_from=$src"
        echo "acc_iter_at_launch=$acc_before"
        echo "global_step_at_launch=$step_before"
        echo "epoch_in_source_run=$epoch_before"
        echo "batch_size=$BATCH_SIZE accum_steps=$ACCUM_STEPS"
        echo "updated=$(date --iso-8601=seconds)"
    } >"$STATE"

    started="$(date +%s)"
    # CONFIG must be forwarded explicitly. It is a plain (unexported) variable here, so
    # without this line the resume script falls back to ITS default config -- and a
    # watchdog armed for one experiment would quietly resume a different one.
    SRC="$src" EXP_NAME="$exp_name" CONFIG="$CONFIG" \
    BATCH_SIZE="$BATCH_SIZE" ACCUM_STEPS="$ACCUM_STEPS" NUM_WORKERS="$NUM_WORKERS" \
    GPUS="$GPUS" \
        "$REPO/scripts/resume_funcbind_receptor_ed.sh"
    rc=$?
    elapsed=$(( $(date +%s) - started ))

    if [ "$rc" -eq 0 ]; then
        say "trainer exited cleanly after ${elapsed}s; nothing left to supervise"
        exit 0
    fi

    acc_after=""
    if prog_after="$(ckpt_progress "$EXPS/$exp_name")"; then
        acc_after="${prog_after%% *}"
    fi
    if [ -n "$acc_after" ] && [ "$acc_after" -gt "$acc_before" ]; then
        say "cycle made progress: acc_iter $acc_before -> $acc_after (rc=$rc, ${elapsed}s)"
        no_progress=0
    else
        no_progress=$((no_progress + 1))
        say "cycle wrote no newer checkpoint (rc=$rc, ${elapsed}s) -- $no_progress/$MAX_NO_PROGRESS"
    fi

    if [ "$no_progress" -ge "$MAX_NO_PROGRESS" ]; then
        say "giving up after $no_progress cycles without progress; see $EXPS/$exp_name/run.log"
        {
            echo "gave_up=$(date --iso-8601=seconds)"
            echo "last_exp=$exp_name"
            echo "last_rc=$rc"
            echo "acc_iter_stuck_at=$acc_before"
        } >"$GAVE_UP"
        exit 6
    fi

    say "retrying in ${RETRY_DELAY}s"
    sleep "$RETRY_DELAY"
done
