#!/usr/bin/env bash
# Resume the receptor-ED FuncBind fine-tune once an MCP sampling round finishes.
#
#   RUN=cmp10_r14 setsid nohup bash scripts/chain_resume_funcbind_after_mcp_run.sh &
#
# The four H200s get handed to sampling to score a mid-training checkpoint the way
# cmp10 scored _r3. Everything that would otherwise put work on those GPUs must be
# stopped first, and this puts it all back:
#   - watchdog_funcbind_receptor_ed.sh, which resumes the trainer on its own
#   - crashguard_funcbind_receptor_ed.sh, which kills ranks it reads as wedged
#   - chain_voxbind_fallback_after_funcbind.sh, which claims the GPUs for the
#     VoxBind v4 fusion run 30 min after funcbind disappears
# All three come back here, so the pause costs one epoch and nothing else.
#
# Relaunching the watchdog IS the resume: with no trainer running it picks the
# furthest-along checkpoint by acc_iter -- never by epoch, which restarts at 0 on
# every resume -- and continues into the next free _rN.
set -uo pipefail

REPO=/home1/irteam/funcbind
: "${RUN:?set RUN, e.g. cmp10_r14 (the dir under artifacts/reproduction/mcpp/)}"
CMP="$REPO/artifacts/reproduction/mcpp/$RUN"
CHUNKS=(ft_a ft_b ft_c ft_d)
POLL=${POLL:-120}
TIMEOUT=${TIMEOUT:-28800}          # 8 h; a chunk took 1h31m on 2026-08-27
LOG="$REPO/exps/funcbind/chain_resume_after_${RUN}.log"

say() { echo "[$(date --iso-8601=seconds)] [chain-$RUN] $*" | tee -a "$LOG"; }

say "waiting for ${#CHUNKS[@]} sampling chunks under $CMP"
waited=0
while :; do
    done_n=0
    for c in "${CHUNKS[@]}"; do [ -f "$CMP/$c/exit_code" ] && done_n=$((done_n + 1)); done
    (( done_n >= ${#CHUNKS[@]} )) && { say "all $done_n chunks finished"; break; }
    if (( waited >= TIMEOUT )); then
        say "TIMEOUT after ${waited}s with $done_n/${#CHUNKS[@]} done -- resuming training anyway"
        break
    fi
    sleep "$POLL"; waited=$((waited + POLL))
done

for c in "${CHUNKS[@]}"; do
    rc=$(cat "$CMP/$c/exit_code" 2>/dev/null || echo MISSING)
    [ "$rc" = "0" ] || say "WARNING: chunk $c exited $rc"
done

# A rank that is still tearing down still holds its GPU memory, and the trainer
# needs 138 of the 140 GiB on every card.
say "waiting for the sampling ranks to release the GPUs"
for _ in $(seq 1 90); do
    pgrep -f "[s]ample_fb.py" >/dev/null || break
    sleep 10
done

cd "$REPO" || exit 1
if pgrep -f "[b]ash scripts/watchdog_funcbind_receptor_ed.sh" >/dev/null; then
    say "a watchdog is already running; leaving it alone"
else
    say "relaunching the watchdog (resumes from the furthest checkpoint into the next _rN)"
    setsid nohup bash scripts/watchdog_funcbind_receptor_ed.sh \
        >>"$REPO/exps/funcbind/watchdog_launch.out" 2>&1 &
    sleep 30
fi

if pgrep -f "[b]ash ./scripts/crashguard_funcbind_receptor_ed.sh|[b]ash scripts/crashguard_funcbind_receptor_ed.sh" >/dev/null; then
    say "crashguard already running; leaving it alone"
else
    say "relaunching the crashguard"
    setsid nohup bash scripts/crashguard_funcbind_receptor_ed.sh >/dev/null 2>&1 &
fi

# Re-arm the VoxBind fallback last: it treats "no watchdog and no trainer" as a
# failure signal, so arming it before the watchdog is up starts its counter for
# no reason.
if pgrep -f "[b]ash scripts/chain_voxbind_fallback_after_funcbind.sh" >/dev/null; then
    say "the VoxBind fallback chain is already armed; leaving it alone"
else
    say "re-arming chain_voxbind_fallback_after_funcbind.sh"
    setsid nohup bash scripts/chain_voxbind_fallback_after_funcbind.sh \
        >>"$REPO/exps/funcbind/chain_v4_launch.out" 2>&1 &
fi

sleep 60
say "trainer ranks now: $(pgrep -cf '[t]rain_fb\.py --config-name train_fb_mcpp_holo_density')"
say "done"
