#!/usr/bin/env bash
# Hand the GPUs from funcbind to the frozen_v3 WJS sampling eval, then hand them back.
#
# WHY THE WAIT
#   Only the SAMPLING half of the evaluation needs GPUs (~2h10m measured on the
#   2026-08-15 run: 11:22:57 -> 13:32:05 across four chunks). The docking half is
#   CPU-only Vina and can run alongside training afterwards. So funcbind gives up
#   the GPUs for about two hours, not for the whole evaluation.
#
#   Waiting for the in-flight epoch to checkpoint costs ~74 min and saves the same
#   ~74 min of training that killing it now would throw away -- the eval finishes
#   at the same wall-clock either way, so the wait is free.
#
# WHY THE FALLBACK CHAIN MUST BE STOPPED FIRST
#   chain_voxbind_fallback_after_funcbind.sh treats "no watchdog and no trainer for
#   30 min" as funcbind having failed, and would launch the VoxBind fusion run onto
#   the GPUs this eval is using. The planned outage looks exactly like the failure
#   it watches for, so it is stopped here and re-armed at the end.
set -uo pipefail

FB=/home1/irteam/funcbind
VB=/home1/irteam/VoxBind/voxbind
R4="$FB/exps/funcbind/20260816_fb_mcpp_champion_receptor_ed_zeroinit_resumed_bsz6_ga32_r4"
CKPT="$R4/checkpoint.pth.tar"
BASE_MTIME="${BASE_MTIME:-$(stat -c %Y "$CKPT")}"   # epoch-4 checkpoint, 2026-08-23 16:26:41

EXP="${EXP:-voxbind_frozen_v3_100m_mask090_default_mlp3_h32_sig0.9}"
OUT="${OUT:-samples_frozen_v3_mask090_ep561}"
LOG="${LOG:-$FB/exps/funcbind/eval_frozen_v3_handoff.log}"
POLL="${POLL:-120}"

say() { echo "[$(date --iso-8601=seconds)] [handoff] $*" | tee -a "$LOG"; }
trainer_up() { pgrep -f "[t]rain_fb\.py --config-name train_fb_mcpp_holo_density" >/dev/null 2>&1; }
gpus_busy()  { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }

say "waiting for r4's epoch-5 checkpoint (baseline mtime $(date -d @$BASE_MTIME +%T))"
deadline=$(( $(date +%s) + 21600 ))
while :; do
    m=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
    if [ "$m" -gt "$BASE_MTIME" ]; then
        s1=$(stat -c %s "$CKPT"); sleep 45; s2=$(stat -c %s "$CKPT")
        [ "$s1" = "$s2" ] && { say "checkpoint landed $(date -d @$m +%T) ($s2 bytes)"; break; }
    fi
    [ "$(date +%s)" -ge "$deadline" ] && { say "ABORT: no checkpoint within 6h; nothing touched"; exit 1; }
    sleep "$POLL"
done

say "stopping the fallback chain so the planned outage is not read as a funcbind failure"
pkill -f "[c]hain_voxbind_fallback_after_funcbind.sh" 2>/dev/null; sleep 2

say "stopping the funcbind watchdog"
pkill -TERM -f "[b]ash scripts/watchdog_funcbind_receptor_ed.sh" 2>/dev/null
sleep 10
pkill -KILL -f "[b]ash scripts/watchdog_funcbind_receptor_ed.sh" 2>/dev/null

if trainer_up; then
    say "stopping funcbind ranks"
    pkill -TERM -f "[t]rain_fb\.py --config-name train_fb_mcpp_holo_density" 2>/dev/null
    for _ in $(seq 1 24); do trainer_up || break; sleep 5; done
    trainer_up && { say "ranks survived SIGTERM; SIGKILL"; pkill -KILL -f "[t]rain_fb\.py --config-name train_fb_mcpp_holo_density"; sleep 15; }
fi

for _ in $(seq 1 60); do gpus_busy || break; say "waiting for GPU memory to release"; sleep 20; done
gpus_busy && { say "ABORT: GPUs still occupied; re-arm by hand"; exit 1; }
say "GPUs free"

say "launching eval: EXP=$EXP OUT=$OUT (4 chunks x 25 targets, 100 samples/pocket)"
cd "$VB" || exit 1
EXP="$EXP" OUT="$OUT" bash scripts/62_eval_frozenenc_4gpu.sh >>"$LOG" 2>&1
SAVE="$VB/exps/$OUT"

say "waiting for all 4 chunks to finish"
while :; do
    done_n=$(ls "$SAVE"/_run_gpu*/exit_code 2>/dev/null | wc -l)
    [ "$done_n" -ge 4 ] && break
    sleep "$POLL"
done
for d in "$SAVE"/_run_gpu*; do say "  $(basename "$d") exit=$(cat "$d/exit_code")"; done
say "targets produced: $(ls -d "$SAVE"/target_*/ 2>/dev/null | wc -l)"

say "handing the GPUs back to funcbind"
cd "$FB" || exit 1
setsid nohup scripts/watchdog_funcbind_receptor_ed.sh >> exps/funcbind/watchdog_launch.out 2>&1 < /dev/null &
disown
sleep 45
# The 2026-08-18 outage was a relaunch that lost the flock race and exited 3 while
# reporting nothing. Assume nothing; check that a watchdog is actually up.
if pgrep -f "[b]ash scripts/watchdog_funcbind_receptor_ed.sh" >/dev/null; then
    say "watchdog back up (pids $(pgrep -f '[b]ash scripts/watchdog_funcbind_receptor_ed.sh' | tr '\n' ' '))"
else
    say "PROBLEM: watchdog did NOT come back -- funcbind is unsupervised, relaunch by hand"
fi

setsid nohup bash scripts/chain_voxbind_fallback_after_funcbind.sh >/dev/null 2>&1 < /dev/null &
disown
sleep 3
pgrep -f "[c]hain_voxbind_fallback" >/dev/null && say "fallback chain re-armed" || say "PROBLEM: fallback chain did not re-arm"
say "done -- docking is CPU-only and can now run alongside training"
