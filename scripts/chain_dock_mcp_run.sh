#!/usr/bin/env bash
# MCP mid-training check: sampling -> eval tree -> chemistry summary -> Vina docking.
#
#   RUN=cmp10_r14 setsid nohup bash scripts/chain_dock_mcp_run.sh &
#
# Scores a fine-tune checkpoint the way cmp10 scored _r3 (3.17M) on 2026-08-20 and
# cmp10_r10 scored _r10 (8.21M), so every round is comparable: same 10 targets,
# same 25/receptor x 256 chains x 1 attempt, same full deposited receptor and
# reference cyclic peptide, same Vina exhaustiveness 16.
#
# The vanilla arm is NOT re-run: fb_unified has not changed and the sampling
# settings and seed are identical, so cmp10/_eval/vanilla stays the baseline.
#
# WORKER COUNT. The container is capped at 32 cores (nproc reports 128 and lies,
# see the cpu-quota note), and this shares them with the p78 VoxBind docking eval
# and with the funcbind trainer's 24 loader workers once the resume chain brings
# it back. 5 x 2 is what cmp10 used; raising it here would slow all three.
#
# Docking output is appended to dock_chain.log, which is also the driver's default
# --resume-log: re-running this script after an interruption reuses every Vina
# call already in that log instead of redoing it.
set -uo pipefail

FB="${FUNCBIND_ROOT:-${FUNCBIND_ROOT:-/home1/irteam/funcbind}}"
VB="${VOXBIND_PYTHON_ROOT:-${VOXBIND_PYTHON_ROOT:-/home1/irteam/VoxBind}}"
: "${RUN:?set RUN, e.g. cmp10_r14 (the dir under artifacts/reproduction/mcpp/)}"
ROOT="$FB/artifacts/reproduction/mcpp/$RUN"
EVAL="$ROOT/_eval/finetuned"
CHUNKS=(ft_a ft_b ft_c ft_d)
# Arms to print alongside this one. Every past round stays comparable, so the list
# only grows.
COMPARE=${COMPARE:-"$FB/artifacts/reproduction/mcpp/cmp10/_eval/vanilla \
                    $FB/artifacts/reproduction/mcpp/cmp10/_eval/finetuned \
                    $FB/artifacts/reproduction/mcpp/cmp10_r10/_eval/finetuned"}
WORKERS=${WORKERS:-5}
CPU=${CPU:-2}
EXH=${EXH:-16}
POLL=${POLL:-120}
TIMEOUT=${TIMEOUT:-28800}
LOG="$ROOT/dock_chain_driver.log"
VOXDOCK=/opt/conda/envs/voxdock/bin/python
# The driver shells out to pdb2pqr30 and prepare_receptor4 for receptor prep, and
# both live in the voxdock env's bin. Calling its python by absolute path is NOT
# enough: without the env on PATH every dock returns None in 0.0s with
# "FileNotFoundError: pdb2pqr30" buried in per_mol, and the run still exits 0 with
# a complete-looking JSON of nulls. That is exactly what the 04:03 attempt did.
export PATH=/opt/conda/envs/voxdock/bin:$PATH

mkdir -p "$ROOT"
say() { echo "[$(date --iso-8601=seconds)] [dock-$RUN] $*" | tee -a "$LOG"; }

say "waiting for ${#CHUNKS[@]} sampling chunks"
waited=0
while :; do
    done_n=0
    for c in "${CHUNKS[@]}"; do [ -f "$ROOT/$c/exit_code" ] && done_n=$((done_n + 1)); done
    (( done_n >= ${#CHUNKS[@]} )) && { say "all $done_n chunks finished"; break; }
    if (( waited >= TIMEOUT )); then say "TIMEOUT with $done_n/${#CHUNKS[@]} done -- docking what exists"; break; fi
    sleep "$POLL"; waited=$((waited + POLL))
done
for c in "${CHUNKS[@]}"; do
    rc=$(cat "$ROOT/$c/exit_code" 2>/dev/null || echo MISSING)
    [ "$rc" = "0" ] || say "WARNING: chunk $c exited $rc"
done

say "assembling the eval tree"
"$FB/.repro-env/bin/python" "$FB/scripts/build_mcpp_eval_tree.py" \
    --chunks "${CHUNKS[@]/#/$ROOT/}" --out "$EVAL" 2>&1 | tee -a "$LOG"
n_targets=$(find "$EVAL" -maxdepth 1 -name 'target_*' -type d | wc -l)
[ "$n_targets" -gt 0 ] || { say "ABORT: no targets in $EVAL"; exit 1; }
say "$n_targets targets assembled"

# Cheap and immediately useful: yield and chemistry need no Vina at all, and the
# docking below runs for hours.
say "chemistry summary (no docking)"
( cd "$VB" && "$VOXDOCK" "$FB/scripts/summarize_mcpp_chem.py" "$EVAL" \
    --compare $COMPARE \
    --out "$ROOT/chem_summary.json" ) 2>&1 | tee -a "$LOG"

say "docking: workers=$WORKERS cpu=$CPU exh=$EXH -> $EVAL/eval_docking_results.json"
( cd "$VB" && "$VOXDOCK" voxbind/exps/frozenenc_probes/run_docking_eval_parallel.py \
    "$EVAL" --workers "$WORKERS" --cpu "$CPU" --exh "$EXH" ) 2>&1 | tee -a "$EVAL/dock_chain.log"
# $? after a pipe is tee's status, which is 0 even when the driver died.
say "docking exited with code ${PIPESTATUS[0]}"
say "done"
