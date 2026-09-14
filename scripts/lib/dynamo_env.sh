#!/usr/bin/env bash
# Decide TORCHDYNAMO_DISABLE from the training config, and export it.
#
#   PY=... CONFIG=... REPO=... source scripts/lib/dynamo_env.sh
#
# WHY THIS IS NOT JUST "=1". `performance.compile_backend` in the config asks for
# torch.compile, and train_fb.py acts on it; hardcoding TORCHDYNAMO_DISABLE=1 in a
# launcher silently defeats that, so the same run trains differently depending on which
# script started it. That actually happened: 20260910_fb_mcpp_default_atomblob7_zeroinit
# ran eager for 20 epochs from the fresh launcher, then compiled after the watchdog
# resumed it -- and the compiled half is FASTER.
#
# MEASURED on this box (5.14B denoiser, 4xH200, bsz 5 x accum 38), samples/s:
#     eager    median 13.66  (n=4564 steps)
#     compile  median 16.42  (n=738 steps)   -> 1.20x
# Inductor can build here because the funcbind venv ships its own conda toolchain
# (.repro-env/bin/x86_64-conda-linux-gnu-gcc); the container itself has no cc, which is
# why this was assumed impossible for a long time.
#
# An explicit TORCHDYNAMO_DISABLE in the environment still wins, so a caller can force
# eager for a debugging run without editing anything.
: "${PY:?dynamo_env.sh needs PY}"
: "${CONFIG:?dynamo_env.sh needs CONFIG}"
: "${REPO:?dynamo_env.sh needs REPO}"

if [ -z "${TORCHDYNAMO_DISABLE:-}" ]; then
    _cb="$("$PY" - "$REPO/funcbind/configs/$CONFIG.yaml" <<'PYCFG' 2>/dev/null || true
import sys, yaml
try:
    c = yaml.safe_load(open(sys.argv[1])) or {}
    print(str((c.get("performance") or {}).get("compile_backend", "") or ""))
except Exception:
    print("")
PYCFG
)"
    if [ -n "$_cb" ]; then TORCHDYNAMO_DISABLE=0; else TORCHDYNAMO_DISABLE=1; fi
    unset _cb
fi
export TORCHDYNAMO_DISABLE

# Inductor needs a compiler at runtime. The venv's toolchain is the one that works here;
# without it dynamo burns time recompiling and falls back to eager anyway.
if [ "$TORCHDYNAMO_DISABLE" = "0" ] && [ -x "$REPO/.repro-env/bin/x86_64-conda-linux-gnu-gcc" ]; then
    export CC="${CC:-$REPO/.repro-env/bin/x86_64-conda-linux-gnu-gcc}"
    export CXX="${CXX:-$REPO/.repro-env/bin/x86_64-conda-linux-gnu-g++}"
fi
