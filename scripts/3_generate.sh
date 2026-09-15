#!/usr/bin/env bash
# Generate density-conditioned MCP samples from the trained checkpoint.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

export FUNCBIND_ROOT="${FUNCBIND_ROOT:-$REPO}"
export FB_PATH="${FB_PATH:-$REPO/exps/funcbind/fb_mcpp_holo_density}"
export CONFIG="${CONFIG:-sample_fb_mcpp_holo_density}"

case "${SMOKE:-0}" in
    1|true|yes)
        export NAME="${NAME:-smoke}"
        export GPU="${GPU:-0}"
        export IDS="${IDS:-[0]}"
        export NTARGETS="${NTARGETS:-1}"
        export NPR="${NPR:-8}"
        export NCHAINS="${NCHAINS:-64}"
        export NATT="${NATT:-1}"
        if [ -n "${DRY_RUN:-}" ]; then
            printf 'DRY_RUN: smoke gpu=%s ids=%s targets=%s npr=%s checkpoint=%s\n' \
                "$GPU" "$IDS" "$NTARGETS" "$NPR" "$FB_PATH"
            exit 0
        fi
        exec "$HERE/run_mcpp_sampling.sh"
        ;;
    0|false|no|'')
        export GPU_COUNT="${GPU_COUNT:-8}"
        export CHUNKS="${CHUNKS:-0,1,2,3,4,5,6,7}"
        export NPR="${NPR:-100}"
        exec "$HERE/run_mcpp_paper_run.sh"
        ;;
    *)
        echo "SMOKE must be 0 or 1" >&2
        exit 2
        ;;
esac
