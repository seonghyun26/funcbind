#!/usr/bin/env bash
# Fine-tune FuncBind with deposited X-ray density and the frozen CDG v2 encoder.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

export REPO
export FUNCBIND_ROOT="${FUNCBIND_ROOT:-$REPO}"
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export CONFIG="${CONFIG:-train_fb_mcpp_holo_density_h100}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"

case "${SMOKE:-0}" in
    1|true|yes)
        export EXP_NAME="${EXP_NAME:-fb_mcpp_holo_density_h100_smoke}"
        export N_SAMPLES="${N_SAMPLES:-8}"
        export NUM_EPOCHS="${NUM_EPOCHS:-1}"
        export WANDB_ENABLED="${WANDB_ENABLED:-false}"
        export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-false}"
        ;;
    0|false|no|'')
        # 3_generate.sh uses this stable path by default.
        export EXP_NAME="${EXP_NAME:-fb_mcpp_holo_density}"
        ;;
    *)
        echo "SMOKE must be 0 or 1" >&2
        exit 2
        ;;
esac

exec "$HERE/02_train_mcp_density_conditioned.sh"
