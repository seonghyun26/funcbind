#!/usr/bin/env bash
# Direct 60M VoxBind segment launcher for the epoch-500 supervisor.
set -euo pipefail

if (( $# != 2 )); then
    echo "usage: $0 COMPLETED_EPOCHS TARGET_EPOCHS" >&2
    exit 2
fi
completed="$1"
target="$2"
if ! [[ "${completed}" =~ ^[0-9]+$ && "${target}" =~ ^[0-9]+$ ]]; then
    echo "completed and target must be integers" >&2
    exit 2
fi
(( completed < target )) || exit 0

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXBIND_RUNTIME="${project_root}/.voxbind-env-backup/voxbind"
experiment_name="voxbind_frozen_efficient60m_holo_xrayfull_20260729"
experiment_dir="${VOXBIND_PYTHON_ROOT:-/home1/irteam/VoxBind}/voxbind/exps/${experiment_name}"
model_zoo="${VOXBIND_PYTHON_ROOT:-/home1/irteam/VoxBind}/voxbind/model_zoo"

exec "${project_root}/scripts/train_voxbind.sh" \
    --model-zoo-root "${model_zoo}" \
    --model-zoo efficient_60m_v3_mask085 \
    --gpus 0,1,2,3 \
    --exp-name "${experiment_name}" \
    --output-dir "${experiment_dir}" \
    --completed-epochs "${completed}" \
    --target-epoch "${target}" \
    --optimizer adamw \
    --master-port 29629 \
    --wandb-mode online \
    --wandb-run-id mbx79h9v \
    --tags faithful_direct,efficient_60m,resumed_to_500 \
    --num-workers 16 \
    --prefetch-factor 8
