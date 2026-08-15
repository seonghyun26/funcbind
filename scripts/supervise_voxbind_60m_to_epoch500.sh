#!/usr/bin/env bash
set -u -o pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
voxbind_root="/home1/irteam/VoxBind/voxbind"
runtime="${project_root}/.voxbind-env-backup/voxbind"
experiment_name="voxbind_frozen_efficient60m_holo_xrayfull_20260729"
experiment_dir="${voxbind_root}/exps/${experiment_name}"
checkpoint="${experiment_dir}/checkpoint.pth.tar"
checkpoint_350="${experiment_dir}/checkpoint_epoch_0350.pth.tar"
checkpoint_500="${experiment_dir}/checkpoint_epoch_0500.pth.tar"
retry_delay_seconds=300

training_active() {
    pgrep -f "[t]rain_ddp.*${experiment_name}" >/dev/null
}

gpu_is_idle() {
    local gpu_index="$1"
    local compute_pids memory_used utilization
    compute_pids="$(
        nvidia-smi -i "${gpu_index}" --query-compute-apps=pid --format=csv,noheader \
            2>/dev/null | sed '/^[[:space:]]*$/d'
    )"
    memory_used="$(
        nvidia-smi -i "${gpu_index}" --query-gpu=memory.used --format=csv,noheader,nounits
    )"
    utilization="$(
        nvidia-smi -i "${gpu_index}" --query-gpu=utilization.gpu --format=csv,noheader,nounits
    )"
    [[ -z "${compute_pids}" ]] \
        && (( memory_used < 2048 )) \
        && (( utilization < 10 ))
}

wait_for_all_gpus() {
    local busy
    while true; do
        busy=()
        for gpu_index in 0 1 2 3; do
            if ! gpu_is_idle "${gpu_index}"; then
                busy+=("${gpu_index}")
            fi
        done
        if (( ${#busy[@]} == 0 )); then
            echo "[$(date --iso-8601=seconds)] GPUs 0-3 are idle"
            return 0
        fi
        echo "[$(date --iso-8601=seconds)] waiting on GPU(s) ${busy[*]}"
        sleep 30
    done
}

completed_epochs() {
    if [[ ! -f "${checkpoint}" ]]; then
        echo 0
        return 0
    fi
    "${runtime}/bin/python" -c \
        "import torch; c=torch.load('${checkpoint}',map_location='cpu',weights_only=False,mmap=True); print(int(c['epoch'])+1)"
}

validate_checkpoint() {
    local path="$1" wanted="$2"
    "${runtime}/bin/python" -c \
        "import torch; p='${path}'; c=torch.load(p,map_location='cpu',weights_only=False,mmap=True); got=int(c['epoch'])+1; assert got==${wanted},(got,${wanted}); assert 'state_dict_ema' in c and 'optimizer' in c; print(f'validated {p}: {got} completed epochs')"
}

archive_checkpoint() {
    local wanted="$1" destination="$2"
    local temporary="${destination}.tmp"
    if [[ -f "${destination}" ]]; then
        validate_checkpoint "${destination}" "${wanted}"
        return 0
    fi
    validate_checkpoint "${checkpoint}" "${wanted}"
    cp --reflink=auto --preserve=timestamps "${checkpoint}" "${temporary}"
    mv "${temporary}" "${destination}"
    validate_checkpoint "${destination}" "${wanted}"
}

echo "[$(date --iso-8601=seconds)] direct VoxBind 60M supervisor armed: archive 350 -> train/archive 500"
while true; do
    while training_active; do
        echo "[$(date --iso-8601=seconds)] training active; monitoring"
        sleep 300
    done

    completed="$(completed_epochs 2>/dev/null || echo 0)"
    if ! [[ "${completed}" =~ ^[0-9]+$ ]]; then
        sleep "${retry_delay_seconds}"
        continue
    fi
    echo "[$(date --iso-8601=seconds)] durable progress: ${completed} completed epochs"

    if (( completed >= 350 )) && [[ ! -f "${checkpoint_350}" ]]; then
        if (( completed != 350 )); then
            echo "passed epoch 350 without milestone copy" >&2
            exit 1
        fi
        archive_checkpoint 350 "${checkpoint_350}"
    fi
    if (( completed >= 500 )); then
        if (( completed != 500 )); then
            echo "checkpoint passed epoch 500" >&2
            exit 1
        fi
        archive_checkpoint 500 "${checkpoint_500}"
        wait_for_all_gpus
        "${project_root}/scripts/sample_voxbind_60m_epoch500.sh"
        touch "${experiment_dir}/supervisor_epoch500.complete"
        echo "[$(date --iso-8601=seconds)] epoch-500 pipeline complete"
        exit 0
    fi

    target=350
    if (( completed >= 350 )); then
        target=500
    fi
    wait_for_all_gpus
    if training_active; then
        continue
    fi
    completed="$(completed_epochs 2>/dev/null || echo 0)"
    if ! [[ "${completed}" =~ ^[0-9]+$ ]]; then
        completed=0
    fi
    if (( completed >= target )); then
        continue
    fi

    if "${project_root}/scripts/run_voxbind_60m_segment.sh" \
        "${completed}" "${target}"; then
        echo "[$(date --iso-8601=seconds)] direct 60M segment to ${target} exited successfully"
    else
        status=$?
        echo "[$(date --iso-8601=seconds)] direct 60M segment failed with status ${status}; retrying"
        sleep "${retry_delay_seconds}"
    fi
done
