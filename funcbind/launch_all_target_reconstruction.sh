#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.repro-env/bin/python"
output_root="${repo_root}/exps/decoder_ablation/all_targets_mcpp_20260728"
inr_output="${output_root}/inr_gpu2"
gaussian_output="${output_root}/gaussian_splat_gpu3"

mkdir -p "${inr_output}" "${gaussian_output}"

idle_memory_mib=2048
idle_utilization_pct=5
required_idle_checks=2
poll_seconds=20
idle_checks=0

echo "Waiting for physical GPUs 2 and 3 to be idle."
echo "Outputs will be written under ${output_root}."

while true; do
    read -r gpu2_memory gpu2_utilization < <(
        nvidia-smi --id=2 \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits |
            tr -d ',' 
    )
    read -r gpu3_memory gpu3_utilization < <(
        nvidia-smi --id=3 \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits |
            tr -d ','
    )
    cgroup_current="$(sed -n '1p' /sys/fs/cgroup/memory.current)"
    cgroup_max="$(sed -n '1p' /sys/fs/cgroup/memory.max)"
    if [[ "${cgroup_max}" == "max" ]]; then
        cgroup_percent=0
    else
        cgroup_percent=$((cgroup_current * 100 / cgroup_max))
    fi

    echo "GPU2 ${gpu2_memory} MiB/${gpu2_utilization}%; GPU3 ${gpu3_memory} MiB/${gpu3_utilization}%; cgroup RAM ${cgroup_percent}%."

    if (( gpu2_memory <= idle_memory_mib &&
          gpu3_memory <= idle_memory_mib &&
          gpu2_utilization <= idle_utilization_pct &&
          gpu3_utilization <= idle_utilization_pct &&
          cgroup_percent <= 65 )); then
        idle_checks=$((idle_checks + 1))
    else
        idle_checks=0
    fi

    if (( idle_checks >= required_idle_checks )); then
        break
    fi
    sleep "${poll_seconds}"
done

echo "GPUs 2 and 3 are stably idle; launching both reconstruction jobs."

(
    cd "${repo_root}"
    exec env \
        PYTHONPATH="${repo_root}" \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES=2 \
        OMP_NUM_THREADS=4 \
        MKL_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 \
        NUMEXPR_NUM_THREADS=4 \
        TOKENIZERS_PARALLELISM=false \
        "${python_bin}" -u -m funcbind.reconstruct_all_targets \
        variant=inr \
        output_dir="${inr_output}" \
        hydra.run.dir="${inr_output}"
) >"${inr_output}/run.log" 2>&1 &
inr_pid=$!

(
    cd "${repo_root}"
    exec env \
        PYTHONPATH="${repo_root}" \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES=3 \
        OMP_NUM_THREADS=4 \
        MKL_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 \
        NUMEXPR_NUM_THREADS=4 \
        TOKENIZERS_PARALLELISM=false \
        "${python_bin}" -u -m funcbind.reconstruct_all_targets \
        variant=gaussian_splat \
        output_dir="${gaussian_output}" \
        hydra.run.dir="${gaussian_output}"
) >"${gaussian_output}/run.log" 2>&1 &
gaussian_pid=$!

echo "INR PID ${inr_pid} on GPU 2."
echo "Gaussian PID ${gaussian_pid} on GPU 3."

set +e
wait "${inr_pid}"
inr_status=$?
wait "${gaussian_pid}"
gaussian_status=$?
set -e

echo "INR exit status: ${inr_status}."
echo "Gaussian exit status: ${gaussian_status}."

if (( inr_status != 0 || gaussian_status != 0 )); then
    exit 1
fi
