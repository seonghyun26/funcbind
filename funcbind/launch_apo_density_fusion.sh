#!/usr/bin/env bash
# Run the apo (ligand-erased) density control for the 5MGL holo-density POC.
#
# The job needs ~25 GiB of GPU memory: the FuncBind denoiser alone is 5.14 B
# parameters (19.1 GiB in fp32), plus the INR decoder, the VoxBind density
# encoder, and 8-chain sampling activations. It waits for a GPU with enough
# free memory rather than assuming one is idle, so it can share a card with a
# training run if MIN_FREE_MIB is lowered.
#
#   MIN_FREE_MIB=45000 ./funcbind/launch_apo_density_fusion.sh    # default
#   MIN_FREE_MIB=30000 MAX_UTIL=100 ./funcbind/launch_apo_density_fusion.sh
#   GPU_ID=2 SKIP_WAIT=1 ./funcbind/launch_apo_density_fusion.sh  # run now
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.repro-env/bin/python"
output_dir="${repo_root}/exps/density_fusion/apo_5mgl_target69_20260730"

min_free_mib="${MIN_FREE_MIB:-45000}"
max_util="${MAX_UTIL:-101}"
max_cgroup_percent="${MAX_CGROUP_PERCENT:-90}"
poll_seconds="${POLL_SECONDS:-60}"
gpu_id="${GPU_ID:-}"

mkdir -p "${output_dir}"
log="${output_dir}/run.log"

cgroup_percent() {
    local current max
    current="$(sed -n '1p' /sys/fs/cgroup/memory.current)"
    max="$(sed -n '1p' /sys/fs/cgroup/memory.max)"
    if [[ "${max}" == "max" ]]; then
        echo 0
    else
        # Page cache is reclaimable, so only anonymous memory is counted.
        local anon
        anon="$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat)"
        echo $((anon * 100 / max))
    fi
}

if [[ -z "${gpu_id}" && "${SKIP_WAIT:-0}" != "1" ]]; then
    echo "Waiting for a GPU with >= ${min_free_mib} MiB free and util <= ${max_util}%."
    while true; do
        chosen=""
        while IFS=, read -r index used total util; do
            index="${index// /}"; used="${used// /}"
            total="${total// /}"; util="${util// /}"
            free=$((total - used))
            echo "  GPU ${index}: ${free} MiB free, util ${util}%"
            if (( free >= min_free_mib && util <= max_util )) && [[ -z "${chosen}" ]]; then
                chosen="${index}"
            fi
        done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                     --format=csv,noheader,nounits)

        ram="$(cgroup_percent)"
        echo "  cgroup anonymous RAM: ${ram}%"
        if [[ -n "${chosen}" ]] && (( ram <= max_cgroup_percent )); then
            gpu_id="${chosen}"
            break
        fi
        sleep "${poll_seconds}"
    done
fi
gpu_id="${gpu_id:-0}"

echo "Launching apo-density fusion on GPU ${gpu_id}; log: ${log}"
(
    cd "${repo_root}/funcbind"
    exec env \
        PYTHONPATH="${repo_root}" \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        OMP_NUM_THREADS=4 \
        MKL_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 \
        NUMEXPR_NUM_THREADS=4 \
        TOKENIZERS_PARALLELISM=false \
        "${python_bin}" -u -m funcbind.holo_density_fusion \
        --config-name apo_density_fusion
) >"${log}" 2>&1

echo "Run finished; rebuilding the report."
(
    cd "${repo_root}"
    exec env PYTHONPATH="${repo_root}" \
        "${python_bin}" -u funcbind/notebook/build_holo_density_report.py
)
echo "Done: ${output_dir}/result.json"
