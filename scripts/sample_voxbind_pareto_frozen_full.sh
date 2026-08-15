#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
voxbind_root="/home1/irteam/VoxBind/voxbind"
runtime="${project_root}/.repro-env"
experiment_name="voxbind_frozen_pareto100m_holo_xrayfull_20260728"
experiment_dir="${project_root}/exps/voxbind/${experiment_name}"
sample_dir="${experiment_dir}/samples_holo_density_target69"
density_dir="${voxbind_root}/dataset/data/xray_crops_aligned_v5"

if [[ ! -f "${experiment_dir}/checkpoint.pth.tar" ]]; then
    echo "trained checkpoint not found: ${experiment_dir}/checkpoint.pth.tar" >&2
    exit 1
fi

if [[ -f "${sample_dir}/target_69/samples.sdf" ]]; then
    echo "[$(date --iso-8601=seconds)] post-training target-69 samples already exist; skipping"
    exit 0
fi

mkdir -p "${sample_dir}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home1/irteam/VoxBind${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

echo "[$(date --iso-8601=seconds)] sampling test target 69 with real holo density"
cd "${voxbind_root}"
"${runtime}/bin/python" sample.py \
    --config-name=config_sample \
    dset=crossdocked_xray \
    hydra.job.chdir=false \
    hydra.run.dir="${sample_dir}" \
    pretrained_path="${experiment_dir}" \
    save_dir="${sample_dir}" \
    out_dir=holo_density_target69 \
    seed=1269 \
    num_workers=0 \
    dset.data_dir="${voxbind_root}/dataset/data" \
    dset.crops_dir="${density_dir}" \
    dset.normalize=false \
    dset.use_xray=true \
    dset.subset_n=null \
    dset.subset_xray_only=false \
    dset.subset_val_n=null \
    wjs.split=test \
    wjs.start=69 \
    wjs.end=69 \
    wjs.n_targets=100 \
    wjs.n_samples_per_pocket=10 \
    wjs.chain_init=denovo

sample_path="${sample_dir}/target_69/samples.sdf"
if [[ ! -f "${sample_path}" ]]; then
    echo "sampling completed without the expected SDF: ${sample_path}" >&2
    exit 1
fi
molecule_count="$(rg -c '^\$\$\$\$$' "${sample_path}")"
echo "[$(date --iso-8601=seconds)] target-69 sampling complete: ${molecule_count} molecules"
