#!/usr/bin/env bash
# Single entrypoint for frozen-density VoxBind training.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
voxbind_repository="${VOXBIND_REPOSITORY:-${VOXBIND_PYTHON_ROOT:-/home1/irteam/VoxBind}}"
voxbind_root="${VOXBIND_ROOT:-${voxbind_repository}/voxbind}"
runtime="${VOXBIND_RUNTIME:-${project_root}/.repro-env}"
model_zoo_root="${MODEL_ZOO_ROOT:-${voxbind_root}/model_zoo}"

usage() {
    cat <<'EOF'
Usage:
  train_voxbind.sh --model-zoo NAME [OPTIONS] [-- HYDRA_OVERRIDE ...]
  train_voxbind.sh --encoder-dir DIR [OPTIONS] [-- HYDRA_OVERRIDE ...]
  train_voxbind.sh --encoder-config CFG --encoder-checkpoint CKPT [OPTIONS]
  train_voxbind.sh --list-models [--model-zoo-root DIR]

Encoder:
  --model-zoo NAME         Folder under model_zoo, for example
                           champion_100m_v2_mask075.
  --model-zoo-root DIR     Model-zoo root. Default: VoxBind/model_zoo.
  --encoder-dir DIR        Folder containing cfg.yaml and checkpoint.
  --encoder-config PATH    Explicit model-zoo cfg.yaml.
  --encoder-checkpoint PATH
                           Explicit pretrained encoder checkpoint.
  --checkpoint-name NAME   Filename inside an encoder folder.
                           Default: checkpoint_e0049.pth.tar.
  --list-models            List local model-zoo entries and exit.

Run:
  --gpus CSV               Physical GPUs. Default: 0,1,2,3.
  --exp-name NAME          Defaults from the encoder folder name.
  --output-dir DIR         Default: VoxBind/exps/EXP_NAME.
  --epochs N               Epochs in this invocation. Default: 350.
  --completed-epochs N     Durable completed epochs; implies resume when N>0.
  --target-epoch N         Run TARGET-COMPLETED epochs.
  --resume                 Resume OUTPUT_DIR/checkpoint.pth.tar. Requires
                           --completed-epochs for an exact restart.
  --optimizer NAME         adamw (default) or muon.
  --muon-lr FLOAT          Hidden-weight Muon LR. Default: 0.02.
  --master-port PORT       torchrun port. Default: 29629.
  --preflight              Validate and print; do not launch.

Training:
  --batch-size N           Per-rank batch. Default: 32.
  --num-workers N          DataLoader workers per rank. Default: 16.
  --prefetch-factor N      Default: 8.
  --train-count N          Density-backed training rows. Default: 78512.
  --val-count N            Held-out validation rows. Default: 100.
  --lr FLOAT               AdamW/Aux-AdamW LR. Default: 1e-5.
  --weight-decay FLOAT     Default: 1e-2.
  --smooth-sigma FLOAT     Default: 0.9.
  --fusion NAME            Default: default.
  --density-mask-ligand BOOL
                           Default: false (holo-map experiment).
  --pyuul-chunks N         Muon-wrapper voxel chunks. Direct AdamW ignores it.

Logging:
  --wandb-mode MODE        online (default), offline, or disabled.
  --wandb-run-id ID        Resume a specific W&B run.
  --tags CSV               Additional W&B tags.

Examples:
  scripts/train_voxbind.sh --model-zoo champion_100m_v2_mask075 \
    --exp-name voxbind_champion_holo --gpus 0,1,2,3

  scripts/wait_for_gpus.sh --gpus 0,1,2,3 -- \
    scripts/train_voxbind.sh --model-zoo pareto_100m_v3_mask095 \
      --exp-name voxbind_pareto_holo --gpus 0,1,2,3

  scripts/train_voxbind.sh --model-zoo pareto_100m_v3_mask095 \
    --exp-name voxbind_pareto_holo --completed-epochs 350 \
    --target-epoch 500 --wandb-run-id RUN_ID
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

need_value() {
    (( $# >= 2 )) || die "$1 requires a value"
}

model_zoo_name=""
encoder_dir=""
encoder_config=""
encoder_checkpoint=""
checkpoint_name="checkpoint_e0049.pth.tar"
list_models=0
gpus="0,1,2,3"
experiment_name=""
output_dir=""
epochs=350
completed_epochs=0
completed_epochs_set=0
target_epoch=""
resume=0
optimizer="adamw"
muon_lr="0.02"
master_port=29629
preflight=0
batch_size=32
num_workers=16
prefetch_factor=8
train_count=78512
validation_count=100
learning_rate="1e-5"
weight_decay="1e-2"
smooth_sigma="0.9"
fusion="default"
density_mask_ligand="false"
pyuul_chunks=2
wandb_mode="online"
wandb_run_id=""
extra_tags=""
hydra_extra=()

while (( $# > 0 )); do
    case "$1" in
        --model-zoo)
            need_value "$@"; model_zoo_name="$2"; shift 2 ;;
        --model-zoo-root)
            need_value "$@"; model_zoo_root="$2"; shift 2 ;;
        --encoder-dir)
            need_value "$@"; encoder_dir="$2"; shift 2 ;;
        --encoder-config)
            need_value "$@"; encoder_config="$2"; shift 2 ;;
        --encoder-checkpoint)
            need_value "$@"; encoder_checkpoint="$2"; shift 2 ;;
        --checkpoint-name)
            need_value "$@"; checkpoint_name="$2"; shift 2 ;;
        --list-models)
            list_models=1; shift ;;
        --gpus)
            need_value "$@"; gpus="$2"; shift 2 ;;
        --exp-name)
            need_value "$@"; experiment_name="$2"; shift 2 ;;
        --output-dir)
            need_value "$@"; output_dir="$2"; shift 2 ;;
        --epochs)
            need_value "$@"; epochs="$2"; shift 2 ;;
        --completed-epochs)
            need_value "$@"; completed_epochs="$2"; completed_epochs_set=1; shift 2 ;;
        --target-epoch)
            need_value "$@"; target_epoch="$2"; shift 2 ;;
        --resume)
            resume=1; shift ;;
        --optimizer)
            need_value "$@"; optimizer="$2"; shift 2 ;;
        --muon-lr)
            need_value "$@"; muon_lr="$2"; shift 2 ;;
        --master-port)
            need_value "$@"; master_port="$2"; shift 2 ;;
        --preflight)
            preflight=1; shift ;;
        --batch-size)
            need_value "$@"; batch_size="$2"; shift 2 ;;
        --num-workers)
            need_value "$@"; num_workers="$2"; shift 2 ;;
        --prefetch-factor)
            need_value "$@"; prefetch_factor="$2"; shift 2 ;;
        --train-count)
            need_value "$@"; train_count="$2"; shift 2 ;;
        --val-count)
            need_value "$@"; validation_count="$2"; shift 2 ;;
        --lr)
            need_value "$@"; learning_rate="$2"; shift 2 ;;
        --weight-decay)
            need_value "$@"; weight_decay="$2"; shift 2 ;;
        --smooth-sigma)
            need_value "$@"; smooth_sigma="$2"; shift 2 ;;
        --fusion)
            need_value "$@"; fusion="$2"; shift 2 ;;
        --density-mask-ligand)
            need_value "$@"; density_mask_ligand="$2"; shift 2 ;;
        --pyuul-chunks)
            need_value "$@"; pyuul_chunks="$2"; shift 2 ;;
        --wandb-mode)
            need_value "$@"; wandb_mode="$2"; shift 2 ;;
        --wandb-run-id)
            need_value "$@"; wandb_run_id="$2"; shift 2 ;;
        --tags)
            need_value "$@"; extra_tags="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift; hydra_extra=("$@"); break ;;
        *)
            die "unknown argument: $1" ;;
    esac
done

if (( list_models )); then
    [[ -d "${model_zoo_root}" ]] || die "model-zoo root not found: ${model_zoo_root}"
    found=0
    for candidate in "${model_zoo_root}"/*; do
        [[ -d "${candidate}" ]] || continue
        if [[ -f "${candidate}/cfg.yaml" && -f "${candidate}/${checkpoint_name}" ]]; then
            printf '%s\n' "$(basename "${candidate}")"
            found=1
        fi
    done
    (( found )) || die "no model folders found under ${model_zoo_root}"
    exit 0
fi

selection_count=0
[[ -n "${model_zoo_name}" ]] && (( selection_count += 1 ))
[[ -n "${encoder_dir}" ]] && (( selection_count += 1 ))
if [[ -n "${encoder_config}" || -n "${encoder_checkpoint}" ]]; then
    [[ -n "${encoder_config}" && -n "${encoder_checkpoint}" ]] \
        || die "--encoder-config and --encoder-checkpoint must be supplied together"
    (( selection_count += 1 ))
fi
(( selection_count == 1 )) \
    || die "choose exactly one of --model-zoo, --encoder-dir, or explicit encoder paths"

if [[ -n "${model_zoo_name}" ]]; then
    [[ "${model_zoo_name}" != */* ]] \
        || die "--model-zoo must be a folder name, not a path"
    encoder_dir="${model_zoo_root}/${model_zoo_name}"
fi
if [[ -n "${encoder_dir}" ]]; then
    encoder_config="${encoder_dir}/cfg.yaml"
    encoder_checkpoint="${encoder_dir}/${checkpoint_name}"
fi

[[ -f "${encoder_config}" ]] || die "encoder config not found: ${encoder_config}"
[[ -f "${encoder_checkpoint}" ]] \
    || die "encoder checkpoint not found: ${encoder_checkpoint}"
[[ -x "${runtime}/bin/python" ]] || die "runtime Python not found: ${runtime}/bin/python"
[[ -x "${runtime}/bin/torchrun" ]] || die "torchrun not found: ${runtime}/bin/torchrun"
[[ -f "${voxbind_root}/train_ddp.py" ]] \
    || die "VoxBind trainer not found: ${voxbind_root}/train_ddp.py"

integer_fields=(
    "${epochs}" "${completed_epochs}" "${master_port}" "${batch_size}"
    "${num_workers}" "${prefetch_factor}" "${train_count}"
    "${validation_count}" "${pyuul_chunks}"
)
for value in "${integer_fields[@]}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || die "expected non-negative integer, got: ${value}"
done
(( epochs > 0 )) || die "--epochs must be positive"
(( master_port > 0 && master_port < 65536 )) || die "invalid --master-port"
(( batch_size > 0 )) || die "--batch-size must be positive"
(( num_workers > 0 )) || die "--num-workers must be positive"
(( prefetch_factor > 0 )) || die "--prefetch-factor must be positive"
(( pyuul_chunks > 0 )) || die "--pyuul-chunks must be positive"
[[ "${gpus}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid --gpus list: ${gpus}"
[[ "${density_mask_ligand}" == "true" || "${density_mask_ligand}" == "false" ]] \
    || die "--density-mask-ligand must be true or false"
case "${optimizer}" in adamw|muon) ;; *) die "--optimizer must be adamw or muon";; esac
case "${wandb_mode}" in online|offline|disabled) ;; *)
    die "--wandb-mode must be online, offline, or disabled";;
esac

IFS=',' read -r -a gpu_indices <<< "${gpus}"
nproc="${#gpu_indices[@]}"
(( nproc > 0 )) || die "at least one GPU is required"

if [[ -n "${target_epoch}" ]]; then
    [[ "${target_epoch}" =~ ^[0-9]+$ ]] || die "--target-epoch must be an integer"
    (( target_epoch > completed_epochs )) \
        || die "--target-epoch must exceed --completed-epochs"
    epochs=$(( target_epoch - completed_epochs ))
fi
if (( completed_epochs > 0 )); then
    resume=1
fi
if (( resume && ! completed_epochs_set )); then
    die "--resume requires --completed-epochs for exact epoch accounting"
fi

encoder_label="${model_zoo_name:-$(basename "$(dirname "${encoder_config}")")}"
[[ "${encoder_label}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "encoder folder names may contain only letters, numbers, ., _, and -"
if [[ -z "${experiment_name}" ]]; then
    experiment_name="voxbind_frozen_${encoder_label}_holo"
fi
[[ "${experiment_name}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "--exp-name may contain only letters, numbers, ., _, and -"
if [[ -z "${output_dir}" ]]; then
    output_dir="${voxbind_root}/exps/${experiment_name}"
fi

checkpoint="${output_dir}/checkpoint.pth.tar"
if (( resume )); then
    [[ -f "${checkpoint}" ]] || die "resume checkpoint not found: ${checkpoint}"
else
    [[ ! -e "${checkpoint}" ]] \
        || die "checkpoint already exists; use --resume or a new --output-dir"
fi

density_dir="${voxbind_root}/dataset/data/xray_crops_aligned_v5"
availability="${density_dir}/train_available.npy"
[[ -f "${availability}" ]] || die "density availability index not found: ${availability}"
available_count="$(
    "${runtime}/bin/python" -c \
        "import numpy as np; print(int(np.load(r'${availability}').sum()))"
)"
(( train_count + validation_count <= available_count )) || die \
    "requested ${train_count}+${validation_count} rows, only ${available_count} are density-backed"

mapfile -t encoder_hydra_args < <(
    "${runtime}/bin/python" "${project_root}/scripts/model_zoo_hydra_args.py" \
        --config "${encoder_config}"
)
(( ${#encoder_hydra_args[@]} > 0 )) \
    || die "encoder config produced no compatible Hydra overrides"

trainer_entry="${voxbind_root}/train_ddp.py"
if [[ "${optimizer}" == "muon" ]]; then
    trainer_entry="${project_root}/scripts/train_ddp_muon_entry.py"
fi

wandb_enabled=true
if [[ "${wandb_mode}" == "disabled" ]]; then
    wandb_enabled=false
fi
tags=(
    voxbind density_cond frozen_encoder model_zoo "${encoder_label}"
    holo full_density_crossdocked direct_voxbind_trainer
)
[[ "${optimizer}" == "muon" ]] && tags+=(muon)
if [[ -n "${extra_tags}" ]]; then
    IFS=',' read -r -a user_tags <<< "${extra_tags}"
    for tag in "${user_tags[@]}"; do
        [[ "${tag}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid W&B tag: ${tag}"
        tags+=("${tag}")
    done
fi
tag_string="$(IFS=','; echo "[${tags[*]}]")"

hydra_args=(
    "--config-path=${voxbind_root}/configs"
    --config-name=config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1
    hydra.job.chdir=false
    wandb="${wandb_enabled}"
    "wandb_tags=${tag_string}"
    num_workers="${num_workers}"
    prefetch_factor="${prefetch_factor}"
    exp_name="${experiment_name}"
    output_dir="${output_dir}"
    num_epochs="${epochs}"
    bsz="${batch_size}"
    accum_steps=1
    lr="${learning_rate}"
    wd="${weight_decay}"
    smooth_sigma="${smooth_sigma}"
    with_gradmag=false
    dset.crops_dir="${density_dir}"
    dset.normalize=false
    dset.pocket_radius=-1
    dset.ligand_radius=0.5
    dset.use_xray=true
    dset.subset_xray_only=true
    dset.subset_n="${train_count}"
    dset.subset_val_n="${validation_count}"
    dset.cache_size=32
    model.with_density=true
    model.density_encoder_type=vit
    model.density_freeze=true
    model.density_pretrained_path="${encoder_checkpoint}"
    "${encoder_hydra_args[@]}"
    model.density_mask_ligand="${density_mask_ligand}"
    model.fusion="${fusion}"
    wjs.n_targets=0
)
if (( resume )); then
    hydra_args+=(resume="${output_dir}" resume_epoch="${completed_epochs}")
fi
hydra_args+=("${hydra_extra[@]}")

command=(
    "${runtime}/bin/torchrun"
    --nproc_per_node="${nproc}"
    --master_addr=127.0.0.1
    --master_port="${master_port}"
    "${trainer_entry}"
    "${hydra_args[@]}"
)

echo "[$(date --iso-8601=seconds)] VoxBind training configuration"
echo "  encoder:    ${encoder_label}"
echo "  checkpoint: ${encoder_checkpoint}"
echo "  optimizer:  ${optimizer}"
echo "  GPUs:       ${gpus} (${nproc} ranks)"
echo "  epochs:     ${completed_epochs} -> $(( completed_epochs + epochs ))"
echo "  output:     ${output_dir}"
echo "  W&B:        ${wandb_mode}${wandb_run_id:+ (run ${wandb_run_id})}"
printf '  command:  '
printf ' %q' "${command[@]}"
printf '\n'

if (( preflight )); then
    echo "[$(date --iso-8601=seconds)] preflight complete; no training launched"
    exit 0
fi

mkdir -p "${output_dir}"
export CUDA_VISIBLE_DEVICES="${gpus}"
export PYTHONPATH="${voxbind_repository}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${wandb_mode}"
export WANDB_DIR="${output_dir}"
export WANDB__SERVICE_WAIT=300
if [[ -n "${wandb_run_id}" ]]; then
    export WANDB_RUN_ID="${wandb_run_id}"
    export WANDB_RESUME=allow
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VOXBIND_PYUUL_CHUNKS="${pyuul_chunks}"
export VOXBIND_MUON_LR="${muon_lr}"

cd "${voxbind_root}"
exec "${command[@]}"
