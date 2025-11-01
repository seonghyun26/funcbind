#!/bin/bash
#SBATCH --job-name=funcbind_omni
#SBATCH -q preempt
#SBATCH --error ./sbatch_output/err.%J
#SBATCH --output ./sbatch_output/out.%J
#SBATCH --nodes 1
#SBATCH --cpus-per-task 24
#SBATCH --ntasks-per-node 1
#SBATCH --partition b200
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=249G
#SBATCH -t 7-00:00

source ~/.bashrc
micromamba activate funcbind

export PYTHONFAULTHANDLER=1
export WANDB_INSECURE_DISABLE_SSL=true
