#!/bin/bash
#SBATCH --job-name=funcbind
#SBATCH --error ./sbatch_output/err.%J
#SBATCH --output ./sbatch_output/out.%J
#SBATCH --nodes 1
#SBATCH --cpus-per-task 24
#SBATCH --ntasks-per-node 8
#SBATCH --partition b200
#SBATCH --gpus 8
#SBATCH --mem-per-gpu=243G
#SBATCH -t 7-00:00

source ~/.bashrc
micromamba activate funcbind

export PYTHONFAULTHANDLER=1


srun python train_fb.py --config-name train_fb