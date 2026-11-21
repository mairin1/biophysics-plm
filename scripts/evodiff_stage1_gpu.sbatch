#!/bin/bash
#SBATCH --job-name=evodiff-stage1
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/evodiff_stage1_%A_%a.out
#SBATCH --error=logs/evodiff_stage1_%A_%a.err
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run
conditions=(A B C AB BC ABC)
seeds=(20261116 20261117 20261118)
if (( SLURM_ARRAY_TASK_ID == 18 )); then
    condition=BASELINE
    seed=20261116
else
    condition="${conditions[SLURM_ARRAY_TASK_ID/3]}"
    seed="${seeds[SLURM_ARRAY_TASK_ID%3]}"
fi
printf 'Condition: %s, seed: %s, array task %s\n' "$condition" "$seed" "$SLURM_ARRAY_TASK_ID"
python scripts/train_evodiff_stage1.py --condition "$condition" --seed "$seed" --resume
