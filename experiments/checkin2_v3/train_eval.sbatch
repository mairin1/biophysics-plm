#!/bin/bash
#SBATCH --job-name=evo-ci2-ft
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/evo_ci2_train_%A_%a.out
#SBATCH --error=logs/evo_ci2_train_%A_%a.err
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run
methods=(GENTLE L2SP PARTIAL)
seeds=(20261116 20261117 20261118)
method="${methods[SLURM_ARRAY_TASK_ID/3]}"
seed="${seeds[SLURM_ARRAY_TASK_ID%3]}"
echo "METHOD=$method SEED=$seed JOB=$SLURM_JOB_ID" 
python -u experiments/checkin2_v3/train.py --method "$method" --seed "$seed"
python -u experiments/checkin2_v3/evaluate.py --method "$method" --seed "$seed"
