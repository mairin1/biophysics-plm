#!/bin/bash
#SBATCH --job-name=esm2-pg-ft
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/esm2_pg_train_%A_%a.out
#SBATCH --error=logs/esm2_pg_train_%A_%a.err
set -eo pipefail
cd "${BIOPHYS_ROOT:-$HOME/biophysics-plm}"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate "${ESM2_ENV:-esm2-run}"

# Shared checkpoint cache so the nine array tasks download ESM-2 (~2.6GB) once.
# Point HF_HOME elsewhere if the home quota is tight.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TOKENIZERS_PARALLELISM=false

methods=(GENTLE L2SP PARTIAL)
seeds=(20261116 20261117 20261118)
method="${methods[SLURM_ARRAY_TASK_ID/3]}"
seed="${seeds[SLURM_ARRAY_TASK_ID%3]}"
echo "METHOD=$method SEED=$seed JOB=$SLURM_JOB_ID"

python -u experiments/esm2_pg/train.py --method "$method" --seed "$seed"
python -u experiments/esm2_pg/proteingym_eval.py --condition "$method" --seed "$seed"
