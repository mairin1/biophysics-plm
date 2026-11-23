#!/bin/bash
#SBATCH --job-name=esm2-pg-base
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/esm2_pg_baseline_%j.out
#SBATCH --error=logs/esm2_pg_baseline_%j.err
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate "${ESM2_ENV:-esm2-run}"

export HF_HOME="${HF_HOME:-$HOME/orcd/pool/hf_cache}"
export TOKENIZERS_PARALLELISM=false

# Frozen pretrained ESM-2: the reference every fine-tuned encoder is paired against.
python -u experiments/esm2_pg/proteingym_eval.py --condition BASELINE
