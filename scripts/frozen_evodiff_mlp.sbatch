#!/bin/bash
#SBATCH --job-name=evodiff-mlp
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/evodiff_mlp_%j.out
#SBATCH --error=logs/evodiff_mlp_%j.err

set -eo pipefail

cd "$HOME/orcd/pool/biophysics-plm"

module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run

SOURCE="metl-global/exports/run_29051193"
EMBEDDINGS="metl-global/embeddings/evodiff_oa38m_frozen"
OUTPUT="metl-global/analysis/evodiff_frozen_mlp_${SLURM_JOB_ID}"

echo "Started: $(date)"

python scripts/train_frozen_evodiff.py \
  --source "$SOURCE" \
  --embeddings "$EMBEDDINGS" \
  --out "$OUTPUT"

echo "Completed: $(date)"
