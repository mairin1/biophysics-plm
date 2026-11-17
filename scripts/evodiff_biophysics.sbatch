#!/bin/bash
#SBATCH --job-name=evodiff-biophys
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/evodiff_biophysics_%j.out
#SBATCH --error=logs/evodiff_biophysics_%j.err

set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run

SOURCE="metl-global/exports/run_29051193"
DEST="metl-global/analysis/evodiff_biophysics_${SLURM_JOB_ID}"
mkdir -p "$DEST"

echo "Smoke started: $(date)"
python scripts/train_evodiff_biophysics.py \
  --source "$SOURCE" \
  --out "$DEST/smoke" \
  --max-train-per-protein 128 \
  --max-val-per-protein 64 \
  --epochs 1 \
  --batch-size 16

echo "Full training started: $(date)"
python scripts/train_evodiff_biophysics.py \
  --source "$SOURCE" \
  --out "$DEST" \
  --epochs 8 \
  --batch-size 64

echo "Completed: $(date)"
