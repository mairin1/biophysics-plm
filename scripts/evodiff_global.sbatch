#!/bin/bash
#SBATCH --job-name=evodiff-global
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/evodiff_global_%j.out
#SBATCH --error=logs/evodiff_global_%j.err

set -eo pipefail

cd "$HOME/orcd/pool/biophysics-plm"

module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run

SOURCE="metl-global/exports/run_29051193/sequences.csv.gz"
OUTPUT="metl-global/embeddings/evodiff_oa38m_frozen"

mkdir -p "$OUTPUT"

echo "Job: $SLURM_JOB_ID"
echo "Started: $(date)"

python scripts/extract_evodiff_global.py \
  --source "$SOURCE" \
  --out "$OUTPUT/smoke" \
  --limit 64

python scripts/extract_evodiff_global.py \
  --source "$SOURCE" \
  --out "$OUTPUT" \
  --batch-size 32

echo "Completed: $(date)"
