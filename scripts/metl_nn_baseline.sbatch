#!/bin/bash
#SBATCH --job-name=metl-nn
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/metl_nn_%j.out
#SBATCH --error=logs/metl_nn_%j.err

set -euo pipefail

ROOT="$HOME/orcd/pool/biophysics-plm"
SOURCE="$ROOT/metl-global/exports/run_29051193"
STAGE="$HOME/orcd/scratch/metl_nn_${SLURM_JOB_ID}"
DEST="$ROOT/metl-global/analysis/nn_${SLURM_JOB_ID}"
PYTHON="$HOME/.conda/envs/metl-data/bin/python"

mkdir -p "$STAGE" "$DEST"

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

"$PYTHON" "$ROOT/scripts/metl_nn_baseline.py" \
  "$SOURCE" "$STAGE"

cp -a "$STAGE/." "$DEST/"
touch "$DEST/SUCCESS"

echo "Completed: $(date)"
echo "Results: $DEST"

rm -rf "$STAGE"
