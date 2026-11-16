#!/bin/bash
#SBATCH --job-name=metl-export
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/metl_export_%j.out
#SBATCH --error=logs/metl_export_%j.err

set -euo pipefail

ROOT="$HOME/orcd/pool/biophysics-plm"
STAGE="$HOME/orcd/scratch/metl_export_${SLURM_JOB_ID}"
DEST="$ROOT/metl-global/exports/run_${SLURM_JOB_ID}"
PYTHON="$HOME/.conda/envs/metl-data/bin/python"

mkdir -p "$STAGE/results" "$DEST"

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

cp "$ROOT/metl-global/extracted/global/global.tsv" \
   "$STAGE/global.tsv"

"$PYTHON" "$ROOT/scripts/export_metl_dataset.py" \
  --tsv "$STAGE/global.tsv" \
  --out "$STAGE/results"

cp -a "$STAGE/results/." "$DEST/"
cp "$ROOT/metl-global/sample_manifest.csv" "$DEST/"

sha256sum "$ROOT/scripts/export_metl_dataset.py" \
  > "$DEST/script_checksum.txt"

touch "$DEST/SUCCESS"

echo "Completed: $(date)"
echo "Results: $DEST"

rm -rf "$STAGE"
