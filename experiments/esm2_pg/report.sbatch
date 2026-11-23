#!/bin/bash
#SBATCH --job-name=esm2-pg-report
#SBATCH --partition=mit_normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/esm2_pg_report_%j.out
#SBATCH --error=logs/esm2_pg_report_%j.err
set -eo pipefail
cd "${BIOPHYS_ROOT:-$HOME/biophysics-plm}"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate "${ESM2_ENV:-esm2-run}"

python -u experiments/esm2_pg/report.py
