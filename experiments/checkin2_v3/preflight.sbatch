#!/bin/bash
#SBATCH --job-name=evo-ci2-audit
#SBATCH --partition=mit_normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/evo_ci2_audit_%j.out
#SBATCH --error=logs/evo_ci2_audit_%j.err
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run
python -u experiments/checkin2_v3/preflight.py
