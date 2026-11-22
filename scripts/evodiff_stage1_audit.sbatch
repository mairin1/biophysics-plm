#!/bin/bash
#SBATCH --job-name=evodiff-audit
#SBATCH --partition=mit_normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/evodiff_stage1_audit_%j.out
#SBATCH --error=logs/evodiff_stage1_audit_%j.err
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run
python scripts/audit_evodiff_stage1.py
