#!/bin/bash
#SBATCH --job-name=evo-transfer-audit
#SBATCH --partition=mit_normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/evo_transfer_audit_%j.out
#SBATCH --error=logs/evo_transfer_audit_%j.err
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run
python scripts/audit_transfer.py
