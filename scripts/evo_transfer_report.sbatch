#!/bin/bash
#SBATCH --job-name=evo-transfer-report
#SBATCH --partition=mit_normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/evo_transfer_report_%j.out
#SBATCH --error=logs/evo_transfer_report_%j.err
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run
python scripts/report_transfer.py
