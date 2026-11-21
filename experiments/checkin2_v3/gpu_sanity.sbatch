#!/bin/bash
#SBATCH --job-name=evo-ci2-smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/evo_ci2_sanity_%j.out
#SBATCH --error=logs/evo_ci2_sanity_%j.err
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run
python -u experiments/checkin2_v3/gpu_sanity.py
