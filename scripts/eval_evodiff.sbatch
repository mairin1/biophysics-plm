#!/bin/bash
#SBATCH --job-name=evodiff-pg-eval
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --output=logs/evodiff_pg_eval_%A_%a.out
#SBATCH --error=logs/evodiff_pg_eval_%A_%a.err

# Linear-probe evaluation of fine-tuned vs. pretrained EvoDiff (OA_DM_38M)
# mean-pooled embeddings on the ProteinGym assays in data/protein_gym/{indel,sub}.
# Each array task uses a different seed for the train/test split.
#
# Usage:
#   sbatch scripts/eval_evodiff.sbatch metl-global/analysis/evodiff_biophysics_<JOBID>
# The argument may be either the training output directory or best_model.pt itself.

set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate evodiff-run

RUN="${1:?Pass the fine-tuning run directory (or best_model.pt) as the first argument}"
if [[ -d "$RUN" ]]; then
    MODEL="$RUN/best_model.pt"
    OUT="$RUN/proteingym_eval"
else
    MODEL="$RUN"
    OUT="$(dirname "$RUN")/proteingym_eval"
fi
if [[ ! -f "$MODEL" ]]; then
    echo "Checkpoint not found: $MODEL" >&2
    exit 1
fi
if [[ -d "$RUN" && ! -f "$RUN/SUCCESS" ]]; then
    echo "Warning: $RUN has no SUCCESS marker; training may not have finished." >&2
fi

seeds=(20261116 20261117 20261118)
seed="${seeds[${SLURM_ARRAY_TASK_ID:-0}]}"
mkdir -p "$OUT"

echo "Model: $MODEL | seed: $seed | out: $OUT | job: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Started: $(date)"
python -u scripts/eval_evodiff.py --model "$MODEL" --out "$OUT" --seed "$seed"
echo "Completed: $(date)"
