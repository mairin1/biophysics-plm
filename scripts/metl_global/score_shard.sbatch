#!/usr/bin/env bash
#SBATCH --job-name=metl-global
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/metl-global/%A_%a.out
#SBATCH --error=logs/metl-global/%A_%a.err

set -euo pipefail

: "${BASES:?Set BASES to the bases.tsv path}"
: "${TASKS:?Set TASKS to the mutation-shard task list}"
: "${OUT_DIR:?Set OUT_DIR to the score-output directory}"

# Replace with your cluster's environment setup.
# module load python
# source /path/to/venv/bin/activate

mkdir -p logs/metl-global

SHARD="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")"

if [[ -z "${SHARD}" ]]; then
    echo "No task at array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

python scripts/metl_global/score_shard.py \
    --bases "$BASES" \
    --variants "$SHARD" \
    --out-dir "$OUT_DIR" \
    --repack \
    --write-per-residue