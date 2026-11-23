#!/usr/bin/env bash
set -euo pipefail

BASES="${BASES:-data/metl_global/bases.tsv}"
VARIANT_DIR="${VARIANT_DIR:-work/metl_global/variant_shards}"
OUT_DIR="${OUT_DIR:-data/metl_global/scores}"
SEED="${SEED:-20240315}"
MAX_CONCURRENT="${MAX_CONCURRENT:-100}"

mkdir -p work/metl_global logs/metl-global

python scripts/metl_global/make_variants.py \
    --bases "$BASES" \
    --out-dir "$VARIANT_DIR" \
    --n-per-protein 200000 \
    --max-mutations 5 \
    --shard-size 1000 \
    --seed "$SEED"

TASKS="$VARIANT_DIR/tasks.txt"
N_TASKS="$(wc -l < "$TASKS")"

if [[ "$N_TASKS" -eq 0 ]]; then
    echo "No scoring tasks were created." >&2
    exit 1
fi

sbatch \
    --array="0-$((N_TASKS - 1))%${MAX_CONCURRENT}" \
    --export="ALL,BASES=${BASES},TASKS=${TASKS},OUT_DIR=${OUT_DIR}" \
    scripts/metl_global/score_shard.sbatch