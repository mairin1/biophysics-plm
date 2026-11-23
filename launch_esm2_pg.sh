#!/bin/bash
set -eo pipefail
cd "${BIOPHYS_ROOT:-$HOME/biophysics-plm}"
BASE=metl-global/analysis/esm2_pg_20261122
SCRIPTS=experiments/esm2_pg
mkdir -p "$BASE" logs
if [[ -f "$BASE/SUBMITTED" || -f "$BASE/PARTIAL_SUBMISSION" ]]; then
    echo 'An existing submission record was found. No duplicate jobs submitted.'
    cat "$BASE/SUBMITTED" 2>/dev/null || cat "$BASE/PARTIAL_SUBMISSION"
    exit 0
fi
python3 -m py_compile "$SCRIPTS"/*.py
if [[ ! -f scripts/train_evodiff_stage1.py ]]; then
    echo 'Missing scripts/train_evodiff_stage1.py. Nothing submitted.' >&2
    exit 1
fi
# Confirms the METL export exists and is byte-identical to the data the
# EvoDiff Check-in 2 runs used, so the two sets of results stay comparable.
if ! python3 "$SCRIPTS/preflight.py"; then
    echo 'Export preflight failed. Nothing submitted.' >&2
    exit 1
fi

# Download ProteinGym and freeze the evaluation design.
prep=$(sbatch --parsable "$SCRIPTS/prepare.sbatch")
prep=${prep%%;*}
printf 'ProteinGym fetch + manifest: %s\n' "$prep" > "$BASE/PARTIAL_SUBMISSION"

# Frozen pretrained ESM-2 reference.
base=$(sbatch --parsable --dependency="afterok:$prep" "$SCRIPTS/baseline_eval.sbatch")
base=${base%%;*}
printf 'Baseline ProteinGym evaluation: %s\n' "$base" >> "$BASE/PARTIAL_SUBMISSION"

# Nine fine-tuning runs (3 methods x 3 seeds), each followed by its own
# ProteinGym evaluation. Three at a time to stay within the GPU allocation.
array=$(sbatch --parsable --dependency="afterok:$prep" --array='0-8%3' "$SCRIPTS/train_eval.sbatch")
array=${array%%;*}
printf 'Fine-tuning/evaluation array: %s\n' "$array" >> "$BASE/PARTIAL_SUBMISSION"

report=$(sbatch --parsable --dependency="afterany:$array:$base" "$SCRIPTS/report.sbatch")
report=${report%%;*}
{
  printf 'ProteinGym fetch + manifest: %s\n' "$prep"
  printf 'Baseline ProteinGym evaluation: %s\n' "$base"
  printf 'Fine-tuning/evaluation array: %s (9 tasks, max 3 simultaneous)\n' "$array"
  printf 'Final report: %s\n' "$report"
  printf 'Submission: %s\n' "$(date -Is)"
  printf 'Output: %s\n' "$BASE"
} > "$BASE/SUBMITTED"
rm "$BASE/PARTIAL_SUBMISSION"
cat "$BASE/SUBMITTED"
