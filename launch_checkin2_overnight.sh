#!/bin/bash
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
BASE=metl-global/analysis/evodiff_checkin2_20261119
SCRIPTS=experiments/checkin2_v3
mkdir -p "$BASE" logs
if [[ -f "$BASE/SUBMITTED" || -f "$BASE/PARTIAL_SUBMISSION" ]]; then
    echo 'An existing submission record was found. No duplicate jobs submitted.'
    cat "$BASE/SUBMITTED" 2>/dev/null || cat "$BASE/PARTIAL_SUBMISSION"
    exit 0
fi
python3 -m py_compile "$SCRIPTS"/*.py
for required in \
  metl-global/analysis/evodiff_stage1_20261116/report/REPORT_READY \
  metl-global/analysis/evodiff_transfer_20261119/report/TRANSFER_COMPLETE \
  metl-global/analysis/evodiff_transfer_20261119/evaluation_manifest.json \
  scripts/train_evodiff_stage1.py \
  scripts/transfer_common.py \
  scripts/evaluate_transfer.py; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required artifact: $required. Nothing submitted." >&2
        exit 1
    fi
done
audit=$(sbatch --parsable "$SCRIPTS/preflight.sbatch")
audit=${audit%%;*}
printf 'CPU audit: %s\n' "$audit" > "$BASE/PARTIAL_SUBMISSION"
sanity=$(sbatch --parsable --dependency="afterok:$audit" "$SCRIPTS/gpu_sanity.sbatch")
sanity=${sanity%%;*}
printf 'GPU retention check: %s\n' "$sanity" >> "$BASE/PARTIAL_SUBMISSION"
array=$(sbatch --parsable --dependency="afterok:$sanity" --array='0-8%3' "$SCRIPTS/train_eval.sbatch")
array=${array%%;*}
printf 'Fine-tuning/evaluation array: %s\n' "$array" >> "$BASE/PARTIAL_SUBMISSION"
report=$(sbatch --parsable --dependency="afterany:$array" "$SCRIPTS/report.sbatch")
report=${report%%;*}
{
  printf 'CPU audit: %s\n' "$audit"
  printf 'GPU retention check: %s\n' "$sanity"
  printf 'Fine-tuning/evaluation array: %s (9 tasks, max 3 simultaneous)\n' "$array"
  printf 'Final report: %s\n' "$report"
  printf 'Submission: %s\n' "$(date -Is)"
  printf 'Output: %s\n' "$BASE"
} > "$BASE/SUBMITTED"
rm "$BASE/PARTIAL_SUBMISSION"
cat "$BASE/SUBMITTED"
