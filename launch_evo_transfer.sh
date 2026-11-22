#!/bin/bash
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
OUT=metl-global/analysis/evodiff_transfer_20261119
mkdir -p logs "$OUT"
if [[ -f "$OUT/SUBMITTED" || -f "$OUT/PARTIAL_SUBMISSION" ]]; then
    echo "Submission record exists. Refusing duplicate batch."
    cat "$OUT/SUBMITTED" 2>/dev/null || cat "$OUT/PARTIAL_SUBMISSION"
    exit 0
fi
python3 -m py_compile scripts/transfer_common.py scripts/audit_transfer.py scripts/evaluate_transfer.py scripts/report_transfer.py
if [[ ! -f metl-global/analysis/evodiff_stage1_20261116/report/REPORT_READY ]]; then
    echo 'Stage I report not found; nothing submitted.' >&2
    exit 1
fi
audit=$(sbatch --parsable scripts/evo_transfer_audit.sbatch)
audit=${audit%%;*}
printf 'Audit: %s\n' "$audit" > "$OUT/PARTIAL_SUBMISSION"
gpu=$(sbatch --parsable --dependency="afterok:$audit" --array='0-18%2' scripts/evo_transfer_gpu.sbatch)
gpu=${gpu%%;*}
printf 'GPU array: %s\n' "$gpu" >> "$OUT/PARTIAL_SUBMISSION"
report=$(sbatch --parsable --dependency="afterany:$gpu" scripts/evo_transfer_report.sbatch)
report=${report%%;*}
cat > "$OUT/SUBMITTED" <<EOS
Audit: $audit
GPU array: $gpu (19 tasks, maximum 2 simultaneous)
Report: $report
Submission: $(date -Is)
EOS
rm "$OUT/PARTIAL_SUBMISSION"
cat "$OUT/SUBMITTED"
