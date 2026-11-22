#!/bin/bash
set -eo pipefail
cd "$HOME/orcd/pool/biophysics-plm"
mkdir -p logs metl-global/analysis/evodiff_stage1_20261116
OUT=metl-global/analysis/evodiff_stage1_20261116
if [[ -f "$OUT/SUBMITTED" || -f "$OUT/PARTIAL_SUBMISSION" ]]; then
    echo "Submission record already exists; not launching a duplicate sweep."
    cat "$OUT"/{SUBMITTED,PARTIAL_SUBMISSION} 2>/dev/null || true
    exit 0
fi
python3 -m py_compile scripts/audit_evodiff_stage1.py scripts/train_evodiff_stage1.py scripts/summarize_evodiff_stage1.py
audit=$(sbatch --parsable scripts/evodiff_stage1_audit.sbatch)
audit=${audit%%;*}
printf "Audit submitted: %s\n" "$audit" > "$OUT/PARTIAL_SUBMISSION"
gpu=$(sbatch --parsable --dependency="afterok:$audit" --array='0-18%2' scripts/evodiff_stage1_gpu.sbatch)
gpu=${gpu%%;*}
printf "GPU array submitted: %s\n" "$gpu" >> "$OUT/PARTIAL_SUBMISSION"
report=$(sbatch --parsable --dependency="afterany:$gpu" scripts/evodiff_stage1_report.sbatch)
report=${report%%;*}
cat > "$OUT/SUBMITTED" <<EOF
Audit: $audit
GPU array: $gpu (19 tasks, max 2 simultaneous)
Report: $report
Submit time: $(date -Is)
EOF
rm "$OUT/PARTIAL_SUBMISSION"
cat "$OUT/SUBMITTED"
echo "Once everything finishes, inspect: $OUT/report/summary.md"
echo "If the audit fails, the dependent GPU jobs will not start."
