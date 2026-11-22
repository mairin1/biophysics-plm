#!/bin/bash
cd "$HOME/orcd/pool/biophysics-plm" || exit 1
BASE=metl-global/analysis/evodiff_checkin2_20261119
if [[ -f "$BASE/SUBMITTED" ]]; then
    cat "$BASE/SUBMITTED"
else
    echo 'No submission record exists yet.'
fi
echo
echo '========== YOUR ACTIVE JOBS =========='
squeue -u "$USER" -o '%.18i %.22j %.12T %.12M %.20R'
echo
echo '========== EXPERIMENT STATUS =========='
if [[ -f "$BASE/report/run_status.csv" ]]; then cat "$BASE/report/run_status.csv"; else echo 'Report not generated yet.'; fi
if [[ -f "$BASE/report/summary.md" ]]; then
    echo
    cat "$BASE/report/summary.md"
fi
