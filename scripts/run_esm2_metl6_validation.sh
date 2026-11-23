#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

output_dir="${OUTPUT_DIR:-results/esm2_metl6/full_validation}"

if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output directory: $output_dir" >&2
    echo "Inspect it first. For a deliberate clean rerun, remove it or invoke the Python script with --force." >&2
    exit 2
fi

export PYTHONUNBUFFERED=1

exec python scripts/validate_esm2_metl6_panel.py \
    --panel results/esm2_metl6/structure_selection/provisional_panel.tsv \
    --audit data/global/metl6_reference_pdb_rcsb/reference_audit.tsv \
    --coordinate-manifest data/global/metl6_reference_pdb_rcsb/coordinate_manifest.tsv \
    --db data/global/global.db \
    --tsv data/global/global.tsv \
    --split-dir data/global/splits/standard_tr0.9_tu0.05_te0.05_w2a93d88bac32_r2098 \
    --supertest-alias data/global/splits/supertest_w2a93d88bac32_s0.05_r6853.txt \
    --output "$output_dir"
