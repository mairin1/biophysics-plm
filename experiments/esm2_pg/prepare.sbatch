#!/bin/bash
#SBATCH --job-name=esm2-pg-prep
#SBATCH --partition=mit_normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/esm2_pg_prep_%j.out
#SBATCH --error=logs/esm2_pg_prep_%j.err
set -eo pipefail
cd "${BIOPHYS_ROOT:-$HOME/biophysics-plm}"
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate "${ESM2_ENV:-esm2-run}"

# Downloads ~1.0GB of ProteinGym v1.3 substitution assays, then freezes the
# subsample and fold assignment that every encoder will be scored on.
# If compute nodes have no outbound network, run fetch_proteingym.py on a login
# node first; this script will then skip straight to the manifest.
python -u experiments/esm2_pg/fetch_proteingym.py
python -u experiments/esm2_pg/build_eval_manifest.py
