# Reproducing the EvoDiff experiments

The purpose of this repository is to keep the **code, decisions, and lightweight evidence** together. The 29-million-row METL dataset, large model weights, embeddings, raw predictions, and optimizer states are not stored on GitHub.

## Where the runs happened

Engaging project root: `~/orcd/pool/biophysics-plm/`

| Experiment | Slurm IDs | Output directory under project root |
|---|---|---|
| METL-Global export | `29051193` | `metl-global/exports/run_29051193/` |
| Nearest-neighbor baseline | `29126077` | `metl-global/analysis/nn_29126077/` |
| Original 18-target pilot | `29288461` | `metl-global/analysis/evodiff_biophysics_29288461/` |
| Stage I audit / 19 GPU jobs / report | `29421154`, `29421155`, `29421156` | `metl-global/analysis/evodiff_stage1_20261116/` |
| Stage II audit / 19 evaluations / report | `29613275`, `29613276`, `29613277` | `metl-global/analysis/evodiff_transfer_20261119/` |
| Check-in 2 preflight / GPU check / 9 evaluations / report | `29720590`, `29720591`, `29720592`, `29720593` | `metl-global/analysis/evodiff_checkin2_20261119/` |

These directories contain the full configs, histories, audit outputs, report CSVs, and original figures. The `results/evodiff/` directories in this repository are curated lightweight copies, synced from those paths. We do **not** claim the text in this repo replaces the saved checkpoints or large source files.

## Source code checked into Git

- Stage I: `scripts/train_evodiff_stage1.py`, `scripts/audit_evodiff_stage1.py`, `scripts/summarize_evodiff_stage1.py`, and three corresponding `.sbatch` files; `launch_stage1.sh`.
- Stage II: `scripts/transfer_common.py`, `scripts/audit_transfer.py`, `scripts/evaluate_transfer.py`, `scripts/report_transfer.py`, and `.sbatch` files; `launch_evo_transfer.sh`.
- Check-in 2: `experiments/checkin2_v3/*.py`, its four `.sbatch` files, and `launch_checkin2_overnight.sh`.

The source scripts come from the experiment bundles. The Stage II launcher and audit script include the two report-marker path corrections that were applied on Engaging before the successful run. The import script compares these files against the copies currently on Engaging before committing. We did not rewrite training logic or figures and pretend the edited version had been executed.

## Source data / setup

- METL-Global: Zenodo record 14916528; prepared structures from `gitter-lab/metl-sim`.
- Original model: `microsoft/evodiff` `OA_DM_38M`, using the `evodiff-run` conda environment on Engaging.
- The job scripts require the original METL export, original EvoDiff package and checkpoint, and the paths they reference under the Engaging project root.
- Seeds and optimizer settings are listed in the executable scripts and stored JSON configs. The exact environment lockfile and upstream package commit hashes were **not captured in the materials available for this GitHub update**. That is a reproducibility gap, not something to guess.

## Artifacts we deliberately omit

Pretrained and fine-tuned `.pt` weights, optimizer states, `.npz` embeddings or predictions, the large METL HDF5 and sequence exports, `.npy` files, and full Slurm logs. These remain on the cluster. The publication sync excludes them; do not bypass that filter without checking size and licenses.

## Evaluation restrictions

The Stage II target-protein support probes use a separate target dev set of 500 labeled variants. Results labeled `32`, `128`, and `512` mean **support size**, not total target labels. The Stage II heldouts were inspected before Check-in 2 model selection. The source-protein and held-out Rosetta scores are simulated; the current work is not ProteinGym validation.
