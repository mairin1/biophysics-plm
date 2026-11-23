# ESM-2 biophysical supervision, evaluated on ProteinGym

It repeats the Check-in 2 design on a larger, different pretrained model and
scores the result on the ProteinGym v1.3 DMS substitution assays with a
per-assay supervised probe.

See [methods.md](methods.md) for the full protocol and its limitations.

## What is held fixed from Check-in 2

- The same 18 Rosetta targets (groups A + B + C), with `fa_atr` and `fa_sol`
  still reserved and unused as training targets.
- The same three preservation-aware fine-tuning methods: GENTLE, L2SP, PARTIAL.
- The same three seeds: `20261116`, `20261117`, `20261118`.
- The same METL-Global export and six source proteins.

## What changed

- **The encoder.** ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) replaces EvoDiff
  `OA_DM_38M`, roughly 17x more parameters, and a transformer instead of a
  ByteNet stack.
- **The evaluation.** Per-assay Spearman on ProteinGym DMS assays, rather than
  held-out-protein Rosetta probes.

ESM-2 was chosen over ESM-3 because the ESM-3 open checkpoint is gated behind a
licence acknowledgement and token. Note that Check-in 1 used actual ESM-3
embeddings (`embeddings/esm3_embeddings_9912.npz`), so those earlier numbers and
these are **not** the same model and should not be compared directly.

## Running it

One command submits the whole chain:

```bash
bash launch_esm2_pg.sh
```

That submits, with dependencies:

1. `prepare.sbatch` — downloads ProteinGym v1.3 (~1.0GB) and freezes the
   evaluation design (CPU).
2. `baseline_eval.sbatch` — scores the frozen pretrained ESM-2 (GPU).
3. `train_eval.sbatch` — a 9-task array, 3 methods x 3 seeds, each task
   fine-tuning then evaluating (GPU, max 3 concurrent).
4. `report.sbatch` — aggregates and writes the paired comparison (CPU).

Expect roughly 5-6 hours per array task on an L40S, so about 12 hours wall clock
at 3 concurrent tasks.

### Environment

The EvoDiff environment does not have `transformers`. Create a separate one:

```bash
module load miniforge/24.3.0-0
conda create -n esm2-run python=3.11 -y
conda activate esm2-run
pip install torch transformers numpy scikit-learn pandas
```

The sbatch scripts read `ESM2_ENV` (default `esm2-run`) and set `HF_HOME` to
`$HOME/orcd/pool/hf_cache` so the ~2.6GB checkpoint does not land on the home
quota.

If compute nodes have no outbound network, run the download on a login node
first; `prepare.sbatch` then skips straight to building the manifest:

```bash
python experiments/esm2_pg/fetch_proteingym.py
```

### Smoke test

**Every script resolves its paths from the repository root, so run them from
there, not from inside `experiments/esm2_pg/`.** The data must also exist first:

```bash
cd <repo root>
python experiments/esm2_pg/fetch_proteingym.py
python experiments/esm2_pg/build_eval_manifest.py
python experiments/esm2_pg/proteingym_eval.py --condition BASELINE \
    --allow-cpu --model facebook/esm2_t6_8M_UR50D --limit 2 --max-variants 100
```

**Do not smoke test the 650M model on CPU.** One forward pass costs roughly
`2 x 650M x n_tokens`, about 390 GFLOP for a 300-residue sequence, so a login
node needs *seconds per sequence*: even 200 sequences is a half-hour job that
looks like a hang. Pass `--model facebook/esm2_t6_8M_UR50D` instead. The 8M
checkpoint exercises every line of the real code path, takes a few seconds, and
is the only sensible way to check plumbing without a GPU. To time the real
encoder, get an interactive GPU shell (`srun --gpus=l40s:1 --pty bash`) rather
than using the login node.

Any of `--limit`, `--max-variants`, or a non-default `--model` marks the run a
smoke test: output goes to a `_smoke` directory, `status` is `SMOKE`, and no
`SUCCESS` marker is written, so a truncated run can never be mistaken for a
result or cause the real run to skip itself.

The evaluator prints a line per assay as it goes, so silence means it is
genuinely stuck rather than merely quiet.

## Output

Everything lands under `metl-global/analysis/esm2_pg_20261122/`:

| Path | Contents |
|---|---|
| `proteingym_manifest/` | The frozen variant subsample and fold assignment |
| `runs/<METHOD>_seed<SEED>/` | Checkpoint, history, training summary |
| `proteingym/<CONDITION>/per_assay_metrics.csv` | One Spearman per assay per split scheme |
| `report/summary.md` | Aggregates and paired change versus frozen ESM-2 |
| `report/paired_vs_baseline.csv` | Per-assay deltas, for figures |
