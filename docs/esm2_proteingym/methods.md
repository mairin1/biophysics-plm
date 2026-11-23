# Methods: ESM-2 biophysical supervision on ProteinGym

## Model and feature extraction

We used ESM-2 650M, HuggingFace `facebook/esm2_t33_650M_UR50D`: 33 transformer
layers, hidden size 1280, and a positional limit of 1026 tokens, which leaves
1022 residues once `<cls>` and `<eos>` are added.

The feature vector for a sequence is the mean of the final hidden states over
residue positions only, with the two special tokens excluded from the average.
In the HuggingFace implementation `emb_layer_norm_after` is applied before
`last_hidden_state` is returned, so this pools already-normalised states. That
makes it the direct analogue of the EvoDiff extraction
(`last_norm(embedder(tokens, None))`, mean over residues), but it is our own
pooling choice, not a separately validated ESM-2 embedding protocol.

Each sequence gives a 1280-dimensional vector, against 1024 for EvoDiff.

## Rosetta fine-tuning

The training data, the per-protein target standardisation, and the batching are
reused unchanged from `scripts/train_evodiff_stage1.py` through a tokenizer
shim that exposes ESM-2 token ids under the `a_to_i` interface that loader
expects. The six source proteins, the 240,000 training rows, the 12,000
validation rows, and the 18 A+B+C targets are therefore identical to Check-in 2.
`fa_atr` and `fa_sol` remain excluded from training.

The regression head is `Linear(1280, 256) -> GELU -> Dropout(0.1) -> Linear(256, 18)`.

The three methods are the Check-in 2 methods, re-expressed for a transformer:

| Method | Change |
|---|---|
| GENTLE | Encoder learning rate `2e-6` instead of `2e-5` |
| L2SP | Standard learning rate, coefficient `50` on normalised squared displacement from the pretrained weights |
| PARTIAL | Token embeddings and the first 75% of the 33 transformer layers frozen; the last 9 layers, the final layer norm, and the head train |

For PARTIAL the run asserts at the end that no frozen parameter moved.

### Hyperparameters we did not re-tune

The encoder and head learning rates, the weight decay, the gradient clipping
threshold, the L2-SP coefficient, and the PARTIAL freezing fraction were all
inherited from the EvoDiff protocol. None was re-tuned for a model 17 times
larger. A value that preserved EvoDiff reasonably may be too aggressive or too
timid here, so a weak result for any one method is as consistent with a bad
inherited setting as with the method itself failing.

### Training schedule

A 650M-parameter encoder does not fit the EvoDiff batch size, so the per-step
batch is 16 with 4 gradient accumulation steps, keeping the effective batch at
64. The L2-SP penalty is a property of the weights rather than the batch, but it
is divided by the accumulation depth along with the data loss so that the
effective coefficient matches the un-accumulated case.

Training runs for at most 8 epochs with a minimum of 3 and patience 2, against
40 epochs for EvoDiff; the larger model needs fewer passes and the compute
budget is finite. Warmup is 5% of the planned steps, followed by cosine decay to
10% of the initial rate. Checkpoints are selected on minimum Rosetta validation
MSE over the six source proteins. No ProteinGym outcome is used for selection.

## ProteinGym evaluation

### Data

ProteinGym v1.3 DMS substitutions: 217 assays, about 2.47M variants, each with a
continuous `DMS_score`. The reference file is pinned from the OATML-Markslab
repository and the assay archive from the Harvard mirror; both are checksummed
into `fetch_manifest.json`.

This is deliberately not the clinical set already sitting in
`data/protein_gym/`, which carries only binarised `Pathogenic`/`Benign` labels.
Per-assay Spearman on a binary label is a monotone transform of AUROC and
carries no extra information, so continuous scores are required for the metric
to mean anything.

### Fixing the evaluation design before any model runs

`build_eval_manifest.py` decides, once, and writes the result to disk:

- **Subsample.** Assays are capped at 2,000 variants, drawn with seed
  `20261122`. Embedding all 2.47M variants once per model, for ten models, was
  not affordable; the cap brings it to about 355,000 per model.
- **Window.** 16 assays are longer than the 1022-residue limit. For those we
  choose the single window covering the most variants, computed exactly with a
  difference-array sweep over candidate starts, and drop variants whose
  mutations fall outside it. A fixed per-assay window keeps every variant in the
  assay comparable; a per-variant window would not.
- **Folds.** Both split schemes are assigned here.

Assays are skipped, with the reason recorded, if they have fewer than 100 usable
variants, if too few survive windowing, or if their scores are constant.

Because the manifest is shared, every encoder is scored on identical variants
and identical folds, so baseline-to-method differences are paired per assay.

### The probe

One ridge probe per assay, never pooled across assays. Features are standardised
using training-fold statistics only. The penalty is chosen from
`{1, 10, 100, 1000, 10000}` on an inner 20% split of the training folds, so the
held-out fold never informs the penalty. Five-fold cross-validation produces
out-of-fold predictions for every variant, and the reported Spearman is computed
once over the pooled out-of-fold predictions. The mean of the per-fold
correlations is also recorded.

Pooling across assays at fit time would let between-protein variation stand in
for within-protein ranking skill, which is what the superseded
`scripts/eval_evodiff.py` did.

### Two split schemes

| Scheme | Definition | What it does not control |
|---|---|---|
| `random` | Shuffled 5-fold | Near-identical variants of the same protein can land on both sides of a boundary |
| `modulo` | Fold = first mutated position mod 5 | For multi-mutant assays only the first position is pinned, so the remaining positions can still leak |

Both come from the same embeddings, so reporting both is nearly free. The
`random` numbers should be read as an upper bound. This distinction matters here
for the same reason the nearest-neighbour baseline mattered in Stage I: a local
test can be passed by memorising a neighbourhood.

### Statistics

For each method, seed, and scheme we take the per-assay difference against the
frozen pretrained ESM-2 and report the mean, the median, the fraction of assays
improved, and a 95% paired bootstrap interval over assays with 10,000
resamples. Assays are the unit that varies, so they are the unit resampled.

The intervals describe variation across assays for a single run. They do not
include seed-to-seed variation, which is reported separately as three rows.

## Limitations

- **Inherited hyperparameters.** As above: no setting was tuned for ESM-2.
- **Not a matched model comparison.** ESM-2 650M and EvoDiff 38M differ in size,
  architecture, pretraining corpus, and objective simultaneously. This
  experiment cannot attribute any difference to one of those.
- **Supervised probe only.** We do not report ProteinGym zero-shot scores. The
  standard masked-marginal zero-shot metric would be the natural analogue of the
  Check-in 2 retention test and is not yet implemented, so nothing here speaks
  to whether fine-tuning damaged the masked language model.
- **Subsampling.** Assays above 2,000 variants are evaluated on a subsample,
  which lowers precision for the largest assays relative to the published
  benchmark and makes our absolute numbers not directly comparable to
  leaderboard values.
- **Windowed assays.** The 16 long assays are scored on a truncated context.
- **Pretraining contamination.** ProteinGym proteins were not excluded from
  ESM-2's UniRef pretraining. "Held out" here means held out of our Rosetta
  supervision, nothing more.
- **Fold leakage.** Neither split scheme fully prevents near-duplicate leakage
  in multi-mutant assays.
