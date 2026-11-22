Dataset and methods

Data and protein selection

We used METL-Global (Gelman et al., 2025; Zenodo 14916528), which contains 29,227,460 variants from 148 reference proteins. From this dataset, I sampled 274,000 variants across eight proteins. The sampling seed was `20261114`.

I reconstructed the mutated sequences from the prepared PDB references and the recorded substitutions. The export was checked for sequence/row alignment, repeated row IDs, valid mutation positions, and finite Rosetta values. One detail that mattered was that the HDF5 physical row positions are not the same as the nonconsecutive labels on its row axis. We used the physical positions to access the correct records.

The six source proteins each contributed 40,000 training variants, 2,000 validation variants, and 2,000 within-protein test variants. The two excluded proteins each contributed 5,000 test variants. The full split was:

- Training: 240,000 variants.
- Validation: 12,000 variants.
- Within-protein test: 12,000 variants.
- Held-out-protein test: 10,000 variants.

| Reference | Length (aa) | Role | Structure/function description |
|---|---:|---|---|
| `1fna_A_p.pdb` | 91 | source | Fibronectin-like beta sandwich |
| `1r26_remod_p.pdb` | 125 | source | Thioredoxin, mixed alpha/beta |
| `1tzv_remod_p.pdb` | 142 | source | NusB, mainly alpha-helical |
| `1d1q_remod_p.pdb` | 161 | source | Tyrosine phosphatase, mixed alpha/beta |
| `1im5_remod_p.pdb` | 180 | source | Pyrazinamidase, mixed alpha/beta |
| `1f6b_remod_p.pdb` | 198 | source | SAR1 GTPase, mixed alpha/beta |
| `1i1j_remod_p.pdb` | 108 | held out | MIA, SH3-like beta barrel |
| `1xdz_remod_p.pdb` | 240 | held out | GidB, mixed alpha/beta |

We wanted both some variation within the source proteins and a pair of proteins that never contributed Rosetta gradients. We chose the holdout pair to cover two different broad structural descriptions. This was a practical eight-protein design, not a quantitatively fold-stratified sample. We didn't calculate structural similarity scores or verify that the holdouts are distant from all six training proteins.

Also, “held out” means excluded from our Rosetta fine-tuning. It doesn't mean their sequences were absent from EvoDiff's original UniRef pretraining.

Other splits we considered

We had proposed rotating the held-out pair to see how sensitive the results were to the choice of proteins. These runs have not been done.

| Split | Proposed holdout | Reason | Status |
|---|---|---|---|
| A | `1i1j`, `1xdz` | Two different broad architectures | Completed |
| B | `1fna`, `1tzv` | Beta-sandwich versus largely helical | Not run |
| C | `1d1q`, `1im5` | Mixed alpha/beta proteins with different functions | Not run |
| D | `1r26`, `1f6b` | Additional pair from the original pool | Not run |

Each rotation would also change which proteins are used for fine-tuning. The current export only includes source training/validation rows for the original six, so the alternative splits would require a new manifest and evaluation of their structural relationships. We cannot get a valid rotation just by changing the role labels.

Rosetta supervision

The initial pilot used 18 Rosetta terms. For the follow-up experiments, we divided them into three groups of six. The separation was meant to compare broad kinds of supervision, not to imply that the terms are statistically or physically independent.

| Group | Rosetta terms | Rough category |
|---|---|---|
| A | `fa_dun`, `omega`, `p_aa_pp`, `rama_prepro`, `pro_close`, `yhh_planarity` | Side-chain and backbone conformation |
| B | `fa_elec`, `fa_rep`, `hbond_bb_sc`, `hbond_lr_bb`, `hbond_sc`, `hbond_sr_bb` | Electrostatics, repulsion, hydrogen bonding |
| C | `fa_intra_rep`, `fa_intra_sol_xover4`, `lk_ball_wtd`, `cbeta`, `cenpack`, `env` | Packing, solvation, and local environment |

Stage I evaluated A, B, C, AB, BC, and ABC, with three seeds each (`20261116`, `20261117`, `20261118`). Single groups had six targets, pairs had twelve, and ABC had all eighteen. The unmodified pretrained EvoDiff encoder was the baseline.

Family D (`buried_all`, `buried_np`, `contact_all`, `degree`, `exposed_hydrophobics`, `ss_mis`) was proposed but not used in a completed run. The screen was not a full test of every possible target combination.

`fa_atr` and `fa_sol` were never used as fine-tuning targets in these experiments. We reserved them for the common probe tasks. The held-out protein evaluation also included `total_sasa`, `rg`, `pack`, and `unsat_hbond`. All six are calculated Rosetta properties and may be related to the supervised terms.

Each training target was standardized using only the training rows of its source protein. A target with near-zero variance in a particular protein was omitted from that protein's loss. Standardization makes the different numerical scales manageable; it doesn't mean the properties are equally useful or independent.

Model and training

We used pretrained EvoDiff `OA_DM_38M`. Our feature vector was made by taking `backbone.last_norm(backbone.embedder(tokens, None))` for a complete, unmasked sequence and averaging across residues. Each sequence gave a 1024-dimensional vector. This is the extraction procedure in our scripts; it isn't an established, separately validated EvoDiff embedding protocol.

The new regression head was:

`Linear(1024, 256) -> GELU -> Dropout(0.1) -> Linear(256, number_of_targets)`

For Stage I, the backbone and regression head were updated using AdamW. The main settings were:

- Encoder learning rate: `2e-5`; head learning rate: `2e-4`.
- Weight decay: `0.01`; gradient norm clipping: `0.5`.
- BF16 autocast, learning-rate warmup, and cosine decay.
- Maximum 40 epochs, with validation-based early stopping after at least 10 epochs.
- Three seeds for each of the six supervision conditions.

The earlier 18-target pilot ran for eight epochs and used a different stopping schedule. We treat it as an initial test of the training pipeline, not as one of the controlled Stage I replicates.

Evaluation

Within-protein probes (Stage I)

For each frozen encoder, a ridge regression probe predicted the reserved `fa_atr` and `fa_sol` values on variants of the six source proteins. We evaluated the original and updated EvoDiff encoders using the same setup. These are mostly close-mutant tests: 69.7% of test variants had a training neighbor at mutation distance one and 98.7% at distance two or less.

Zero-shot transfer (Stage II)

We fitted a ridge probe using labeled variants from the six source proteins, including source-protein validation data for choosing its regularization. We then applied it to `1i1j` and `1xdz` without using labels from those two proteins to train or select the probe. The encoder was frozen during this evaluation.

We reported per-protein Spearman correlation, which measures ranking agreement. The raw Rosetta energy scales and offsets are not calibrated across proteins.

Target-protein support experiments (Stage II)

We also trained a separate ridge probe for each held-out protein using 32, 128, or 512 labeled support variants. For each fit, a separate set of 500 labeled target-protein development variants was used to select the ridge regularization strength. The query variants were disjoint from support and development examples, although near-identical mutants could still occur across the sets.

So “32-shot”, “128-shot”, and “512-shot” in the older output refer only to the support examples. The effective labeled target-protein budget includes the additional 500 development labels. We should describe these as 32/128/512-support experiments instead of strict few-shot evaluations.

Masked-token prediction

We applied the same masking patterns to held-out sequences with mask fractions of 15%, 50%, and 80%. For each updated encoder, we attached EvoDiff's original frozen output head and measured negative log likelihood on masked residues. A higher value means the model is worse at recovering masked amino acids under this scoring setup.

This is a check of how well the changed encoder still works with the old output head. It isn't a direct test of generating functional proteins, and it doesn't isolate the cause of any loss increase.

Preservation-aware fine-tuning (Check-in 2)

We kept the ABC targets, the same source/holdout proteins, and the same evaluation examples and masking patterns. We changed the fine-tuning method:

- GENTLE: `2e-6` encoder learning rate, ten times smaller than before.
- L2SP: original encoder learning rate, with a coefficient of `50` on normalized squared displacement from pretrained weights.
- PARTIAL: frozen token embeddings and first 75% of ByteNet blocks; the remaining blocks, final normalization, and regression head were trainable.

Each method was trained with the same three seeds and the same training-target validation stopping rule. These were individual, prespecified settings, not hyperparameters selected from a larger search.

Limits on interpretation

We had already inspected results on `1i1j` and `1xdz` before the preservation runs. Comparisons on those proteins are therefore exploratory, not an independent held-out confirmation. Three seeds on the same two proteins tell us about run-to-run variation, not variation across protein folds. The proposed alternative protein rotations have not been completed. There are also no experimental ProteinGym results in this set of runs.
