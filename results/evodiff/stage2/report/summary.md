# EvoDiff Stage II exploratory transfer evaluation

**Transfer jobs complete:** 19/19
**Retention evaluated:** 19/19

## Scope

This compares original pretrained EvoDiff with six Rosetta-supervised conditions (three seeds each) on the two proteins omitted from Stage I encoder fine-tuning.
Primary zero-shot result: a ridge probe fitted to source-protein labels and evaluated without any held-out protein labels. Spearman only; raw energy offsets are not calibrated.
Few-shot result: independent support, dev, and query subsets of held-out protein variants. Alpha selection uses dev only, query is untouched.
The base EvoDiff checkpoint may have seen related natural sequences in its original UniRef pretraining; "unseen" refers to the Rosetta-supervised stage.
This is exploratory because the user has previously viewed some held-out data, and it is not an independent experimental validation.
Retention uses the original pretrained OADM output head with updated encoder components. NLL compares identical mask patterns and sequences.

## Results tables

- `run_status.csv`: every condition, completion and retention status
- `paired_zero_shot.csv`: per-protein and per-target paired differences
- `paired_few_shot.csv`: paired support-budget results
- `paired_retention.csv`: pretrained OADM head compatibility
- `per_protein_aggregates.csv`: averages across encoder seeds, not uncertainty across independent proteins

## Figures

- `zero_shot_fa_atr.png`
- `few_shot_fa_atr.png`
- `zero_shot_fa_sol.png`
- `few_shot_fa_sol.png`
- `heatmap_zero_shot.png`
- `heatmap_few_shot.png`
- `retention.png`

## Interpretation boundaries

The two held-out proteins are insufficient to estimate performance across protein fold space.
Stage I variants are close mutants, and held-out support/query data may also have very close sequence neighbors; only exact duplicates were removed.
Six Rosetta diagnostic properties are correlated; do not treat their metrics as independent observations.
No ProteinGym data were used; no experimental sequence-function improvement is established.
A successful GPU job with a failed retention diagnostic counts only as transfer-complete.

**Plot errors:** none
