# EvoDiff Stage I: first screening batch

**Completed jobs:** 19 / 19

Six known-target configurations: A, B, C, AB, BC, ABC; three seeds each, plus one pretrained baseline.

The primary output here is the *paired difference versus frozen pretrained EvoDiff* on common reserved Rosetta properties.
Use validation probe performance for provisional decisions; inspect test results descriptively, not for choosing winners.
These are within-protein tests with close-mutant overlap; they do NOT establish transfer to unseen proteins or experiments.
This batch is an initial screening subset, not the complete planned ten-condition matrix.

## Files

- `run_status.csv`: which GPU experiments succeeded
- `all_probe_metrics.csv`: every protein/seed/target/split metric
- `paired_differences.csv`: matched model-minus-pretrained comparisons
- `mean_paired_differences.csv`: per-seed means across proteins
- `paired_probe_performance.png`: common-task comparison
- `learning_curves.png`: training and validation objectives
- `checkpoint_epochs.png`: training lengths

**Plot errors:** none

## Next gates

Audit the D family and construct a balanced eight-protein dataset before rotated protein-fold experiments.
Evaluate EvoDiff original masked-sequence likelihood before declaring pretrained ability retained.
Experimental ProteinGym assays remain separate and pending team agreement.
