# Figures

These are the figures produced by the Engaging report scripts and copied here without redrawing them. Each has a corresponding CSV in the same stage's report directory. There are no manually invented error bars or reconstructed summary plots in this folder. The scripts that generated the images are committed alongside them.

## Before the transfer experiment

**Stage I: supervision ablation on the six source proteins**

![Paired Spearman improvements for six supervised conditions](../../results/evodiff/stage1/report/paired_probe_performance.png)

![Validation learning curves for each condition](../../results/evodiff/stage1/report/learning_curves.png)

**A note on these figures:** Stage I is about variants of the source proteins. The score differences are *not* evidence of transfer to proteins withheld from fine-tuning.

## Transfer to two other proteins

![Zero-shot transfer heatmap across targets and supervision conditions](../../results/evodiff/stage2/report/heatmap_zero_shot.png)

![Few-shot target-support heatmap](../../results/evodiff/stage2/report/heatmap_few_shot.png)

![Masked-token NLL for the first set of checkpoints](../../results/evodiff/stage2/report/retention.png)

**Note:** the 'few-shot' support budgets in the original plots refer to 32, 128, or 512 training variants **plus a separate 500-label target development set for probe selection**. See [methods.md](methods.md).

## Changes to the fine-tuning strategy

![Per-protein zero-shot comparison after preservation-aware fine-tuning](../../results/evodiff/checkin2/report/zero_shot_per_protein_heatmap.png)

![Effect of training strategies on masked-token NLL](../../results/evodiff/checkin2/report/masked_token_retention.png)

![Transfer and original-task compatibility tradeoff](../../results/evodiff/checkin2/report/transfer_retention_tradeoff.png)

![Training and validation curves](../../results/evodiff/checkin2/report/training_validation_curves.png)

For the full set of PNGs and raw values, see `results/evodiff/stage1/report/`, `results/evodiff/stage2/report/`, and `results/evodiff/checkin2/report/`.
