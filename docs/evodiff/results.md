Results up to November 20, 2026

The numbers below come from the completed Engaging reports. The per-run CSVs and original plots are saved under `results/evodiff/` when the repository results are synchronized.

The main comparison is ΔSpearman: the difference in rank correlation between a fine-tuned EvoDiff representation and the original pretrained EvoDiff representation, evaluated with matched probes. Positive values mean the fine-tuned representation performed better on that probe. All the targets reported here are calculated Rosetta properties.

1. Initial EvoDiff fine-tuning

The first 18-target run trained on 240,000 variants. Its lowest validation standardized MSE was 0.4582 at epoch 7, and the encoder weights changed by about 2.98% in relative norm.

This showed that the pipeline was running and updating the encoder. It did not tell us whether the updated representation generalized.

2. Stage I: choosing the Rosetta supervision

We compared six target configurations on the six source proteins, with three training seeds per configuration. After training, we froze each encoder and tested how well a ridge probe could predict `fa_atr` and `fa_sol`, neither of which was a fine-tuning target.

Mean difference in test Spearman correlation relative to pretrained EvoDiff:

| Fine-tuning targets | `fa_atr` | `fa_sol` |
|---|---:|---:|
| A (conformation) | +0.0392 | +0.0057 |
| B (interactions) | +0.0815 | +0.0941 |
| C (environment) | +0.0325 | +0.0399 |
| AB | +0.0962 | +0.1154 |
| BC | +0.1013 | +0.1266 |
| ABC | +0.1078 | +0.1335 |

B gave the largest improvement of the individual groups. ABC was the best of the six tested configurations, but only slightly better than BC on these two targets.

The improvement was fairly stable across seeds, but the test variants came from proteins represented during fine-tuning. Most of those variants had very close neighbors in the training data. We couldn't tell from these results whether the representations would transfer to another protein.

3. Stage II: transfer to two excluded proteins

We then tested `1i1j` and `1xdz`, which had not contributed data to the Rosetta fine-tuning. For zero-shot evaluation, the ridge probe was trained and selected using source-protein labels only.

Mean zero-shot ΔSpearman across the two proteins and three seeds:

| Fine-tuning targets | `fa_atr` | `fa_sol` |
|---|---:|---:|
| A | −0.0376 | −0.0189 |
| B | −0.0144 | +0.0042 |
| C | −0.0358 | +0.0280 |
| AB | −0.0765 | −0.0173 |
| BC | −0.0295 | +0.0185 |
| ABC | −0.0242 | +0.0323 |

Every condition was worse than pretrained EvoDiff on average for zero-shot `fa_atr`. The `fa_sol` results were mixed, with some small improvements.

The two proteins sometimes behaved differently. Under condition A, the mean `fa_sol` difference was about +0.086 for `1i1j` but −0.124 for `1xdz`. An average across only two proteins can hide differences like this, so it would be premature to generalize to other folds.

We also evaluated ridge probes trained on 32, 128, and 512 labeled variants from each target protein. Some fine-tuned encoders did better at the smallest support size, but the advantage usually became smaller or disappeared as support size increased. Each probe also used 500 other labeled target-protein development variants to select its ridge penalty. These results are not strict 32/128/512-label evaluations.

Finally, we measured masked-token prediction using the original EvoDiff output head with the fine-tuned encoder. The loss increased substantially after standard ABC fine-tuning. This was a reason to revisit the fine-tuning procedure, although it is a compatibility test with the original output head rather than a direct measure of generation quality.

4. Check-in 2: less aggressive fine-tuning

We kept the ABC labels and protein assignments fixed and tried three methods: a smaller encoder learning rate (GENTLE), a penalty on weight movement away from pretraining (L2SP), and freezing the first 75% of ByteNet blocks (PARTIAL). Each method was run with three seeds.

Mean differences relative to original pretrained EvoDiff:

| Method | Zero-shot `fa_atr` | Zero-shot `fa_sol` | 32-support `fa_atr` | 32-support `fa_sol` | Δ masked-token NLL |
|---|---:|---:|---:|---:|---:|
| ABC (earlier method) | −0.0242 | +0.0323 | +0.0153 | +0.0315 | +4.1674 |
| GENTLE | −0.0214 | +0.0381 | +0.0218 | +0.0021 | +1.0277 |
| L2SP | −0.0376 | +0.0023 | −0.0065 | −0.0082 | +3.4879 |
| PARTIAL | −0.0573 | +0.0557 | +0.0507 | +0.0521 | +0.6772 |

For the masked-token metric, smaller increases are better. PARTIAL had the smallest loss increase, followed by GENTLE. L2SP was much closer to the earlier ABC result at the one penalty strength we tested.

PARTIAL also had the largest average improvement at 32 support examples for `fa_atr` and `fa_sol`. That did not translate into consistently better zero-shot results: the mean zero-shot `fa_atr` difference remained negative for every method.

At 512 support examples, PARTIAL had differences of +0.0044 for `fa_atr` and −0.0088 for `fa_sol`. GENTLE gave +0.0062 and −0.0078. There is little evidence of a consistent advantage at this larger support size.

5. Current interpretation

The Rosetta-supervised encoder can improve prediction of other Rosetta properties for proteins included in fine-tuning. Which labels we use matters, and the interaction-related group contributed the most among the individual groups we tested.

We have not seen the same clear improvement when the source-trained probe is applied to two different proteins. The preservation experiments suggest that restricting parameter updates helps keep the encoder compatible with EvoDiff's original masked-token prediction head. PARTIAL also looks promising when a small target-protein support set is available, although those experiments use additional development labels and the result is limited to two proteins.

I would not claim that these results show improved general protein representations or experimental protein function. A separate evaluation on experimental data, such as ProteinGym, is still needed and has not been run as part of this work.

Limitations to keep in mind

- There are eight proteins in the dataset and only two proteins excluded from fine-tuning. We did not measure structural distances between them.
- Most within-protein test variants are close in sequence to training variants.
- All the reported probe labels come from Rosetta. They can be correlated with the terms used for fine-tuning.
- We had already inspected the two held-out protein results before running the preservation experiments. Those later comparisons are exploratory.
- The support-size probes used an additional 500 labeled development variants per target-protein setup.
- The masked-token loss uses the old output head with updated encoder components. It does not measure sequence generation directly.
- We have not run the proposed alternate protein-holdout rotations or completed an experimental DMS evaluation.
