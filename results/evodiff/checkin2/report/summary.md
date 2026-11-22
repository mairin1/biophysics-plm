# Check-in 2: preservation-aware EvoDiff fine-tuning

**New training and transfer evaluations complete:** 9/9
**Comparators:** original EvoDiff and original ABC fine-tuning (3 seeds, cached Stage II evaluation)

## Design

Same eight reference-protein pool, original six fine-tuning proteins, ABC 18 Rosetta attributes, and exactly the Stage II held-out evaluation manifest.
New methods: encoder LR 2e-6 (GENTLE), explicit L2-SP anchoring (L2SP), and early ByteNet block freezing (PARTIAL).
Each uses the same Rosetta validation stopping rule, matched seeds, probes, and masking patterns.

## Numeric summary

| Method | Target | Zero-shot ΔSpearman | Few-shot 32 ΔSpearman | Few-shot 512 ΔSpearman |
|---|---|---:|---:|---:|
| ABC | fa_atr | -0.0242 | +0.0153 | -0.0269 |
| ABC | fa_sol | +0.0323 | +0.0315 | -0.0488 |
| GENTLE | fa_atr | -0.0214 | +0.0218 | +0.0062 |
| GENTLE | fa_sol | +0.0381 | +0.0021 | -0.0078 |
| L2SP | fa_atr | -0.0376 | -0.0065 | -0.0363 |
| L2SP | fa_sol | +0.0023 | -0.0082 | -0.0575 |
| PARTIAL | fa_atr | -0.0573 | +0.0507 | +0.0044 |
| PARTIAL | fa_sol | +0.0557 | +0.0521 | -0.0088 |

## Retention changes

| Method | Mean Δ masked-token NLL (nats) |
|---|---:|
| ABC | +4.1674 |
| GENTLE | +1.0277 |
| L2SP | +3.4879 |
| PARTIAL | +0.6772 |

## Figures

- `zero_shot_fa_atr.png`
- `few_shot_fa_atr.png`
- `zero_shot_fa_sol.png`
- `few_shot_fa_sol.png`
- `zero_shot_per_protein_heatmap.png`
- `masked_token_retention.png`
- `transfer_retention_tradeoff.png`
- `training_validation_curves.png`

## Interpretation boundaries

- Positive zero-shot delta means a source-trained linear probe ranks Rosetta labels better for the two omitted proteins. It does not establish experimental function transfer.
- The original Stage II query set has already been viewed. This is exploratory testing, not a fresh held-out confirmation.
- The two proteins are not independent evidence for all protein folds. Variants may be very close sequence neighbors.
- All probes use Rosetta labels. Target properties are correlated, and score improvements do not imply mechanism learning.
- Masked-token NLL here reuses the frozen original OADM output head with the modified ByteNet components. It is a retention-compatibility diagnostic, not measured generation quality.
- Checkpoint selection uses the six training proteins only, not held-out transfer results.

**Plot errors:** none
