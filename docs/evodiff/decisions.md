Decisions and how the experiments developed

1. Starting point: frozen embeddings

For Check-in 1, we used around 10,000 avGFP variants and embeddings from ESM-3, ProGen2, and EvoDiff. The models themselves were frozen. We trained small regression heads to predict a Rosetta attraction-energy term, then tried adding `fa_sol` as an input. That helped the EvoDiff baseline in our initial test, but not the other two models.

This was not fine-tuning. Still, EvoDiff seemed worth investigating further. The question was whether biophysical supervision could change its representations in a useful way, rather than just help a regression head predict another Rosetta score.

2. Why we expanded the dataset

avGFP alone was too limited for the generalization question. We switched to METL-Global, which has about 29.2 million variants across 148 reference proteins. I downloaded the data, checked the row indexing and available Rosetta fields, reconstructed sequences using the reference structures, and exported a manageable subset.

We selected eight proteins with different lengths, broad structures, and functions. Six supplied the fine-tuning data. The other two, `1i1j` and `1xdz`, were kept out of Rosetta fine-tuning.

The idea was to include some structural variety instead of using variants from one protein throughout. `1i1j` is described as SH3-like/beta-barrel, while `1xdz` is mixed alpha/beta. The six source proteins also span beta-rich, helical, and mixed alpha/beta descriptions, with different functions. We chose them using those broad descriptions; we did not calculate pairwise TM-scores or otherwise measure structural distances. The eight proteins are not a representative sample of protein folds, and two holdouts cannot establish fold-wide generalization.

3. Why we added a nearest-neighbor baseline

Before doing more training, I wanted to know how difficult the within-protein test actually was. For each test sequence, I looked for the closest variant of the same protein in the training set, using mutation distance.

The test set turned out to be very local: 69.7% of test variants had a training neighbor one mutation away, and 98.7% were within two mutations. Copying the nearest neighbor's Rosetta score already gave an average test R² of 0.343 for `fa_atr` and 0.419 for `fa_sol`.

This changed how we interpreted the later results. A model can look good on those variants by learning local relationships around sequences it has effectively already seen. We therefore needed a separate test on proteins excluded from fine-tuning.

4. Why 18 attributes, and why A, B, and C

The first successful EvoDiff fine-tuning run used 18 Rosetta terms. They gave us a workable set covering several kinds of physical information. There was no evidence at that point that 18 was the best number or that those particular terms were optimal. The first run mainly showed that the encoder could be updated with this training setup.

Next, we wanted to know whether the type of supervision mattered. I kept the same 18 terms but grouped them into three sets of six:

- A, conformation: side-chain preferences, backbone geometry, and related local terms.
- B, interactions: electrostatics, steric repulsion, and hydrogen bonding.
- C, environment: intramolecular solvation, packing, and environment-related terms.

We first tested A, B, and C separately. Having six labels in each group made those single-family comparisons easier to interpret. We then tested AB and BC to see how adding a second group changed performance, and ABC to reproduce the complete 18-term setup using the same training protocol.

These groups are rough categories, not independent physical processes. Some terms overlap in what they measure. We also did not run every possible combination (for example, AC), so this was a targeted comparison rather than a complete factorial study. A fourth group, D, was proposed later but never trained.

`fa_atr` and `fa_sol` were excluded from all of these fine-tuning targets. We used them as common downstream probes so that every condition could be evaluated on the same tasks. That does not make them independent: Rosetta attributes can still be correlated with each other.

5. What Stage I told us, and why we didn't stop there

Stage I compared A, B, C, AB, BC, and ABC with three seeds each. It evaluated the resulting frozen embeddings on other Rosetta properties for the six source proteins.

B was the strongest individual group on the two main probes. ABC performed best overall among the tested combinations, although BC was close. The improvement was reasonably consistent across seeds.

But these results were still on proteins represented during fine-tuning. Given the nearest-neighbor distances, I didn't think we could take the gains as evidence of general transfer. That was the reason for Stage II.

6. Why we tested zero-shot and few-shot transfer

For Stage II, we evaluated the same saved encoders on `1i1j` and `1xdz` without fine-tuning the encoder on either protein.

The zero-shot setup trains a ridge regression probe on labeled variants of the six source proteins, chooses its regularization using source-protein validation data, then applies it to the two held-out proteins. It uses no Rosetta labels from the target protein while fitting or selecting that probe.

We also trained separate target-protein probes using 32, 128, or 512 labeled support variants. There is a qualification here: each of these experiments also used 500 labeled target-protein development variants to select the ridge penalty. So the numbers 32, 128, and 512 describe the training support size, not the full label budget. These are not strict N-shot experiments with only N target labels available.

The held-out results were much weaker than Stage I. Every supervised configuration had a negative mean zero-shot change in `fa_atr` Spearman correlation. Some `fa_sol` results improved, but not consistently across the two proteins. The smaller support sets occasionally favored fine-tuned representations, though the differences tended to shrink or reverse with larger support sets.

We also ran masked-token prediction with EvoDiff's original output head attached to the updated encoder. Its loss increased substantially after the standard ABC fine-tuning. That suggested that the new training might be disrupting the original sequence model. The test measures compatibility with that frozen head, though, so it isn't by itself proof of catastrophic forgetting or reduced generation quality.

7. Why we changed the fine-tuning method

At this point, changing the Rosetta targets again wasn't the most useful next step. We had evidence of stronger within-protein performance, weak cross-protein transfer, and a large increase in the original-task loss. I looked into the transfer-learning and continual-learning material from 6.7960 and related work, particularly methods that restrict updates to pretrained weights.

We kept the ABC targets, six source proteins, two holdouts, and evaluation setup fixed, then tried three changes to the training procedure:

- GENTLE: reduce the encoder learning rate from `2e-5` to `2e-6`.
- L2SP: add a penalty for moving the encoder weights away from their pretrained values.
- PARTIAL: freeze the token embeddings and the first 75% of ByteNet blocks; update the later blocks and the regression head.

Each method was run with three seeds, giving nine new training runs. We chose one setting per method for this comparison. We did not search over learning rates, penalty strengths, or fractions of frozen layers.

PARTIAL had the smallest increase in masked-token loss. GENTLE also preserved substantially more of the original score than standard ABC. PARTIAL showed the best average results with 32 support examples for the two main Rosetta probes, although the zero-shot `fa_atr` comparison was still negative. L2SP was less useful at the penalty strength we tried.

This suggests that how we update EvoDiff matters, but it hasn't resolved the generalization problem. The next question is whether any of these representations are useful for experimental sequence-function data.

8. Work considered but not completed

We had also considered changing which two proteins were held out. The proposed pairs were `1fna`/`1tzv` (beta-sandwich and mainly helical), `1d1q`/`1im5` (two mixed alpha/beta proteins with different functions), and optionally `1r26`/`1f6b`. The point was to check how sensitive transfer was to the source and target proteins, not to claim that these pairs were a rigorous measure of fold distance.

We haven't run those rotations. Our exported dataset includes training and validation partitions only for the original six source proteins, so we would need to rebuild the splits before changing their roles. Relabeling the existing rows would give an invalid comparison.

I have not run a ProteinGym evaluation as part of this work. Before reporting experimental results, I would need to verify the assays, mutant sequence reconstruction, model comparisons, and the label information available during evaluation.
