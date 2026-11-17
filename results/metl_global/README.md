# METL-Global: Dataset Preparation and EvoDiff Experiments

This is a record of what I've worked on since Check-in 1

The main question we're interested in is whether fine-tuning a pretrained protein model on biophysical information can improve its representations, especially for downstream tasks involving experimentally measured protein properties.

Most of the computations were done on MIT Engaging. I'm keeping the code, metadata, and smaller results here, while the full datasets, embeddings, and checkpoints remain on Engaging.

## 1. Preparing METL-Global

I downloaded the processed METL-Global dataset from Zenodo (record 14916528). It contains approximately 29.2 million variants from 148 reference proteins.

I selected six proteins for post-training and two completely held-out proteins for evaluating cross-protein generalization. I considered sequence lengths, structural diversity, and sequence similarity when choosing them.

I reconstructed the variant sequences from the prepared reference PDB structures and mutation annotations.

One important detail: the official split files use physical dataset row positions, not the nonconsecutive index labels stored in the HDF5 file. I verified that the official splits form a complete, nonoverlapping partition before sampling.

The final dataset contains 274,000 variants:

| Split | Variants |
|---|---:|
| Training | 240,000 |
| Validation | 12,000 |
| Within-protein test | 12,000 |
| Held-out-protein test | 10,000 |

Sampling seed: 20261114.

The export includes 60 numerical properties, although not all are appropriate for biophysical supervision.

All exported sequence and label rows passed alignment checks. There were no duplicate row positions or reported nonfinite numerical values.

## 2. Nearest-neighbor baseline

I first wanted to see how well we could predict Rosetta properties using sequence similarity alone.

For each validation/test variant, I found the closest training variant from the same reference protein using amino-acid Hamming distance, then used that variant's property values as predictions.

Mean within-protein test R2:

| Property | R2 |
|---|---:|
| fa_atr | 0.343 |
| fa_sol | 0.419 |

One thing worth noting: around 70% of the evaluation variants have a training neighbor just one mutation away, and almost 99% are within two mutations.

This means our within-protein splits mostly measure local interpolation, rather than generalization to substantially different sequences or proteins.

## 3. Frozen EvoDiff embeddings

I extracted embeddings for all 274,000 variants using pretrained EvoDiff OA_DM_38M.

EvoDiff doesn't have a canonical sequence embedding, so our implementation takes the final normalized ByteNet hidden states from a fully observed sequence and mean-pools them across residues.

This gives a 1,024-dimensional representation per variant. No masking or explicit diffusion timestep was used during extraction.

The embeddings were saved in 70 compressed files. I verified their shapes, row IDs, protein assignments, splits, and finite values against the original sample manifest.

## 4. Initial frozen-encoder regression

I trained an MLP on the frozen EvoDiff embeddings to predict fa_atr and fa_sol.

Importantly, this experiment did not update EvoDiff's pretrained weights. It was just an initial representation baseline.

The best checkpoint was selected at epoch 13 using validation loss.

Mean within-protein test R2:

| Property | Nearest neighbor | Frozen EvoDiff + MLP |
|---|---:|---:|
| fa_atr | 0.343 | 0.309 |
| fa_sol | 0.419 | 0.465 |

The model performed very poorly on the two held-out proteins, with strongly negative R2 values.

One possible explanation is that our current regression setup doesn't adequately account for protein-specific differences in Rosetta energy scales and offsets. We haven't established the cause yet.

These results are exploratory and don't directly answer our main research question.

## 5. Main experiment and revised evaluation
We're shifting the primary evaluation away from predicting one Rosetta energy using closely related Rosetta properties.

Those experiments can still be useful as controls, but they aren't enough to establish that biophysical supervision improves representations.

Our main experiment is now to:

1. Fine-tune EvoDiff's encoder on multiple Rosetta-derived biophysical properties.
2. Extract representations from the original and fine-tuned encoders.
3. Compare them using identical downstream probes.
4. Evaluate on independent, experimentally measured protein properties, potentially using selected ProteinGym assays.
5. Include appropriate controls, repeated seeds, and uncertainty estimates.

We need to distinguish improved Rosetta prediction from genuine downstream transfer. Even if performance improves on external assays, that would not by itself establish that the model learned physical mechanisms.

## 6. Files and reproducibility

All code is in scripts/.

Small results are organized under results/metl_global/.

Protein metadata is in data/metadata/.

Important experiments:

| Job ID | Experiment |
| 29051193 | Dataset export and validation |
| 29126077 | Nearest-neighbor baseline |
| 29247981 | Frozen EvoDiff embedding extraction |
| 29262121 | Frozen EvoDiff regression |
| 29288461 | Multitask EvoDiff fine-tuning; verification pending |

The exported sequences and labels are stored in:

metl-global/exports/run_29051193/

The frozen EvoDiff embeddings are stored in:

metl-global/embeddings/evodiff_oa38m_frozen/

The frozen regression checkpoint and predictions are stored in:

metl-global/analysis/evodiff_frozen_mlp_29262121/

I haven't uploaded the large datasets or checkpoints to GitHub. They're preserved on Engaging, and this repository contains the code and smaller results needed to understand and reproduce the experiments.
