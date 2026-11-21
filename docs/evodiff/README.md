# EvoDiff with Rosetta supervision — update since Check-in 1

**6.3930 | November 2026**

This is the EvoDiff work I did for the project. The original question was whether adding biophysical supervision to a pretrained protein model could make its representations more useful. At Check-in 1, the EvoDiff work was limited to frozen-embedding experiments on avGFP. Since then I moved to METL-Global, actually fine-tuned EvoDiff, compared different groups of Rosetta labels, and checked what happened on two proteins not used for Rosetta fine-tuning.

The short version: the models improved at predicting related Rosetta properties on the six proteins represented during training. That improvement was much less reliable on the two excluded proteins. I then tried smaller encoder updates and partial freezing. Both reduced the increase in masked-token prediction loss; partial freezing also looked better in some low-support probe comparisons. None of this establishes an improvement in experimentally measured protein function.

I don't want to treat a good validation score as a generalization result. The details and the limitations are in the files below.

- [What I did and why](decisions.md) — chronological record of the decisions, including what was only proposed
- [Dataset, models, and evaluation](methods.md) — exact proteins, label groups, and evaluation rules
- [Results and limitations](results.md) — numbers and how far they can be interpreted
- [Figures](figures.md) — plots saved by the actual Engaging reporting jobs
- [How to reproduce and where the data are](reproducibility.md)
- [Use of AI and external code](ai_use.md)
- [References](references.md)

The code here is a copy of the scripts used for the three completed experimental batches. Large datasets, checkpoints, and embedding arrays stay on Engaging. The small result files and original PNGs are copied into `results/evodiff/` by the separate sync script.

**Status as of November 21:** supervised ablations, the original held-out protein evaluation, and nine preservation-aware runs completed. No ProteinGym experimental evaluation is reported here yet. Fold rotations and Family D were part of the design notes but were not run.
