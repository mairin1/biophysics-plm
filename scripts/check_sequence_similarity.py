from pathlib import Path
from itertools import combinations

from Bio import SeqIO, Align
from Bio.Align import substitution_matrices

path = Path("metl-global/candidate_sequences.fasta")
sequences = {
    record.id: str(record.seq)
    for record in SeqIO.parse(path, "fasta")
}

assert len(sequences) == 8
assert all("X" not in seq for seq in sequences.values())

aligner = Align.PairwiseAligner()
aligner.mode = "global"
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5

results = []

for (name1, seq1), (name2, seq2) in combinations(
    sequences.items(), 2
):
    alignment = aligner.align(seq1, seq2)[0]
    indices1, indices2 = alignment.indices

    paired = 0
    matches = 0

    for i, j in zip(indices1, indices2):
        if i >= 0 and j >= 0:
            paired += 1
            if seq1[i] == seq2[j]:
                matches += 1

    identity = matches / paired if paired else 0
    coverage = paired / min(len(seq1), len(seq2))
    gap_adjusted = matches / alignment.length

    results.append((name1, name2, identity, coverage, gap_adjusted))

results.sort(key=lambda x: x[4], reverse=True)

print("Proteins:", len(sequences))
print("Pairwise comparisons:", len(results))
print("\nMost similar pairs:\n")

for name1, name2, identity, coverage, adjusted in results[:10]:
    print(
        f"{name1:<23} {name2:<23} "
        f"Identity: {identity:.1%}  "
        f"Shorter-seq coverage: {coverage:.1%}  "
        f"Gap-adjusted: {adjusted:.1%}"
    )
