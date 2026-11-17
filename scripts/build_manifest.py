from pathlib import Path
from collections import defaultdict
import csv
import random
import numpy as np

root = Path.home() / "orcd/pool/biophysics-plm"
data = root / "metl-global"
splits = data / "extracted/global/splits/standard_tr0.9_tu0.05_te0.05_w2a93d88bac32_r2098"

N = 29227460
SEED = 20261114
rng = random.Random(SEED)

with open(data / "protein_roles.csv") as f:
    roles = {
        row["pdb_fn"]: row["role"]
        for row in csv.DictReader(f)
    }

assert list(roles.values()).count("train") == 6
assert list(roles.values()).count("holdout") == 2

# Code: 1=train, 2=val, 3=test, 4=stest
split_codes = np.zeros(N, dtype=np.uint8)
split_names = {1: "train", 2: "val", 3: "test", 4: "stest"}

for code, name in split_names.items():
    indices = np.fromfile(
        splits / f"{name}.txt",
        dtype=np.int32,
        sep="\n",
    )
    split_codes[indices] = code
    del indices

assert np.all(split_codes != 0)

# Samples per protein, grouped by original split.
quotas = {
    ("train", "train"): 40000,
    ("train", "val"): 2000,
    ("train", "test"): 2000,
    ("holdout", "test"): 5000,
}

samples = defaultdict(list)
seen = defaultdict(int)

# Reservoir sampling: every eligible row has equal probability.
with open(data / "extracted/global/pdb_fns.txt") as f:
    for position, line in enumerate(f):
        protein = line.strip()

        if protein not in roles:
            continue

        role = roles[protein]
        split = split_names[int(split_codes[position])]
        key = (protein, split)

        limit = quotas.get((role, split))
        if limit is None:
            continue

        seen[key] += 1
        bucket = samples[key]

        if len(bucket) < limit:
            bucket.append(position)
        else:
            j = rng.randrange(seen[key])
            if j < limit:
                bucket[j] = position

assert position + 1 == N

rows = []

for (protein, split), positions in samples.items():
    role = roles[protein]
    expected = quotas[(role, split)]

    assert len(positions) == expected, (protein, split)

    for pos in positions:
        rows.append((pos, protein, role, split))

rows.sort()

output = data / "sample_manifest.csv"

with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["row_position", "pdb_fn", "role", "split"])
    writer.writerows(rows)

print("Sampling seed:", SEED)
print("Total sampled:", len(rows))
print("Unique positions:", len({r[0] for r in rows}))

for role in ["train", "holdout"]:
    for split in ["train", "val", "test"]:
        count = sum(
            r[2] == role and r[3] == split
            for r in rows
        )
        if count:
            print(f"{role:8s} / {split:5s}: {count:,}")

print("Saved:", output)
