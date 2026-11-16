from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import csv
import gzip
import json
import re
import sys

import numpy as np

source = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)

data = defaultdict(lambda: defaultdict(list))
pattern = re.compile(r"([A-Z])(\d+)([A-Z])")

with gzip.open(source / "sequences.csv.gz", "rt") as f1, \
     gzip.open(source / "rosetta_numeric.csv.gz", "rt") as f2:

    seq_reader = csv.DictReader(f1)
    label_reader = csv.DictReader(f2)
    columns = [c for c in label_reader.fieldnames if c != "row_position"]

    count = 0
    seen = set()

    for a, b in zip_longest(seq_reader, label_reader):
        assert a is not None and b is not None
        pos = int(a["row_position"])
        assert pos == int(b["row_position"])
        assert pos not in seen
        seen.add(pos)

        mutations = {}
        if a["mutations"].strip():
            for item in a["mutations"].split(","):
                match = pattern.fullmatch(item.strip())
                assert match is not None, item
                site = int(match.group(2))
                assert site not in mutations
                mutations[site] = match.group(3)

        assert len(mutations) == int(a["n_mutations"])

        values = np.fromiter(
            (float(b[c]) for c in columns),
            dtype=np.float64,
            count=len(columns),
        )

        data[a["pdb_fn"]][a["split"]].append({
            "position": pos,
            "mutations": mutations,
            "values": values,
        })
        count += 1

assert count == 274000
assert len(data) == 8

histogram = Counter()
metrics = []
zero_distance = Counter()
total_queries = 0

with gzip.open(output / "nearest_neighbors.csv.gz", "wt", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "row_position", "pdb_fn", "split",
        "nearest_train_position", "hamming_distance"
    ])

    for protein in sorted(data):
        groups = data[protein]
        train = groups.get("train", [])

        if not train:
            print(f"{protein}: held out, skipped", flush=True)
            continue

        assert len(train) == 40000
        assert len(groups["val"]) == 2000
        assert len(groups["test"]) == 2000

        by_site = defaultdict(set)

        for i, record in enumerate(train):
            for site in record["mutations"]:
                by_site[site].add(i)

        lengths = [len(r["mutations"]) for r in train]
        shortest = min(range(len(train)), key=lengths.__getitem__)

        for split in ("val", "test"):
            truth = []
            predictions = []

            for j, query in enumerate(groups[split]):
                qm = query["mutations"]
                best = len(qm) + lengths[shortest]
                best_i = shortest

                candidates = set()
                for site in qm:
                    candidates.update(by_site.get(site, ()))

                for i in candidates:
                    tm = train[i]["mutations"]
                    common = sum(site in tm for site in qm)
                    identical = sum(
                        tm.get(site) == amino_acid
                        for site, amino_acid in qm.items()
                    )

                    distance = len(qm) + lengths[i] - common - identical

                    if distance < best or (
                        distance == best and i < best_i
                    ):
                        best = distance
                        best_i = i

                writer.writerow([
                    query["position"], protein, split,
                    train[best_i]["position"], best
                ])

                histogram[(protein, split, best)] += 1
                zero_distance[(protein, split)] += (best == 0)
                truth.append(query["values"])
                predictions.append(train[best_i]["values"])
                total_queries += 1

                if (j + 1) % 1000 == 0:
                    print(
                        f"{protein} {split}: {j + 1}/2000",
                        flush=True,
                    )

            actual = np.stack(truth)
            predicted = np.stack(predictions)

            mae = np.mean(np.abs(actual - predicted), axis=0)
            rmse = np.sqrt(np.mean((actual - predicted) ** 2, axis=0))
            ss_res = np.sum((actual - predicted) ** 2, axis=0)
            ss_tot = np.sum(
                (actual - actual.mean(axis=0)) ** 2,
                axis=0,
            )

            for k, column in enumerate(columns):
                r2 = (
                    float(1 - ss_res[k] / ss_tot[k])
                    if ss_tot[k] > 0
                    else None
                )
                metrics.append([
                    protein, split, column, len(truth),
                    float(mae[k]), float(rmse[k]), r2
                ])

with open(output / "baseline_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "pdb_fn", "split", "property", "n",
        "MAE", "RMSE", "R2"
    ])
    writer.writerows(metrics)

with open(output / "distance_histogram.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "pdb_fn", "split", "hamming_distance", "count"
    ])
    for key, count in sorted(histogram.items()):
        writer.writerow([*key, count])

assert total_queries == 24000
assert len(metrics) == 6 * 2 * len(columns)

report = {
    "source_variants": 274000,
    "queries_evaluated": total_queries,
    "numeric_properties": len(columns),
    "zero_distance_counts": {
        f"{protein}/{split}": count
        for (protein, split), count in sorted(zero_distance.items())
    },
    "total_zero_distance": sum(zero_distance.values()),
    "heldout_proteins_excluded": sorted(
        protein for protein in data if not data[protein].get("train")
    ),
}

with open(output / "summary.json", "w") as f:
    json.dump(report, f, indent=2)

print("ANALYSIS COMPLETE", flush=True)
print(json.dumps(report, indent=2), flush=True)
