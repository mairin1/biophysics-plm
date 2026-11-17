from pathlib import Path
from collections import Counter
import argparse
import csv
import gzip
import hashlib
import json
import math
import re

import h5py

ROOT = Path.home() / "orcd/pool/biophysics-plm"
DATA = ROOT / "metl-global"
PDB_DIR = ROOT / "metl-sim/pdb_files/prepared_pdb_files"
H5 = DATA / "extracted/global/global.h5"

AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
PATTERN = re.compile(r"([A-Z])(\d+)([A-Z])")


def reference_sequence(name):
    residues = {}
    chains = set()

    with open(PDB_DIR / name) as f:
        for line in f:
            if not line.startswith("ATOM  "):
                continue

            chains.add(line[21])
            pos = int(line[22:26])
            residue = AA.get(line[17:20].strip())

            if residue is None:
                raise ValueError(f"Unknown residue: {name}, {pos}")

            if pos in residues and residues[pos] != residue:
                raise ValueError(f"Inconsistent residue: {name}, {pos}")

            residues[pos] = residue

    if len(chains) != 1:
        raise ValueError(f"Multiple chains: {name}")

    if sorted(residues) != list(range(1, len(residues) + 1)):
        raise ValueError(f"Invalid numbering: {name}")

    return "".join(residues[i] for i in sorted(residues))


def apply_mutations(reference, mutations):
    sequence = list(reference)
    used_positions = set()
    count = 0

    if mutations.strip():
        for mutation in mutations.split(","):
            match = PATTERN.fullmatch(mutation.strip())
            if match is None:
                raise ValueError(f"Invalid mutation: {mutation}")

            wt, number, new = match.groups()
            pos = int(number)

            if not 1 <= pos <= len(sequence):
                raise ValueError(f"Out-of-range mutation: {mutation}")

            if pos in used_positions:
                raise ValueError(f"Duplicate mutation position: {mutation}")

            if reference[pos - 1] != wt:
                raise ValueError(
                    f"Wild-type mismatch: {mutation}; "
                    f"reference has {reference[pos - 1]}"
                )

            if new not in set(AA.values()):
                raise ValueError(f"Invalid mutant residue: {mutation}")

            sequence[pos - 1] = new
            used_positions.add(pos)
            count += 1

    return "".join(sequence), count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    manifest_path = DATA / "sample_manifest.csv"
    manifest = []

    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            manifest.append((
                int(row["row_position"]),
                row["pdb_fn"],
                row["role"],
                row["split"],
            ))

    manifest.sort()
    positions = [row[0] for row in manifest]

    assert len(positions) == 274000
    assert len(set(positions)) == len(positions)

    if args.limit:
        manifest = manifest[:args.limit]

    references = {
        name: reference_sequence(name)
        for name in sorted({row[1] for row in manifest})
    }

    with h5py.File(H5, "r") as h5:
        numeric_columns = [
            x.decode() for x in h5["variant/block0_items"][:]
        ]

    metadata_file = args.out / "sequences.csv.gz"
    labels_file = args.out / "rosetta_numeric.csv.gz"

    counts = Counter()
    missing_values = Counter()
    h5_checks = []
    target_index = 0
    expected = len(manifest)

    with (
        open(args.tsv, "r", buffering=8 * 1024 * 1024) as source,
        gzip.open(metadata_file, "wt", newline="") as meta_out,
        gzip.open(labels_file, "wt", newline="") as labels_out,
    ):
        header = source.readline().rstrip("\n\r").split("\t")
        assert len(header) == len(set(header))

        index = {name: i for i, name in enumerate(header)}
        assert "pdb_fn" in index and "mutations" in index
        assert set(numeric_columns).issubset(index)

        meta_writer = csv.writer(meta_out)
        label_writer = csv.writer(labels_out)

        meta_writer.writerow([
            "row_position", "pdb_fn", "role", "split",
            "mutations", "sequence", "n_mutations"
        ])
        label_writer.writerow(["row_position"] + numeric_columns)

        for position, line in enumerate(source):
            if position % 2000000 == 0:
                print(
                    f"Scanned {position:,} rows; "
                    f"exported {target_index:,}/{expected:,}",
                    flush=True,
                )

            if target_index == expected:
                break

            target = manifest[target_index]

            if position != target[0]:
                continue

            fields = line.rstrip("\n\r").split("\t")
            assert len(fields) == len(header), position

            pdb = fields[index["pdb_fn"]]
            mutations = fields[index["mutations"]]

            if pdb != target[1]:
                raise ValueError(
                    f"Row alignment failure at {position}: "
                    f"{pdb} != {target[1]}"
                )

            sequence, n_mutations = apply_mutations(
                references[pdb], mutations
            )

            meta_writer.writerow([
                position, pdb, target[2], target[3],
                mutations, sequence, n_mutations
            ])

            values = [fields[index[c]] for c in numeric_columns]
            label_writer.writerow([position] + values)

            for name, value in zip(numeric_columns, values):
                try:
                    valid = math.isfinite(float(value))
                except ValueError:
                    valid = False
                if not valid:
                    missing_values[name] += 1

            if target_index < 5 or target_index >= expected - 5:
                h5_checks.append((
                    position,
                    float(fields[index["fa_atr"]]),
                    float(fields[index["fa_sol"]]),
                ))

            counts[f"{target[2]}/{target[3]}"] += 1
            target_index += 1

    if target_index != expected:
        raise RuntimeError(
            f"Only exported {target_index}/{expected} rows"
        )

    # Independently verify selected TSV numerical values against HDF5.
    with h5py.File(H5, "r") as h5:
        block = h5["variant/block0_values"]
        atr_col = numeric_columns.index("fa_atr")
        sol_col = numeric_columns.index("fa_sol")

        for position, atr, sol in h5_checks:
            actual_atr = float(block[position, atr_col])
            actual_sol = float(block[position, sol_col])

            assert math.isclose(
                atr, actual_atr, rel_tol=1e-5, abs_tol=0.005
            ), f"HDF5 fa_atr mismatch at {position}"

            assert math.isclose(
                sol, actual_sol, rel_tol=1e-5, abs_tol=0.005
            ), f"HDF5 fa_sol mismatch at {position}"

    report = {
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "exported_variants": target_index,
        "reference_proteins": len(references),
        "numeric_columns": numeric_columns,
        "group_counts": dict(counts),
        "nonfinite_by_column": dict(missing_values),
        "h5_spot_checks_passed": len(h5_checks),
    }

    with open(args.out / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nEXPORT SUCCESSFUL", flush=True)
    print(json.dumps({
        "exported": target_index,
        "groups": dict(counts),
        "nonfinite_columns": dict(missing_values),
        "h5_checks": len(h5_checks),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
