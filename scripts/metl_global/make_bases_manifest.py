#!/usr/bin/env python3
"""Build the bases.tsv manifest required by the METL-Global scripts.

The selected sequence is read with PyRosetta, ensuring it matches the
sequence later seen by score_shard.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pyrosetta
from pyrosetta import pose_from_file

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        required=True,
        help="Directory containing base-protein .pdb files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/metl_global/bases.tsv"),
        help="Output manifest path.",
    )
    parser.add_argument(
        "--chain-map",
        type=Path,
        help=(
            "Optional TSV with columns protein_id and chain. Required for "
            "structures containing more than one protein chain."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search pdb-dir recursively for .pdb files.",
    )
    return parser.parse_args()


def read_chain_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    required = {"protein_id", "chain"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"{path} must be a TSV with columns: protein_id and chain"
        )

    result = {}
    for row in rows:
        protein_id = row["protein_id"].strip()
        chain = row["chain"].strip()
        if not protein_id or len(chain) != 1:
            raise ValueError(f"Bad chain-map row: {row}")
        if protein_id in result:
            raise ValueError(f"Duplicate protein_id in chain map: {protein_id}")
        result[protein_id] = chain

    return result


def get_protein_chains(pose) -> dict[str, list[int]]:
    chains: dict[str, list[int]] = {}

    for pose_index in range(1, pose.total_residue() + 1):
        residue = pose.residue(pose_index)
        if not residue.is_protein():
            continue

        chain = pose.pdb_info().chain(pose_index).strip()
        if not chain:
            raise ValueError(
                "Structure uses a blank PDB chain ID. Assign explicit chain "
                "IDs before generating the manifest."
            )
        chains.setdefault(chain, []).append(pose_index)

    return chains


def main() -> None:
    args = parse_args()

    if not args.pdb_dir.is_dir():
        raise FileNotFoundError(f"PDB directory does not exist: {args.pdb_dir}")

    chain_map = read_chain_map(args.chain_map)

    pdb_paths = sorted(
        args.pdb_dir.rglob("*.pdb") if args.recursive else args.pdb_dir.glob("*.pdb")
    )
    if not pdb_paths:
        raise FileNotFoundError(f"No .pdb files found in {args.pdb_dir}")

    pyrosetta.init("-mute all -ignore_unrecognized_res true")
    output_rows = []

    for pdb_path in pdb_paths:
        protein_id = pdb_path.stem
        pose = pose_from_file(str(pdb_path))
        chains = get_protein_chains(pose)

        if not chains:
            raise ValueError(f"{pdb_path}: no protein residues found")

        if protein_id in chain_map:
            chain = chain_map[protein_id]
            if chain not in chains:
                raise ValueError(
                    f"{pdb_path}: requested chain {chain!r} is absent; "
                    f"available: {sorted(chains)}"
                )
        elif len(chains) == 1:
            chain = next(iter(chains))
        else:
            raise ValueError(
                f"{pdb_path}: multiple protein chains found: {sorted(chains)}. "
                "Provide --chain-map to select one."
            )

        sequence = "".join(pose.residue(i).name1() for i in chains[chain])
        noncanonical = set(sequence) - CANONICAL_AA
        if noncanonical:
            raise ValueError(
                f"{pdb_path}, chain {chain}: noncanonical/unknown amino acids "
                f"in PyRosetta sequence: {sorted(noncanonical)}"
            )

        # Store PDB paths relative to the manifest, so the repo can move intact.
        relative_pdb_path = os.path.relpath(
            pdb_path.resolve(), start=args.out.parent.resolve()
        )

        output_rows.append(
            {
                "protein_id": protein_id,
                "pdb_path": relative_pdb_path,
                "chain": chain,
                "sequence": sequence,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["protein_id", "pdb_path", "chain", "sequence"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} base proteins to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()