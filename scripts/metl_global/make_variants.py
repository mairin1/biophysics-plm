#!/usr/bin/env python3
"""Create deterministic, unique METL-Global mutation shards."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import random
import re
import shutil
import sys
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-per-protein", type=int, default=200_000)
    parser.add_argument("--max-mutations", type=int, default=5)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20240315)
    parser.add_argument(
        "--mutation-counts",
        default=None,
        help="Comma-separated permitted mutation counts. Default: 1..max-mutations.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def protein_seed(master_seed: int, protein_id: str) -> int:
    payload = f"{master_seed}\0{protein_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def read_bases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    required = {"protein_id", "pdb_path", "chain", "sequence"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must have columns: {sorted(required)}")

    seen_ids: set[str] = set()
    for row in rows:
        row["protein_id"] = row["protein_id"].strip()
        row["chain"] = row["chain"].strip()
        row["sequence"] = row["sequence"].strip().upper()

        if not row["protein_id"] or row["protein_id"] in seen_ids:
            raise ValueError(f"Duplicate or empty protein_id: {row['protein_id']!r}")
        if len(row["chain"]) != 1:
            raise ValueError(f"{row['protein_id']}: chain must be one character")
        if not row["sequence"] or set(row["sequence"]) - set(AA):
            raise ValueError(f"{row['protein_id']}: non-canonical residues in sequence")

        seen_ids.add(row["protein_id"])

    return rows


def open_shard(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = gzip.open(path, "wt", newline="")
    writer = csv.DictWriter(
        handle,
        delimiter="\t",
        fieldnames=["variant_id", "protein_id", "mutation_count", "mutations"],
    )
    writer.writeheader()
    return handle, writer


def main() -> None:
    args = parse_args()

    if args.n_per_protein < 1 or args.shard_size < 1:
        raise ValueError("n-per-protein and shard-size must both be positive")

    allowed_counts = (
        [int(x) for x in args.mutation_counts.split(",")]
        if args.mutation_counts
        else list(range(1, args.max_mutations + 1))
    )
    if not allowed_counts or min(allowed_counts) < 1:
        raise ValueError("mutation counts must be positive")

    bases = read_bases(args.bases)

    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.out_dir} exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(args.out_dir)

    args.out_dir.mkdir(parents=True)
    task_file = args.out_dir / "tasks.txt"

    metadata = args.out_dir / "generation_config.tsv"
    with metadata.open("w") as handle:
        handle.write("key\tvalue\n")
        for key, value in {
            "bases": str(args.bases.resolve()),
            "seed": args.seed,
            "n_per_protein": args.n_per_protein,
            "max_mutations": args.max_mutations,
            "mutation_counts": ",".join(map(str, allowed_counts)),
            "shard_size": args.shard_size,
            "alphabet": AA,
        }.items():
            handle.write(f"{key}\t{value}\n")

    with task_file.open("w") as task_handle:
        for base in bases:
            protein_id = base["protein_id"]
            sequence = base["sequence"]

            if max(allowed_counts) > len(sequence):
                raise ValueError(
                    f"{protein_id}: requested mutation count exceeds sequence length"
                )

            rng = random.Random(protein_seed(args.seed, protein_id))
            generated: set[tuple[str, ...]] = set()
            shard_number = 0
            records_in_shard = 0
            output_handle = None
            writer = None

            for ordinal in range(args.n_per_protein):
                for _ in range(100_000):
                    mutation_count = rng.choice(allowed_counts)
                    positions = sorted(rng.sample(range(len(sequence)), mutation_count))
                    mutations = tuple(
                        f"{sequence[pos]}{pos + 1}{rng.choice(AA.replace(sequence[pos], ''))}"
                        for pos in positions
                    )
                    if mutations not in generated:
                        generated.add(mutations)
                        break
                else:
                    raise RuntimeError(
                        f"{protein_id}: could not generate another unique mutation set"
                    )

                if records_in_shard == 0:
                    shard_path = (
                        args.out_dir
                        / safe_name(protein_id)
                        / f"shard-{shard_number:05d}.tsv.gz"
                    )
                    output_handle, writer = open_shard(shard_path)
                    task_handle.write(f"{shard_path.resolve()}\n")

                writer.writerow(
                    {
                        "variant_id": f"{protein_id}__{ordinal:07d}",
                        "protein_id": protein_id,
                        "mutation_count": len(mutations),
                        "mutations": ";".join(mutations),
                    }
                )
                records_in_shard += 1

                if records_in_shard == args.shard_size:
                    output_handle.close()
                    output_handle = None
                    writer = None
                    records_in_shard = 0
                    shard_number += 1

            if output_handle is not None:
                output_handle.close()

            print(
                f"{protein_id}: wrote {args.n_per_protein:,} variants "
                f"in {shard_number + bool(records_in_shard):,} shards",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()