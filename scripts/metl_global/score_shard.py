#!/usr/bin/env python3
"""Score one METL-Global mutation shard with PyRosetta."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyrosetta
from pyrosetta import get_fa_scorefxn, pose_from_file
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

DEFAULT_TERMS = (
    "fa_atr,fa_rep,fa_sol,fa_intra_rep,fa_elec,pro_close,"
    "hbond_sr_bb,hbond_lr_bb,hbond_bb_sc,hbond_sc,"
    "dslf_fa,omega,fa_dun,p_aa_pp,yhh_planarity,ref,rama_prepro"
)
MUTATION_RE = re.compile(r"([A-Z])(\d+)([A-Z])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--terms", default=DEFAULT_TERMS)
    parser.add_argument("--repack", action="store_true")
    parser.add_argument("--write-per-residue", action="store_true")
    parser.add_argument(
        "--score-all-protein-residues",
        action="store_true",
        help="Default is to emit residues only for the manifest's selected chain.",
    )
    parser.add_argument(
        "--pyrosetta-flags",
        default="-mute all -ignore_unrecognized_res true",
    )
    return parser.parse_args()


def read_bases(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    bases = {}
    for row in rows:
        protein_id = row["protein_id"].strip()
        row["sequence"] = row["sequence"].strip().upper()
        row["chain"] = row["chain"].strip()
        pdb_path = Path(row["pdb_path"].strip())
        row["pdb_path"] = str(
            pdb_path if pdb_path.is_absolute() else path.parent / pdb_path
        )
        bases[protein_id] = row
    return bases


def open_variants(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", newline="")


def chain_positions(pose, chain: str, expected_sequence: str) -> list[int]:
    positions = [
        i
        for i in range(1, pose.total_residue() + 1)
        if pose.residue(i).is_protein() and pose.pdb_info().chain(i) == chain
    ]
    observed_sequence = "".join(pose.residue(i).name1() for i in positions)

    if observed_sequence != expected_sequence:
        raise ValueError(
            f"PDB chain {chain!r} does not match manifest sequence.\n"
            f"Expected ({len(expected_sequence)} aa): {expected_sequence[:80]}...\n"
            f"Observed ({len(observed_sequence)} aa): {observed_sequence[:80]}..."
        )
    return positions


def parse_mutations(text: str, sequence: str) -> list[tuple[int, str]]:
    result = []
    for token in filter(None, text.split(";")):
        match = MUTATION_RE.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid mutation token: {token!r}")

        wild_type, position_text, mutant = match.groups()
        sequence_position = int(position_text)

        if not 1 <= sequence_position <= len(sequence):
            raise ValueError(f"Out-of-range mutation: {token}")
        if sequence[sequence_position - 1] != wild_type:
            raise ValueError(
                f"Wild-type mismatch for {token}: manifest has "
                f"{sequence[sequence_position - 1]} at {sequence_position}"
            )

        result.append((sequence_position, mutant))
    return result


def write_table_atomic(table: pa.Table, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(destination)


def append_rows(rows, writer, temporary: Path):
    if not rows:
        return writer

    table = pa.Table.from_pylist(rows)
    if writer is None:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")

    writer.write_table(table)
    rows.clear()
    return writer


def main() -> None:
    args = parse_args()
    bases = read_bases(args.bases)

    with open_variants(args.variants) as handle:
        variants = list(csv.DictReader(handle, delimiter="\t"))

    if not variants:
        raise ValueError(f"No variants found in {args.variants}")

    protein_ids = {row["protein_id"] for row in variants}
    if len(protein_ids) != 1:
        raise ValueError("Each shard must contain exactly one protein_id")

    protein_id = next(iter(protein_ids))
    if protein_id not in bases:
        raise KeyError(f"{protein_id!r} is absent from {args.bases}")

    base = bases[protein_id]
    shard_name = args.variants.name.removesuffix(".tsv.gz").removesuffix(".tsv")

    output_dir = args.out_dir / protein_id
    variant_output = output_dir / f"{shard_name}.variant.parquet"
    residue_output = output_dir / f"{shard_name}.residue.parquet"

    if variant_output.exists() and (
        not args.write_per_residue or residue_output.exists()
    ):
        print(f"Already complete: {variant_output}", file=sys.stderr)
        return

    pyrosetta.init(args.pyrosetta_flags)
    scorefxn = get_fa_scorefxn()

    terms = [term.strip() for term in args.terms.split(",") if term.strip()]
    score_types = []
    for term in terms:
        try:
            score_type = getattr(pyrosetta.rosetta.core.scoring, term)
        except AttributeError as exc:
            raise ValueError(f"Unknown Rosetta score term: {term}") from exc
        score_types.append((term, score_type, float(scorefxn.weights()[score_type])))

    base_pose = pose_from_file(base["pdb_path"])
    selected_positions = chain_positions(base_pose, base["chain"], base["sequence"])

    if args.score_all_protein_residues:
        output_positions = [
            i
            for i in range(1, base_pose.total_residue() + 1)
            if base_pose.residue(i).is_protein()
        ]
    else:
        output_positions = selected_positions

    sequence_position_by_pose_index = {
        pose_index: sequence_position
        for sequence_position, pose_index in enumerate(selected_positions, start=1)
    }

    variant_rows = []
    residue_rows = []
    residue_writer = None
    residue_temporary = residue_output.with_suffix(residue_output.suffix + ".tmp")

    for index, variant in enumerate(variants, start=1):
        pose = base_pose.clone()

        for sequence_position, mutant in parse_mutations(
            variant["mutations"], base["sequence"]
        ):
            MutateResidue(selected_positions[sequence_position - 1], mutant).apply(pose)

        if args.repack:
            PackRotamersMover(scorefxn).apply(pose)

        total_score = float(scorefxn(pose))
        total_energies = pose.energies().total_energies()

        variant_row = {
            "variant_id": variant["variant_id"],
            "protein_id": protein_id,
            "mutation_count": int(variant["mutation_count"]),
            "mutations": variant["mutations"],
            "total_score": total_score,
        }
        for name, score_type, weight in score_types:
            raw_value = float(total_energies[score_type])
            variant_row[f"{name}_raw"] = raw_value
            variant_row[f"{name}_weighted"] = raw_value * weight
        variant_rows.append(variant_row)

        if args.write_per_residue:
            for pose_index in output_positions:
                residue_energies = pose.energies().residue_total_energies(pose_index)
                residue_row = {
                    "variant_id": variant["variant_id"],
                    "protein_id": protein_id,
                    "pose_index": pose_index,
                    "sequence_position": sequence_position_by_pose_index.get(pose_index),
                    "chain": pose.pdb_info().chain(pose_index),
                    "pdb_residue_number": pose.pdb_info().number(pose_index),
                    "aa": pose.residue(pose_index).name1(),
                }
                for name, score_type, weight in score_types:
                    raw_value = float(residue_energies[score_type])
                    residue_row[f"{name}_raw"] = raw_value
                    residue_row[f"{name}_weighted"] = raw_value * weight
                residue_rows.append(residue_row)

            if len(residue_rows) >= 50_000:
                residue_writer = append_rows(
                    residue_rows, residue_writer, residue_temporary
                )

        if index % 100 == 0:
            print(
                f"{protein_id} {shard_name}: scored {index}/{len(variants)} variants",
                file=sys.stderr,
            )

    write_table_atomic(pa.Table.from_pylist(variant_rows), variant_output)

    if args.write_per_residue:
        residue_writer = append_rows(residue_rows, residue_writer, residue_temporary)
        if residue_writer is not None:
            residue_writer.close()
            residue_temporary.replace(residue_output)

    provenance = output_dir / f"{shard_name}.provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "bases": str(args.bases.resolve()),
                "variants": str(args.variants.resolve()),
                "pdb_path": base["pdb_path"],
                "chain": base["chain"],
                "repack": args.repack,
                "write_per_residue": args.write_per_residue,
                "score_all_protein_residues": args.score_all_protein_residues,
                "pyrosetta_flags": args.pyrosetta_flags,
                "terms": terms,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()