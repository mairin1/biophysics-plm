#!/usr/bin/env python3
"""
Standalone full validator and manifest builder for the METL six-protein ESM-2 study.

This script intentionally:
  * uses variant, never pdb_file;
  * uses only the six proteins in provisional_panel.tsv;
  * uses no legacy esm2_pg modules;
  * uses no EvoDiff or Check-in-2 export;
  * validates mutation/reference reconciliation before reconstructing sequences;
  * preserves official zero-based physical global.tsv split positions;
  * creates manifests and train-only standardization only after a clean pass.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROSETTA_TARGETS = (
    "total_score",
    "dslf_fa13",
    "fa_dun",
    "fa_elec",
    "fa_intra_rep",
    "fa_intra_sol_xover4",
    "fa_rep",
    "hbond_bb_sc",
    "hbond_lr_bb",
    "hbond_sc",
    "hbond_sr_bb",
    "lk_ball_wtd",
    "omega",
    "p_aa_pp",
    "pro_close",
    "rama_prepro",
    "ref",
    "yhh_planarity",
)

SPLIT_FILES = {
    "train": "train.txt",
    "val": "val.txt",
    "test": "test.txt",
    "supertest": "stest.txt",
}

SPLIT_CODES = {
    "train": 1,
    "val": 2,
    "test": 3,
    "supertest": 4,
}

CODE_TO_SPLIT = {value: key for key, value in SPLIT_CODES.items()}

MUTATION_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])$")

AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common crystallographic selenium-methionine representation.
    "MSE": "M",
}

ERROR_FIELDS = (
    "phase",
    "error_code",
    "pdb_fn",
    "sqlite_rowid",
    "physical_row_position_zero_based",
    "job_uuid",
    "mutations",
    "token",
    "position",
    "expected",
    "observed",
    "detail",
)

STANDARDIZATION_ID = "metl6_train_only_per_protein_target_population_std_v1"


@dataclass
class ReferenceInfo:
    pdb_fn: str
    pdb_id: str
    chain: str
    selection_rank: str
    max_tm: str
    mean_tm: str
    coordinate_path: Path
    source_url: str
    coordinate_sha256: str
    sequence: str
    sequence_sha256: str
    atom_sequence: str
    seqres_to_atom: list[int | None]
    alignment_matches: int
    alignment_substitutions: int
    seqres_unresolved_positions: int
    atom_insertions: int
    atom_unmapped_residues: int


class ErrorLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=ERROR_FIELDS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        self.writer.writeheader()
        self.count = 0
        self.by_code: Counter[str] = Counter()
        self.by_protein: Counter[str] = Counter()

    def add(
        self,
        phase: str,
        error_code: str,
        *,
        pdb_fn: str = "",
        sqlite_rowid: Any = "",
        physical_position: Any = "",
        job_uuid: str = "",
        mutations: str = "",
        token: str = "",
        position: Any = "",
        expected: Any = "",
        observed: Any = "",
        detail: str = "",
    ) -> None:
        self.count += 1
        self.by_code[error_code] += 1
        if pdb_fn:
            self.by_protein[pdb_fn] += 1

        self.writer.writerow(
            {
                "phase": phase,
                "error_code": error_code,
                "pdb_fn": pdb_fn,
                "sqlite_rowid": sqlite_rowid,
                "physical_row_position_zero_based": physical_position,
                "job_uuid": job_uuid,
                "mutations": mutations,
                "token": token,
                "position": position,
                "expected": expected,
                "observed": observed,
                "detail": detail,
            }
        )

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


class RunningStats:
    """Numerically stable population-standard-deviation accumulator."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    def population_std(self) -> float:
        if self.n == 0:
            return float("nan")
        return math.sqrt(self.m2 / self.n)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format(value, ".17g")
    return str(value)


def is_finite_value(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_dict_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_tsv_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"No header found in {path}")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows


def load_coordinate_manifest(path: Path) -> dict[str, dict[str, str]]:
    """
    Supports either:
      pdb_id <tab> coordinate_path <tab> source_url <tab> coordinate_sha256 <tab> header_id
    with or without a header row.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter="\t") if row]

    if not rows:
        raise RuntimeError(f"Coordinate manifest is empty: {path}")

    first = [value.strip().lower() for value in rows[0]]
    has_header = bool(first and first[0] in {"pdb_id", "pdb", "id"})

    if has_header:
        header = rows[0]
        data_rows = rows[1:]
        result = {}
        for values in data_rows:
            row = dict(zip(header, values))
            pdb_id = row.get("pdb_id", "").lower()
            if not pdb_id:
                raise RuntimeError(f"Coordinate manifest row lacks pdb_id: {row}")
            result[pdb_id] = {
                "pdb_id": pdb_id,
                "coordinate_path": row.get("coordinate_path", ""),
                "source_url": row.get("source_url", ""),
                "coordinate_sha256": row.get("coordinate_sha256", ""),
                "header_id": row.get("header_id", ""),
            }
        return result

    result = {}
    for values in rows:
        if len(values) < 4:
            raise RuntimeError(
                f"Headerless coordinate manifest row has fewer than four fields: {values!r}"
            )
        pdb_id = values[0].strip().lower()
        result[pdb_id] = {
            "pdb_id": pdb_id,
            "coordinate_path": values[1].strip(),
            "source_url": values[2].strip(),
            "coordinate_sha256": values[3].strip(),
            "header_id": values[4].strip() if len(values) > 4 else "",
        }
    return result


def align_seqres_to_atom(
    seqres: str,
    atom: str,
) -> tuple[list[int | None], int, int, int, int]:
    """
    Global alignment with:
      match = +2, mismatch = -2, gap = -2.

    Returns:
      seqres_position_to_atom_position,
      aligned_matches,
      substitutions,
      seqres_deletions/unresolved positions,
      atom_insertions.
    """
    n = len(seqres)
    m = len(atom)
    width = m + 1

    # 1=diag, 2=up (SEQRES residue aligned to atom gap), 3=left.
    trace = bytearray((n + 1) * (m + 1))
    previous = [-2 * j for j in range(m + 1)]

    for j in range(1, m + 1):
        trace[j] = 3

    for i in range(1, n + 1):
        current = [-2 * i] + [0] * m
        trace[i * width] = 2

        for j in range(1, m + 1):
            diagonal = previous[j - 1] + (2 if seqres[i - 1] == atom[j - 1] else -2)
            up = previous[j] - 2
            left = current[j - 1] - 2

            if diagonal >= up and diagonal >= left:
                current[j] = diagonal
                trace[i * width + j] = 1
            elif up >= left:
                current[j] = up
                trace[i * width + j] = 2
            else:
                current[j] = left
                trace[i * width + j] = 3

        previous = current

    mapping: list[int | None] = [None] * n
    matches = 0
    substitutions = 0
    seqres_deletions = 0
    atom_insertions = 0

    i = n
    j = m
    while i > 0 or j > 0:
        direction = trace[i * width + j] if i > 0 and j > 0 else (2 if i > 0 else 3)

        if i > 0 and j > 0 and direction == 1:
            mapping[i - 1] = j - 1
            if seqres[i - 1] == atom[j - 1]:
                matches += 1
            else:
                substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or direction == 2):
            seqres_deletions += 1
            i -= 1
        else:
            atom_insertions += 1
            j -= 1

    return mapping, matches, substitutions, seqres_deletions, atom_insertions


def parse_reference_chain(
    coordinate_path: Path,
    chain: str,
) -> tuple[str, str, list[int | None], int, int, int, int, int]:
    seqres_residues: list[str] = []
    atom_residues: list[str] = []
    atom_seen: set[tuple[str, str]] = set()
    atom_unmapped_residues = 0

    seen_model = False
    first_model_finished = False

    with coordinate_path.open("r", encoding="latin-1") as handle:
        for line in handle:
            record = line[:6].strip()

            if record == "MODEL":
                if seen_model:
                    break
                seen_model = True
                continue

            if record == "ENDMDL" and seen_model:
                first_model_finished = True
                break

            if record == "SEQRES":
                if len(line) > 11 and line[11:12] == chain:
                    for residue_name in line[19:70].split():
                        residue_name = residue_name.upper()
                        if residue_name not in AA3_TO_AA1:
                            raise RuntimeError(
                                f"Unsupported SEQRES residue {residue_name!r} "
                                f"for chain {chain} in {coordinate_path}"
                            )
                        seqres_residues.append(AA3_TO_AA1[residue_name])
                continue

            if first_model_finished or record not in {"ATOM", "HETATM"}:
                continue

            if len(line) <= 26 or line[21:22] != chain:
                continue

            residue_name = line[17:20].strip().upper()
            residue_key = (line[22:26], line[26:27])

            if residue_key in atom_seen:
                continue

            if residue_name in AA3_TO_AA1:
                atom_seen.add(residue_key)
                atom_residues.append(AA3_TO_AA1[residue_name])
            elif record == "ATOM":
                # Keep a count for provenance. It is not silently used as sequence.
                atom_seen.add(residue_key)
                atom_unmapped_residues += 1

    if not seqres_residues:
        raise RuntimeError(
            f"No canonical SEQRES residues found for selected chain {chain} in {coordinate_path}"
        )

    if not atom_residues:
        raise RuntimeError(
            f"No canonical ATOM/HETATM residues found for selected chain {chain} in {coordinate_path}"
        )

    seqres = "".join(seqres_residues)
    atom = "".join(atom_residues)
    mapping, matches, substitutions, deletions, insertions = align_seqres_to_atom(seqres, atom)

    return (
        seqres,
        atom,
        mapping,
        matches,
        substitutions,
        deletions,
        insertions,
        atom_unmapped_residues,
    )


def parse_mutations_and_validate(
    *,
    reference: ReferenceInfo,
    sqlite_rowid: int,
    mutations: str,
    job_uuid: str,
    errors: ErrorLog,
    statistics: Counter[str],
) -> None:
    if mutations is None or mutations == "":
        errors.add(
            "mutation_validation",
            "empty_mutation_string",
            pdb_fn=reference.pdb_fn,
            sqlite_rowid=sqlite_rowid,
            job_uuid=job_uuid,
            mutations="" if mutations is None else mutations,
        )
        statistics["empty_mutation_strings"] += 1
        return

    seen_positions: dict[int, str] = {}

    for token in mutations.split(","):
        statistics["mutation_tokens"] += 1
        match = MUTATION_RE.fullmatch(token)

        if match is None:
            errors.add(
                "mutation_validation",
                "malformed_mutation_token",
                pdb_fn=reference.pdb_fn,
                sqlite_rowid=sqlite_rowid,
                job_uuid=job_uuid,
                mutations=mutations,
                token=token,
                detail="Expected strict canonical substitution WT<one-based-position>MUT.",
            )
            statistics["malformed_tokens"] += 1
            continue

        wt, position_text, mutant = match.groups()
        position = int(position_text)

        if position in seen_positions:
            errors.add(
                "mutation_validation",
                "duplicate_mutation_position",
                pdb_fn=reference.pdb_fn,
                sqlite_rowid=sqlite_rowid,
                job_uuid=job_uuid,
                mutations=mutations,
                token=token,
                position=position,
                expected=seen_positions[position],
                observed=mutant,
                detail="The same residue position appears more than once in one mutation string.",
            )
            statistics["duplicate_positions"] += 1

            if seen_positions[position] != mutant:
                errors.add(
                    "mutation_validation",
                    "conflicting_mutation_position",
                    pdb_fn=reference.pdb_fn,
                    sqlite_rowid=sqlite_rowid,
                    job_uuid=job_uuid,
                    mutations=mutations,
                    token=token,
                    position=position,
                    expected=seen_positions[position],
                    observed=mutant,
                    detail="Repeated residue position has conflicting mutant amino acids.",
                )
                statistics["conflicting_positions"] += 1
        else:
            seen_positions[position] = mutant

        if position < 1 or position > len(reference.sequence):
            errors.add(
                "mutation_validation",
                "mutation_position_out_of_range",
                pdb_fn=reference.pdb_fn,
                sqlite_rowid=sqlite_rowid,
                job_uuid=job_uuid,
                mutations=mutations,
                token=token,
                position=position,
                expected=f"1..{len(reference.sequence)}",
                observed=position,
            )
            statistics["out_of_range_positions"] += 1
            continue

        expected_wt = reference.sequence[position - 1]
        if wt != expected_wt:
            errors.add(
                "mutation_validation",
                "reference_wild_type_mismatch",
                pdb_fn=reference.pdb_fn,
                sqlite_rowid=sqlite_rowid,
                job_uuid=job_uuid,
                mutations=mutations,
                token=token,
                position=position,
                expected=expected_wt,
                observed=wt,
                detail="Mutation token wild type disagrees with selected-chain SEQRES reference.",
            )
            statistics["wild_type_mismatches"] += 1
            continue

        atom_position = reference.seqres_to_atom[position - 1]
        if atom_position is None:
            statistics["mutation_sites_unresolved_in_atom"] += 1
        else:
            statistics["mutation_sites_resolved_in_atom"] += 1
            atom_residue = reference.atom_sequence[atom_position]
            if atom_residue != expected_wt:
                errors.add(
                    "mutation_validation",
                    "atom_reference_residue_mismatch",
                    pdb_fn=reference.pdb_fn,
                    sqlite_rowid=sqlite_rowid,
                    job_uuid=job_uuid,
                    mutations=mutations,
                    token=token,
                    position=position,
                    expected=expected_wt,
                    observed=atom_residue,
                    detail="SEQRES-to-ATOM alignment maps this mutation position to a different residue.",
                )
                statistics["atom_reference_mismatches"] += 1

        statistics["valid_tokens"] += 1


def apply_mutations(reference_sequence: str, mutations: str) -> str:
    sequence = list(reference_sequence)
    seen: set[int] = set()

    for token in mutations.split(","):
        match = MUTATION_RE.fullmatch(token)
        if match is None:
            raise RuntimeError(f"Unexpected invalid mutation token after clean validation: {token!r}")

        wt, position_text, mutant = match.groups()
        position = int(position_text)

        if position in seen:
            raise RuntimeError(f"Unexpected duplicate mutation position after clean validation: {mutations}")
        seen.add(position)

        if position < 1 or position > len(sequence):
            raise RuntimeError(f"Unexpected out-of-range mutation after clean validation: {mutations}")

        if sequence[position - 1] != wt:
            raise RuntimeError(
                f"Unexpected wild-type mismatch after clean validation at {position}: "
                f"{sequence[position - 1]} != {wt}"
            )

        sequence[position - 1] = mutant

    return "".join(sequence)


def numeric_values_match(db_value: Any, tsv_value: str) -> bool:
    if db_value is None:
        return tsv_value == ""

    try:
        parsed = float(tsv_value)
        database = float(db_value)
    except (TypeError, ValueError):
        return False

    if math.isnan(database):
        return math.isnan(parsed)
    if math.isinf(database):
        return database == parsed

    return math.isclose(database, parsed, rel_tol=0.0, abs_tol=1e-9)


def make_output_directory(output: Path, force: bool) -> None:
    if output.exists():
        if not force:
            raise RuntimeError(
                f"Output directory already exists: {output}\n"
                "Use a new output directory, or rerun with --force to replace it."
            )
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=False)


def load_and_validate_splits(
    split_dir: Path,
    supertest_alias: Path,
    output: Path,
    errors: ErrorLog,
) -> tuple[bytearray, dict[str, Any]]:
    for filename in SPLIT_FILES.values():
        path = split_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Required split file is missing: {path}")

    summaries: dict[str, dict[str, Any]] = {}

    print("[splits] first pass: count/range/checksum", flush=True)
    global_max = -1

    for split_name, filename in SPLIT_FILES.items():
        path = split_dir / filename
        count = 0
        minimum: int | None = None
        maximum: int | None = None

        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                token = raw.strip()
                if not token or not token.isdigit():
                    errors.add(
                        "split_validation",
                        "malformed_split_position",
                        detail=f"{path}:{line_number}: {raw[:120]!r}",
                    )
                    continue

                position = int(token)
                count += 1
                minimum = position if minimum is None else min(minimum, position)
                maximum = position if maximum is None else max(maximum, position)

        if maximum is None:
            errors.add(
                "split_validation",
                "empty_split_file",
                detail=str(path),
            )
            maximum = -1

        global_max = max(global_max, maximum)
        summaries[split_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "integer_count": count,
            "minimum": minimum,
            "maximum": maximum,
        }

    if global_max < 0:
        raise RuntimeError("No valid physical positions found in official split files.")

    expected_physical_rows = global_max + 1
    assignments = bytearray(expected_physical_rows)

    print("[splits] second pass: duplicate/overlap/coverage checks", flush=True)
    for split_name, filename in SPLIT_FILES.items():
        path = split_dir / filename
        code = SPLIT_CODES[split_name]

        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                token = raw.strip()
                if not token or not token.isdigit():
                    continue

                position = int(token)
                if position < 0 or position >= expected_physical_rows:
                    errors.add(
                        "split_validation",
                        "split_position_out_of_range",
                        physical_position=position,
                        detail=f"{path}:{line_number}",
                    )
                    continue

                previous_code = assignments[position]
                if previous_code != 0:
                    errors.add(
                        "split_validation",
                        "duplicate_or_overlapping_split_position",
                        physical_position=position,
                        expected=CODE_TO_SPLIT.get(previous_code, ""),
                        observed=split_name,
                        detail=f"{path}:{line_number}",
                    )
                    continue

                assignments[position] = code

    missing_count = assignments.count(0)
    if missing_count:
        examples: list[int] = []
        for index, code in enumerate(assignments):
            if code == 0:
                examples.append(index)
                if len(examples) == 25:
                    break

        errors.add(
            "split_validation",
            "split_union_has_unassigned_physical_positions",
            detail=(
                f"missing_count={missing_count}; "
                f"first_25_missing_positions={examples}"
            ),
        )

    listed_total = sum(info["integer_count"] for info in summaries.values())
    if listed_total != expected_physical_rows:
        errors.add(
            "split_validation",
            "split_counts_do_not_equal_physical_row_universe",
            expected=expected_physical_rows,
            observed=listed_total,
            detail="A clean exhaustive non-overlapping partition must have equal values.",
        )

    if not supertest_alias.is_file():
        errors.add(
            "split_validation",
            "supertest_alias_missing",
            detail=str(supertest_alias),
        )
        alias_sha = ""
        alias_matches = False
    else:
        alias_sha = sha256_file(supertest_alias)
        alias_matches = alias_sha == summaries["supertest"]["sha256"]
        if not alias_matches:
            errors.add(
                "split_validation",
                "supertest_alias_checksum_mismatch",
                expected=summaries["supertest"]["sha256"],
                observed=alias_sha,
                detail=(
                    f"Expected alias of {split_dir / SPLIT_FILES['supertest']}; "
                    f"observed {supertest_alias}"
                ),
            )

    inventory = {
        "split_directory": str(split_dir),
        "index_kind": "zero_based_physical_global_tsv_data_row_position",
        "header_is_excluded": True,
        "expected_global_tsv_data_rows": expected_physical_rows,
        "splits": summaries,
        "supertest_alias": {
            "path": str(supertest_alias),
            "sha256": alias_sha,
            "matches_stest_txt_by_sha256": alias_matches,
        },
    }

    with (output / "split_inventory.json").open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2, sort_keys=True)
        handle.write("\n")

    specification = {
        "source_split_scheme": split_dir.name,
        "index_kind": "zero_based_physical_global_tsv_data_row_position",
        "index_base": 0,
        "header_is_excluded": True,
        "partitions": {
            split_name: str(split_dir / filename)
            for split_name, filename in SPLIT_FILES.items()
        },
        "supertest_alias": str(supertest_alias),
        "notes": [
            "train, val, test, and supertest are retained as four disjoint official partitions.",
            "The root-level supertest file is an alias only, not an additional partition.",
        ],
    }

    with (output / "split_specification.json").open("w", encoding="utf-8") as handle:
        json.dump(specification, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return assignments, inventory


def database_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro"


def write_validation_summary(
    output: Path,
    *,
    status: str,
    panel: list[dict[str, str]],
    errors: ErrorLog,
    extra: dict[str, Any],
) -> None:
    payload = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_panel": [row["pdb_fn"] for row in panel],
        "error_count": errors.count,
        "errors_by_code": dict(sorted(errors.by_code.items())),
        "errors_by_protein": dict(sorted(errors.by_protein.items())),
        **extra,
    }

    with (output / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_references(
    *,
    panel: list[dict[str, str]],
    audit_by_pdb_fn: dict[str, dict[str, str]],
    coordinate_manifest: dict[str, dict[str, str]],
    output: Path,
    errors: ErrorLog,
) -> dict[str, ReferenceInfo]:
    references: dict[str, ReferenceInfo] = {}
    provenance_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []

    for panel_row in panel:
        pdb_fn = panel_row["pdb_fn"]
        pdb_id = panel_row["pdb_id"].lower()
        chain = panel_row["chain"]

        audit = audit_by_pdb_fn.get(pdb_fn)
        if audit is None:
            errors.add(
                "reference_preflight",
                "selected_panel_entry_missing_from_audit",
                pdb_fn=pdb_fn,
            )
            continue

        if audit.get("status") != "EXPLICIT_CHAIN_SAMPLE_MATCH":
            errors.add(
                "reference_preflight",
                "audit_status_not_explicit_chain_sample_match",
                pdb_fn=pdb_fn,
                expected="EXPLICIT_CHAIN_SAMPLE_MATCH",
                observed=audit.get("status", ""),
            )

        if audit.get("explicit_chain") != chain:
            errors.add(
                "reference_preflight",
                "audit_explicit_chain_disagrees_with_panel",
                pdb_fn=pdb_fn,
                expected=chain,
                observed=audit.get("explicit_chain", ""),
            )

        coordinate = coordinate_manifest.get(pdb_id)
        if coordinate is None:
            errors.add(
                "reference_preflight",
                "coordinate_manifest_missing_pdb_id",
                pdb_fn=pdb_fn,
                expected=pdb_id,
            )
            continue

        expected_hashes = {
            panel_row.get("coordinate_sha256", ""),
            audit.get("coordinate_sha256", ""),
            coordinate.get("coordinate_sha256", ""),
        }

        if len(expected_hashes) != 1 or "" in expected_hashes:
            errors.add(
                "reference_preflight",
                "coordinate_hash_provenance_disagreement",
                pdb_fn=pdb_fn,
                detail=json.dumps(
                    {
                        "panel": panel_row.get("coordinate_sha256", ""),
                        "audit": audit.get("coordinate_sha256", ""),
                        "coordinate_manifest": coordinate.get("coordinate_sha256", ""),
                    },
                    sort_keys=True,
                ),
            )
            continue

        coordinate_path = Path(audit.get("coordinate_path", ""))
        if not coordinate_path.is_file():
            errors.add(
                "reference_preflight",
                "coordinate_file_missing",
                pdb_fn=pdb_fn,
                detail=str(coordinate_path),
            )
            continue

        actual_hash = sha256_file(coordinate_path)
        expected_hash = next(iter(expected_hashes))

        if actual_hash != expected_hash:
            errors.add(
                "reference_preflight",
                "coordinate_sha256_mismatch",
                pdb_fn=pdb_fn,
                expected=expected_hash,
                observed=actual_hash,
                detail=str(coordinate_path),
            )
            continue

        try:
            (
                sequence,
                atom_sequence,
                seqres_to_atom,
                matches,
                substitutions,
                unresolved,
                insertions,
                atom_unmapped,
            ) = parse_reference_chain(coordinate_path, chain)
        except Exception as exc:
            errors.add(
                "reference_preflight",
                "reference_chain_parse_failed",
                pdb_fn=pdb_fn,
                detail=f"{type(exc).__name__}: {exc}",
            )
            continue

        if len(sequence) > 1022:
            errors.add(
                "reference_preflight",
                "reference_sequence_exceeds_esm2_limit",
                pdb_fn=pdb_fn,
                expected="<=1022",
                observed=len(sequence),
            )

        if substitutions:
            errors.add(
                "reference_preflight",
                "seqres_atom_substitution_detected",
                pdb_fn=pdb_fn,
                observed=substitutions,
                detail=(
                    "SEQRES and resolved ATOM chain sequences have one or more "
                    "aligned amino-acid substitutions."
                ),
            )

        if insertions:
            errors.add(
                "reference_preflight",
                "atom_sequence_has_insertions_relative_to_seqres",
                pdb_fn=pdb_fn,
                observed=insertions,
                detail=(
                    "Resolved ATOM sequence has residues not reconciled to "
                    "the selected-chain SEQRES sequence."
                ),
            )

        sequence_hash = sha256_text(sequence)
        references[pdb_fn] = ReferenceInfo(
            pdb_fn=pdb_fn,
            pdb_id=pdb_id,
            chain=chain,
            selection_rank=panel_row.get("selection_rank", ""),
            max_tm=panel_row.get("max_tm_to_other_panel_member", ""),
            mean_tm=panel_row.get("mean_tm_to_other_panel_members", ""),
            coordinate_path=coordinate_path,
            source_url=audit.get("source_url", ""),
            coordinate_sha256=expected_hash,
            sequence=sequence,
            sequence_sha256=sequence_hash,
            atom_sequence=atom_sequence,
            seqres_to_atom=seqres_to_atom,
            alignment_matches=matches,
            alignment_substitutions=substitutions,
            seqres_unresolved_positions=unresolved,
            atom_insertions=insertions,
            atom_unmapped_residues=atom_unmapped,
        )

        provenance_rows.append(
            {
                "pdb_fn": pdb_fn,
                "pdb_id": pdb_id,
                "selected_chain": chain,
                "coordinate_path": str(coordinate_path),
                "source_url": audit.get("source_url", ""),
                "coordinate_sha256_expected": expected_hash,
                "coordinate_sha256_actual": actual_hash,
                "verified": "true",
            }
        )

        reconciliation_rows.append(
            {
                "pdb_fn": pdb_fn,
                "pdb_id": pdb_id,
                "selected_chain": chain,
                "seqres_length": len(sequence),
                "seqres_sha256": sequence_hash,
                "atom_length": len(atom_sequence),
                "seqres_atom_matches": matches,
                "seqres_atom_substitutions": substitutions,
                "seqres_positions_unresolved_in_atom": unresolved,
                "atom_insertions_relative_to_seqres": insertions,
                "atom_unmapped_residue_count": atom_unmapped,
                "esm2_residue_limit_pass": str(len(sequence) <= 1022).lower(),
            }
        )

    write_dict_tsv(
        output / "selected_coordinate_provenance.tsv",
        [
            "pdb_fn",
            "pdb_id",
            "selected_chain",
            "coordinate_path",
            "source_url",
            "coordinate_sha256_expected",
            "coordinate_sha256_actual",
            "verified",
        ],
        provenance_rows,
    )

    write_dict_tsv(
        output / "reference_sequence_reconciliation.tsv",
        [
            "pdb_fn",
            "pdb_id",
            "selected_chain",
            "seqres_length",
            "seqres_sha256",
            "atom_length",
            "seqres_atom_matches",
            "seqres_atom_substitutions",
            "seqres_positions_unresolved_in_atom",
            "atom_insertions_relative_to_seqres",
            "atom_unmapped_residue_count",
            "esm2_residue_limit_pass",
        ],
        reconciliation_rows,
    )

    return references


def create_stage_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE selected_rows (
            ordinal INTEGER PRIMARY KEY,
            pdb_fn TEXT NOT NULL,
            mutations TEXT NOT NULL,
            job_uuid TEXT NOT NULL,
            sqlite_rowid INTEGER NOT NULL,
            physical_position INTEGER,
            split_assignment TEXT,
            UNIQUE (pdb_fn, mutations, job_uuid)
        )
        """
    )
    return connection


def fetch_selected_and_validate_mutations(
    *,
    db: sqlite3.Connection,
    stage: sqlite3.Connection,
    panel: list[dict[str, str]],
    references: dict[str, ReferenceInfo],
    audit_by_pdb_fn: dict[str, dict[str, str]],
    errors: ErrorLog,
) -> tuple[dict[str, Counter[str]], int]:
    columns = [row["name"] for row in db.execute("PRAGMA table_info(variant)")]
    required = {"pdb_fn", "mutations", "job_uuid", *ROSETTA_TARGETS}

    missing = sorted(required.difference(columns))
    if missing:
        raise RuntimeError(f"variant is missing required columns: {missing}")

    selected_columns_sql = ", ".join(quote_identifier(column) for column in columns)
    query = (
        f"SELECT rowid AS sqlite_rowid, {selected_columns_sql} "
        f"FROM variant WHERE pdb_fn = ? ORDER BY rowid"
    )

    ordinal = 0
    statistics: dict[str, Counter[str]] = {
        row["pdb_fn"]: Counter() for row in panel
    }

    for panel_row in panel:
        pdb_fn = panel_row["pdb_fn"]
        ref = references[pdb_fn]
        counter = statistics[pdb_fn]

        print(f"[variant] indexed scan: {pdb_fn}", flush=True)
        cursor = db.execute(query, (pdb_fn,))
        insert_rows: list[tuple[Any, ...]] = []

        while True:
            batch = cursor.fetchmany(5000)
            if not batch:
                break

            for row in batch:
                ordinal += 1
                counter["database_rows"] += 1

                sqlite_rowid = int(row["sqlite_rowid"])
                mutations = row["mutations"]
                job_uuid = row["job_uuid"]

                parse_mutations_and_validate(
                    reference=ref,
                    sqlite_rowid=sqlite_rowid,
                    mutations=mutations,
                    job_uuid=job_uuid,
                    errors=errors,
                    statistics=counter,
                )

                insert_rows.append(
                    (ordinal, pdb_fn, mutations, job_uuid, sqlite_rowid)
                )

            stage.executemany(
                """
                INSERT INTO selected_rows
                    (ordinal, pdb_fn, mutations, job_uuid, sqlite_rowid)
                VALUES (?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
            insert_rows.clear()
            stage.commit()

        expected_count = int(audit_by_pdb_fn[pdb_fn]["variant_rows"])
        if counter["database_rows"] != expected_count:
            errors.add(
                "variant_scan",
                "selected_variant_row_count_disagrees_with_audit",
                pdb_fn=pdb_fn,
                expected=expected_count,
                observed=counter["database_rows"],
            )

    stage.execute(
        "CREATE INDEX selected_rows_lookup ON selected_rows (pdb_fn, mutations, job_uuid)"
    )
    stage.execute(
        "CREATE INDEX selected_rows_pdb_job ON selected_rows (pdb_fn, job_uuid)"
    )
    stage.commit()

    return statistics, ordinal


def reconcile_global_tsv(
    *,
    tsv_path: Path,
    db: sqlite3.Connection,
    stage: sqlite3.Connection,
    selected_pdb_fns: set[str],
    assignments: bytearray,
    errors: ErrorLog,
) -> tuple[int, dict[str, dict[str, int]]]:
    db_columns = [row["name"] for row in db.execute("PRAGMA table_info(variant)")]
    column_index = {column: index for index, column in enumerate(db_columns)}
    label_query = (
        "SELECT pdb_fn, mutations, job_uuid, "
        + ", ".join(quote_identifier(target) for target in ROSETTA_TARGETS)
        + " FROM variant WHERE rowid = ?"
    )

    matched_ordinals = bytearray(
        int(stage.execute("SELECT COUNT(*) FROM selected_rows").fetchone()[0]) + 1
    )
    pending_updates: list[tuple[int, str, int]] = []
    selected_tsv_rows = 0

    print("[global.tsv] streaming once for physical positions and source reconciliation", flush=True)

    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise RuntimeError(f"global.tsv is empty: {tsv_path}")

        header = header_line.rstrip("\r\n").split("\t")
        if header != db_columns:
            errors.add(
                "tsv_reconciliation",
                "global_tsv_header_disagrees_with_variant_schema",
                expected=json.dumps(db_columns),
                observed=json.dumps(header),
            )

        for physical_position, raw_line in enumerate(handle):
            first_field, separator, _ = raw_line.partition("\t")
            if not separator or first_field not in selected_pdb_fns:
                continue

            selected_tsv_rows += 1
            fields = raw_line.rstrip("\r\n").split("\t")

            if len(fields) != len(db_columns):
                errors.add(
                    "tsv_reconciliation",
                    "selected_global_tsv_row_has_wrong_column_count",
                    pdb_fn=first_field,
                    physical_position=physical_position,
                    expected=len(db_columns),
                    observed=len(fields),
                )
                continue

            pdb_fn = fields[column_index["pdb_fn"]]
            mutations = fields[column_index["mutations"]]
            job_uuid = fields[column_index["job_uuid"]]

            staged = stage.execute(
                """
                SELECT ordinal, sqlite_rowid
                FROM selected_rows
                WHERE pdb_fn = ? AND mutations = ? AND job_uuid = ?
                """,
                (pdb_fn, mutations, job_uuid),
            ).fetchone()

            if staged is None:
                errors.add(
                    "tsv_reconciliation",
                    "selected_global_tsv_identity_missing_from_variant",
                    pdb_fn=pdb_fn,
                    physical_position=physical_position,
                    job_uuid=job_uuid,
                    mutations=mutations,
                    detail=(
                        "Join key is (pdb_fn, mutations, job_uuid), the source primary key."
                    ),
                )
                continue

            ordinal = int(staged["ordinal"])
            sqlite_rowid = int(staged["sqlite_rowid"])

            if matched_ordinals[ordinal]:
                errors.add(
                    "tsv_reconciliation",
                    "duplicate_selected_source_identity_in_global_tsv",
                    pdb_fn=pdb_fn,
                    sqlite_rowid=sqlite_rowid,
                    physical_position=physical_position,
                    job_uuid=job_uuid,
                    mutations=mutations,
                )
                continue

            matched_ordinals[ordinal] = 1

            db_row = db.execute(label_query, (sqlite_rowid,)).fetchone()
            if db_row is None:
                errors.add(
                    "tsv_reconciliation",
                    "sqlite_rowid_missing_during_source_reconciliation",
                    pdb_fn=pdb_fn,
                    sqlite_rowid=sqlite_rowid,
                    physical_position=physical_position,
                    job_uuid=job_uuid,
                    mutations=mutations,
                )
                continue

            if (
                db_row["pdb_fn"] != pdb_fn
                or db_row["mutations"] != mutations
                or db_row["job_uuid"] != job_uuid
            ):
                errors.add(
                    "tsv_reconciliation",
                    "sqlite_rowid_identity_disagrees_with_staged_identity",
                    pdb_fn=pdb_fn,
                    sqlite_rowid=sqlite_rowid,
                    physical_position=physical_position,
                    job_uuid=job_uuid,
                    mutations=mutations,
                )

            for target in ROSETTA_TARGETS:
                if not numeric_values_match(db_row[target], fields[column_index[target]]):
                    errors.add(
                        "tsv_reconciliation",
                        "rossetta_label_disagrees_between_variant_and_global_tsv",
                        pdb_fn=pdb_fn,
                        sqlite_rowid=sqlite_rowid,
                        physical_position=physical_position,
                        job_uuid=job_uuid,
                        mutations=mutations,
                        expected=format_value(db_row[target]),
                        observed=fields[column_index[target]],
                        detail=target,
                    )

            if physical_position >= len(assignments):
                errors.add(
                    "tsv_reconciliation",
                    "global_tsv_has_more_data_rows_than_official_split_universe",
                    pdb_fn=pdb_fn,
                    sqlite_rowid=sqlite_rowid,
                    physical_position=physical_position,
                    job_uuid=job_uuid,
                    mutations=mutations,
                )
                continue

            split_code = assignments[physical_position]
            if split_code == 0:
                errors.add(
                    "tsv_reconciliation",
                    "selected_row_has_no_official_split_assignment",
                    pdb_fn=pdb_fn,
                    sqlite_rowid=sqlite_rowid,
                    physical_position=physical_position,
                    job_uuid=job_uuid,
                    mutations=mutations,
                )
                continue

            pending_updates.append(
                (physical_position, CODE_TO_SPLIT[split_code], ordinal)
            )

            if len(pending_updates) >= 5000:
                stage.executemany(
                    """
                    UPDATE selected_rows
                    SET physical_position = ?, split_assignment = ?
                    WHERE ordinal = ?
                    """,
                    pending_updates,
                )
                stage.commit()
                pending_updates.clear()

        physical_data_row_count = physical_position + 1 if "physical_position" in locals() else 0

    if pending_updates:
        stage.executemany(
            """
            UPDATE selected_rows
            SET physical_position = ?, split_assignment = ?
            WHERE ordinal = ?
            """,
            pending_updates,
        )
        stage.commit()

    if physical_data_row_count != len(assignments):
        errors.add(
            "tsv_reconciliation",
            "global_tsv_data_row_count_disagrees_with_split_universe",
            expected=len(assignments),
            observed=physical_data_row_count,
        )

    print("[global.tsv] checking selected database identities were all observed", flush=True)
    missing_cursor = stage.execute(
        """
        SELECT pdb_fn, sqlite_rowid, job_uuid, mutations
        FROM selected_rows
        WHERE physical_position IS NULL
        ORDER BY ordinal
        """
    )

    for row in missing_cursor:
        errors.add(
            "tsv_reconciliation",
            "variant_row_missing_from_global_tsv",
            pdb_fn=row["pdb_fn"],
            sqlite_rowid=row["sqlite_rowid"],
            job_uuid=row["job_uuid"],
            mutations=row["mutations"],
        )

    reconciliation: dict[str, dict[str, int]] = {}
    for row in stage.execute(
        """
        SELECT
            pdb_fn,
            COUNT(*) AS database_rows,
            SUM(CASE WHEN physical_position IS NOT NULL THEN 1 ELSE 0 END) AS matched_rows,
            SUM(CASE WHEN physical_position IS NULL THEN 1 ELSE 0 END) AS missing_rows
        FROM selected_rows
        GROUP BY pdb_fn
        ORDER BY pdb_fn
        """
    ):
        reconciliation[row["pdb_fn"]] = {
            "database_rows": int(row["database_rows"]),
            "matched_rows": int(row["matched_rows"]),
            "missing_rows": int(row["missing_rows"]),
        }

    reconciliation["__all__"] = {
        "database_rows": int(stage.execute("SELECT COUNT(*) FROM selected_rows").fetchone()[0]),
        "matched_rows": int(
            stage.execute(
                "SELECT COUNT(*) FROM selected_rows WHERE physical_position IS NOT NULL"
            ).fetchone()[0]
        ),
        "missing_rows": int(
            stage.execute(
                "SELECT COUNT(*) FROM selected_rows WHERE physical_position IS NULL"
            ).fetchone()[0]
        ),
    }
    reconciliation["__all__"]["selected_tsv_rows_seen"] = selected_tsv_rows

    return physical_data_row_count, reconciliation


def write_diagnostics(
    *,
    output: Path,
    mutation_statistics: dict[str, Counter[str]],
    reconciliation: dict[str, dict[str, int]],
    stage: sqlite3.Connection,
) -> None:
    mutation_fields = [
        "pdb_fn",
        "database_rows",
        "mutation_tokens",
        "valid_tokens",
        "empty_mutation_strings",
        "malformed_tokens",
        "out_of_range_positions",
        "wild_type_mismatches",
        "duplicate_positions",
        "conflicting_positions",
        "mutation_sites_resolved_in_atom",
        "mutation_sites_unresolved_in_atom",
        "atom_reference_mismatches",
    ]

    mutation_rows = []
    for pdb_fn in sorted(mutation_statistics):
        stats = mutation_statistics[pdb_fn]
        mutation_rows.append({field: stats.get(field, 0) if field != "pdb_fn" else pdb_fn for field in mutation_fields})

    write_dict_tsv(
        output / "mutation_validation_by_protein.tsv",
        mutation_fields,
        mutation_rows,
    )

    reconciliation_rows = []
    for pdb_fn, values in sorted(reconciliation.items()):
        if pdb_fn == "__all__":
            continue
        reconciliation_rows.append({"pdb_fn": pdb_fn, **values})

    if "__all__" in reconciliation:
        reconciliation_rows.append({"pdb_fn": "__all__", **reconciliation["__all__"]})

    reconciliation_fields = sorted(
        {field for row in reconciliation_rows for field in row.keys()},
        key=lambda value: (value != "pdb_fn", value),
    )
    write_dict_tsv(
        output / "source_reconciliation.tsv",
        reconciliation_fields,
        reconciliation_rows,
    )

    duplicate_job_groups = int(
        stage.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT pdb_fn, job_uuid
                FROM selected_rows
                GROUP BY pdb_fn, job_uuid
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )

    duplicate_mutation_groups = int(
        stage.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT pdb_fn, mutations
                FROM selected_rows
                GROUP BY pdb_fn, mutations
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )

    write_dict_tsv(
        output / "duplicate_diagnostics.tsv",
        ["scope", "metric", "count", "interpretation"],
        [
            {
                "scope": "variant_source_identity",
                "metric": "duplicate_(pdb_fn,mutations,job_uuid)",
                "count": 0,
                "interpretation": (
                    "SQLite primary key guarantees this identity is unique; "
                    "the validator stages it with a UNIQUE constraint."
                ),
            },
            {
                "scope": "selected_rows",
                "metric": "groups_with_same_(pdb_fn,job_uuid)_and_multiple_mutations",
                "count": duplicate_job_groups,
                "interpretation": (
                    "Expected to be possible; therefore (pdb_fn,job_uuid) is not used "
                    "as a join key."
                ),
            },
            {
                "scope": "selected_rows",
                "metric": "groups_with_same_(pdb_fn,mutations)_and_multiple_job_uuid",
                "count": duplicate_mutation_groups,
                "interpretation": (
                    "Preserved independently because job_uuid is part of source identity."
                ),
            },
        ],
    )


def compute_finite_summaries_and_standardization(
    *,
    db: sqlite3.Connection,
    stage: sqlite3.Connection,
    panel: list[dict[str, str]],
    output: Path,
    errors: ErrorLog,
) -> dict[tuple[str, str], RunningStats]:
    label_query = (
        "SELECT "
        + ", ".join(quote_identifier(target) for target in ROSETTA_TARGETS)
        + " FROM variant WHERE rowid = ?"
    )

    finite: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    statistics: dict[tuple[str, str], RunningStats] = {
        (panel_row["pdb_fn"], target): RunningStats()
        for panel_row in panel
        for target in ROSETTA_TARGETS
    }

    print("[labels] finite-value accounting and train-only standardization", flush=True)

    cursor = stage.execute(
        """
        SELECT pdb_fn, sqlite_rowid, split_assignment
        FROM selected_rows
        ORDER BY ordinal
        """
    )

    for selected_row in cursor:
        pdb_fn = selected_row["pdb_fn"]
        split = selected_row["split_assignment"]
        sqlite_rowid = int(selected_row["sqlite_rowid"])

        values = db.execute(label_query, (sqlite_rowid,)).fetchone()
        if values is None:
            errors.add(
                "label_validation",
                "sqlite_rowid_missing_during_label_accounting",
                pdb_fn=pdb_fn,
                sqlite_rowid=sqlite_rowid,
            )
            continue

        for target in ROSETTA_TARGETS:
            total_and_finite = finite[(pdb_fn, split, target)]
            total_and_finite[0] += 1

            value = values[target]
            if is_finite_value(value):
                total_and_finite[1] += 1
                if split == "train":
                    statistics[(pdb_fn, target)].add(float(value))

    finite_rows: list[dict[str, Any]] = []
    for panel_row in panel:
        pdb_fn = panel_row["pdb_fn"]
        for split in SPLIT_FILES:
            for target in ROSETTA_TARGETS:
                total, finite_count = finite[(pdb_fn, split, target)]
                finite_rows.append(
                    {
                        "pdb_fn": pdb_fn,
                        "split_assignment": split,
                        "target": target,
                        "rows": total,
                        "finite_rows": finite_count,
                        "nonfinite_rows": total - finite_count,
                    }
                )

    write_dict_tsv(
        output / "finite_label_summary.tsv",
        [
            "pdb_fn",
            "split_assignment",
            "target",
            "rows",
            "finite_rows",
            "nonfinite_rows",
        ],
        finite_rows,
    )

    for panel_row in panel:
        pdb_fn = panel_row["pdb_fn"]
        for target in ROSETTA_TARGETS:
            stat = statistics[(pdb_fn, target)]
            std = stat.population_std()

            if stat.n < 2:
                errors.add(
                    "standardization",
                    "insufficient_finite_training_labels",
                    pdb_fn=pdb_fn,
                    expected=">=2",
                    observed=stat.n,
                    detail=target,
                )
            elif not math.isfinite(std) or std <= 0.0:
                errors.add(
                    "standardization",
                    "nonpositive_or_nonfinite_training_standard_deviation",
                    pdb_fn=pdb_fn,
                    observed=format_value(std),
                    detail=target,
                )

    return statistics


def write_standardization(
    *,
    path: Path,
    panel: list[dict[str, str]],
    statistics: dict[tuple[str, str], RunningStats],
) -> None:
    rows: list[dict[str, Any]] = []

    for panel_row in panel:
        pdb_fn = panel_row["pdb_fn"]
        for target in ROSETTA_TARGETS:
            stat = statistics[(pdb_fn, target)]
            rows.append(
                {
                    "pdb_fn": pdb_fn,
                    "target": target,
                    "finite_train_rows": stat.n,
                    "mean": format_value(stat.mean),
                    "std_population_ddof0": format_value(stat.population_std()),
                    "standardization_provenance_id": STANDARDIZATION_ID,
                }
            )

    write_dict_tsv(
        path,
        [
            "pdb_fn",
            "target",
            "finite_train_rows",
            "mean",
            "std_population_ddof0",
            "standardization_provenance_id",
        ],
        rows,
    )


def write_manifests(
    *,
    db: sqlite3.Connection,
    stage: sqlite3.Connection,
    panel: list[dict[str, str]],
    references: dict[str, ReferenceInfo],
    split_inventory: dict[str, Any],
    output: Path,
    tsv_path: Path,
    db_path: Path,
) -> None:
    manifest_partial = output / "manifest.partial"
    manifest_final = output / "manifest"

    if manifest_partial.exists():
        shutil.rmtree(manifest_partial)
    manifest_partial.mkdir(parents=True)

    panel_by_pdb_fn = {row["pdb_fn"]: row for row in panel}
    label_query = (
        "SELECT "
        + ", ".join(quote_identifier(target) for target in ROSETTA_TARGETS)
        + " FROM variant WHERE rowid = ?"
    )

    label_fields = list(ROSETTA_TARGETS)
    finite_fields = [f"finite__{target}" for target in ROSETTA_TARGETS]

    manifest_fields = [
        "pdb_fn",
        "pdb_id",
        "selected_chain",
        "selection_rank",
        "structural_max_tm_to_other_panel_member",
        "structural_mean_tm_to_other_panel_members",
        "coordinate_source_url",
        "coordinate_sha256",
        "reference_sequence_length",
        "reference_sequence_sha256",
        "sqlite_rowid",
        "physical_row_position_zero_based",
        "split_assignment",
        "job_uuid",
        "mutations",
        "reconstructed_sequence",
        "reconstructed_sequence_sha256",
        "standardization_provenance_id",
        *label_fields,
        *finite_fields,
    ]

    handles: dict[str, Any] = {}
    writers: dict[str, csv.DictWriter] = {}

    file_names = {
        "all": "all_validated_rows.tsv.gz",
        "train": "train.tsv.gz",
        "val": "val.tsv.gz",
        "test": "test.tsv.gz",
        "supertest": "supertest.tsv.gz",
    }

    try:
        for key, filename in file_names.items():
            handle = gzip.open(
                manifest_partial / filename,
                "wt",
                encoding="utf-8",
                newline="",
                compresslevel=6,
            )
            writer = csv.DictWriter(
                handle,
                fieldnames=manifest_fields,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            handles[key] = handle
            writers[key] = writer

        print("[manifest] reconstructing validated sequences and writing compressed manifests", flush=True)

        cursor = stage.execute(
            """
            SELECT
                pdb_fn,
                mutations,
                job_uuid,
                sqlite_rowid,
                physical_position,
                split_assignment
            FROM selected_rows
            ORDER BY ordinal
            """
        )

        for row in cursor:
            pdb_fn = row["pdb_fn"]
            ref = references[pdb_fn]
            panel_row = panel_by_pdb_fn[pdb_fn]
            sequence = apply_mutations(ref.sequence, row["mutations"])

            if len(sequence) != len(ref.sequence) or len(sequence) > 1022:
                raise RuntimeError(
                    f"Unexpected reconstructed sequence-length violation for {pdb_fn}"
                )

            label_values = db.execute(
                label_query,
                (int(row["sqlite_rowid"]),),
            ).fetchone()

            if label_values is None:
                raise RuntimeError(
                    f"sqlite_rowid disappeared during manifest write: {row['sqlite_rowid']}"
                )

            manifest_row: dict[str, Any] = {
                "pdb_fn": pdb_fn,
                "pdb_id": ref.pdb_id,
                "selected_chain": ref.chain,
                "selection_rank": panel_row.get("selection_rank", ""),
                "structural_max_tm_to_other_panel_member": panel_row.get(
                    "max_tm_to_other_panel_member", ""
                ),
                "structural_mean_tm_to_other_panel_members": panel_row.get(
                    "mean_tm_to_other_panel_members", ""
                ),
                "coordinate_source_url": ref.source_url,
                "coordinate_sha256": ref.coordinate_sha256,
                "reference_sequence_length": len(ref.sequence),
                "reference_sequence_sha256": ref.sequence_sha256,
                "sqlite_rowid": row["sqlite_rowid"],
                "physical_row_position_zero_based": row["physical_position"],
                "split_assignment": row["split_assignment"],
                "job_uuid": row["job_uuid"],
                "mutations": row["mutations"],
                "reconstructed_sequence": sequence,
                "reconstructed_sequence_sha256": sha256_text(sequence),
                "standardization_provenance_id": STANDARDIZATION_ID,
            }

            for target in ROSETTA_TARGETS:
                value = label_values[target]
                manifest_row[target] = format_value(value)
                manifest_row[f"finite__{target}"] = "1" if is_finite_value(value) else "0"

            writers["all"].writerow(manifest_row)
            writers[row["split_assignment"]].writerow(manifest_row)

    finally:
        for handle in handles.values():
            handle.close()

    provenance = {
        "manifest_schema_version": 1,
        "selected_panel": [row["pdb_fn"] for row in panel],
        "source_database": str(db_path),
        "source_tsv": str(tsv_path),
        "source_split_inventory": split_inventory,
        "source_identity": ["pdb_fn", "mutations", "job_uuid"],
        "physical_position_definition": "zero-based data-row position in global.tsv; header excluded",
        "reference_sequence_source": "selected-chain canonical SEQRES sequence from verified RCSB PDB coordinate file",
        "variant_sequence_construction": "apply fully validated canonical substitutions to verified selected-chain SEQRES sequence",
        "rosetta_targets": list(ROSETTA_TARGETS),
        "explicit_excluded_training_targets": ["fa_atr", "fa_sol"],
        "standardization": {
            "id": STANDARDIZATION_ID,
            "fit_subset": "selected six-protein train partition only",
            "grouping": ["pdb_fn", "target"],
            "standard_deviation": "population standard deviation, ddof=0",
            "validation_and_test_labels_used_for_fit": False,
        },
    }

    with (manifest_partial / "manifest_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if manifest_final.exists():
        raise RuntimeError(f"Refusing to overwrite existing manifest directory: {manifest_final}")

    manifest_partial.rename(manifest_final)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the six-protein METL panel and create standalone manifests."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("results/esm2_metl6/structure_selection/provisional_panel.tsv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/global/metl6_reference_pdb_rcsb/reference_audit.tsv"),
    )
    parser.add_argument(
        "--coordinate-manifest",
        type=Path,
        default=Path("data/global/metl6_reference_pdb_rcsb/coordinate_manifest.tsv"),
    )
    parser.add_argument("--db", type=Path, default=Path("data/global/global.db"))
    parser.add_argument("--tsv", type=Path, default=Path("data/global/global.tsv"))
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path(
            "data/global/splits/"
            "standard_tr0.9_tu0.05_te0.05_w2a93d88bac32_r2098"
        ),
    )
    parser.add_argument(
        "--supertest-alias",
        type=Path,
        default=Path("data/global/splits/supertest_w2a93d88bac32_s0.05_r6853.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/esm2_metl6/full_validation"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate --output. Use only for a deliberate rerun.",
    )
    args = parser.parse_args()

    required_paths = [
        args.panel,
        args.audit,
        args.coordinate_manifest,
        args.db,
        args.tsv,
        args.split_dir,
    ]
    for path in required_paths:
        if not path.exists():
            raise RuntimeError(f"Required input is missing: {path}")

    make_output_directory(args.output, args.force)
    errors = ErrorLog(args.output / "validation_errors.tsv")
    stage_path = args.output / "_validation_stage.sqlite3"
    stage: sqlite3.Connection | None = None
    db: sqlite3.Connection | None = None

    try:
        panel_fields, panel = read_tsv_dicts(args.panel)
        required_panel_fields = {
            "selection_rank",
            "pdb_fn",
            "pdb_id",
            "chain",
            "coordinate_sha256",
        }
        missing_panel_fields = sorted(required_panel_fields.difference(panel_fields))
        if missing_panel_fields:
            raise RuntimeError(
                f"Panel lacks required columns: {missing_panel_fields}"
            )

        if len(panel) != 6:
            raise RuntimeError(
                f"Panel must contain exactly six rows; observed {len(panel)}"
            )

        panel_pdb_fns = [row["pdb_fn"] for row in panel]
        if len(set(panel_pdb_fns)) != 6:
            raise RuntimeError("Panel pdb_fn values must be unique.")

        write_dict_tsv(args.output / "input_panel_snapshot.tsv", panel_fields, panel)

        audit_fields, audit_rows = read_tsv_dicts(args.audit)
        if "pdb_fn" not in audit_fields:
            raise RuntimeError("reference_audit.tsv lacks pdb_fn.")

        audit_by_pdb_fn = {
            row["pdb_fn"]: row
            for row in audit_rows
            if row["pdb_fn"] in set(panel_pdb_fns)
        }

        selected_audit_rows = [
            audit_by_pdb_fn[pdb_fn]
            for pdb_fn in panel_pdb_fns
            if pdb_fn in audit_by_pdb_fn
        ]
        write_dict_tsv(
            args.output / "selected_audit_rows.tsv",
            audit_fields,
            selected_audit_rows,
        )

        coordinate_manifest = load_coordinate_manifest(args.coordinate_manifest)

        print("[reference] verifying panel/audit/coordinate provenance", flush=True)
        references = validate_references(
            panel=panel,
            audit_by_pdb_fn=audit_by_pdb_fn,
            coordinate_manifest=coordinate_manifest,
            output=args.output,
            errors=errors,
        )

        if set(references) != set(panel_pdb_fns) or errors.count:
            errors.close()
            write_validation_summary(
                args.output,
                status="FAILED_REFERENCE_PREFLIGHT",
                panel=panel,
                errors=errors,
                extra={
                    "training_permitted": False,
                    "manifest_created": False,
                    "reason": "Reference provenance or chain reconciliation failed.",
                },
            )
            return 2

        assignments, split_inventory = load_and_validate_splits(
            args.split_dir,
            args.supertest_alias,
            args.output,
            errors,
        )

        if errors.count:
            errors.close()
            write_validation_summary(
                args.output,
                status="FAILED_SPLIT_PREFLIGHT",
                panel=panel,
                errors=errors,
                extra={
                    "training_permitted": False,
                    "manifest_created": False,
                    "reason": "Official split partition validation failed.",
                },
            )
            return 2

        db = sqlite3.connect(database_uri(args.db), uri=True)
        db.row_factory = sqlite3.Row
        stage = create_stage_database(stage_path)

        mutation_statistics, selected_database_rows = fetch_selected_and_validate_mutations(
            db=db,
            stage=stage,
            panel=panel,
            references=references,
            audit_by_pdb_fn=audit_by_pdb_fn,
            errors=errors,
        )

        physical_tsv_rows, reconciliation = reconcile_global_tsv(
            tsv_path=args.tsv,
            db=db,
            stage=stage,
            selected_pdb_fns=set(panel_pdb_fns),
            assignments=assignments,
            errors=errors,
        )

        write_diagnostics(
            output=args.output,
            mutation_statistics=mutation_statistics,
            reconciliation=reconciliation,
            stage=stage,
        )

        if errors.count:
            errors.close()
            write_validation_summary(
                args.output,
                status="FAILED_FULL_VALIDATION",
                panel=panel,
                errors=errors,
                extra={
                    "training_permitted": False,
                    "manifest_created": False,
                    "selected_database_rows": selected_database_rows,
                    "global_tsv_data_rows": physical_tsv_rows,
                    "official_split_expected_rows": len(assignments),
                    "reason": (
                        "No sequences, manifests, or standardization statistics "
                        "were created because one or more full-validation checks failed."
                    ),
                },
            )
            return 2

        statistics = compute_finite_summaries_and_standardization(
            db=db,
            stage=stage,
            panel=panel,
            output=args.output,
            errors=errors,
        )

        if errors.count:
            errors.close()
            write_validation_summary(
                args.output,
                status="FAILED_STANDARDIZATION_PREFLIGHT",
                panel=panel,
                errors=errors,
                extra={
                    "training_permitted": False,
                    "manifest_created": False,
                    "selected_database_rows": selected_database_rows,
                    "global_tsv_data_rows": physical_tsv_rows,
                    "reason": (
                        "Training-only per-protein/per-target standardization "
                        "was not usable for every required target."
                    ),
                },
            )
            return 2

        manifest_partial = args.output / "manifest.partial"
        manifest_partial.mkdir(parents=True, exist_ok=False)
        write_standardization(
            path=manifest_partial / "train_standardization.tsv",
            panel=panel,
            statistics=statistics,
        )
        manifest_partial.rmdir()

        write_manifests(
            db=db,
            stage=stage,
            panel=panel,
            references=references,
            split_inventory=split_inventory,
            output=args.output,
            tsv_path=args.tsv,
            db_path=args.db,
        )

        errors.close()
        write_validation_summary(
            args.output,
            status="PASS",
            panel=panel,
            errors=errors,
            extra={
                "training_permitted": True,
                "manifest_created": True,
                "selected_database_rows": selected_database_rows,
                "global_tsv_data_rows": physical_tsv_rows,
                "official_split_expected_rows": len(assignments),
                "manifest_directory": str(args.output / "manifest"),
                "standardization_provenance_id": STANDARDIZATION_ID,
                "forbidden_legacy_dependencies_used": [],
            },
        )

        print(
            "\nPASS: full six-protein validation completed.\n"
            f"Manifest: {args.output / 'manifest'}\n"
            f"Summary:  {args.output / 'validation_summary.json'}",
            flush=True,
        )
        return 0

    finally:
        if stage is not None:
            stage.close()
        if db is not None:
            db.close()

        # The staging database contains only transient source identities and locators.
        for suffix in ("", "-journal", "-wal", "-shm"):
            candidate = Path(str(stage_path) + suffix)
            if candidate.exists():
                candidate.unlink()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nFATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
