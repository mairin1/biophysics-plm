#!/usr/bin/env python3
"""Select a structurally diverse provisional METL-6 panel.

Inputs:
  data/global/metl6_reference_pdb_rcsb/reference_audit.tsv
  downloaded RCSB coordinate files recorded by that audit

Only EXPLICIT_CHAIN_SAMPLE_MATCH candidates are eligible.  Each candidate's
explicit PDB chain is converted to a CA-only chain file, all eligible pairs are
aligned with US-align, and a deterministic all-pair-seeded greedy max-min
selection proposes six structurally diverse proteins.

The output is a structural shortlist, not permission to train.  The final six
must still pass full mutation/reference-sequence validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


AUDIT_STATUS = "EXPLICIT_CHAIN_SAMPLE_MATCH"
TM_PATTERN = re.compile(r"TM-score=\s*([0-9.]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")

    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    partial.replace(path)


def pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def extract_ca_chain(source: Path, chain: str, destination: Path) -> int:
    """Write one CA atom per residue for one explicit PDB chain."""
    if len(chain) != 1:
        raise ValueError(f"Expected one-character PDB chain ID, got {chain!r}")

    seen_residues = set()
    ca_lines = []

    with source.open("r", encoding="latin-1", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line) < 54:
                continue
            if line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue

            # Retain the primary alternate conformation only.
            altloc = line[16]
            if altloc not in (" ", "A", "1"):
                continue

            residue_key = (line[22:26], line[26])
            if residue_key in seen_residues:
                continue

            seen_residues.add(residue_key)
            ca_lines.append(line.rstrip("\n") + "\n")

    if not ca_lines:
        raise ValueError(
            f"No CA atoms found for chain {chain!r} in {source}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(ca_lines) + "TER\nEND\n", encoding="ascii")
    return len(ca_lines)


def run_alignment(usalign: Path, first: dict, second: dict) -> dict:
    """Run one US-align comparison and retain both length-normalised TM scores."""
    cmd = [str(usalign), first["chain_path"], second["chain_path"]]

    try:
        completed = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "pdb_fn_1": first["pdb_fn"],
            "pdb_fn_2": second["pdb_fn"],
            "tm_norm_1": "",
            "tm_norm_2": "",
            "similarity_max": "",
            "distance": "",
            "status": "TIMEOUT",
            "detail": "US-align exceeded 180 seconds",
        }

    scores = [float(match.group(1)) for match in TM_PATTERN.finditer(completed.stdout)]

    if completed.returncode != 0:
        return {
            "pdb_fn_1": first["pdb_fn"],
            "pdb_fn_2": second["pdb_fn"],
            "tm_norm_1": "",
            "tm_norm_2": "",
            "similarity_max": "",
            "distance": "",
            "status": "USALIGN_FAILED",
            "detail": completed.stdout[-1000:].replace("\n", " "),
        }

    if len(scores) < 2:
        return {
            "pdb_fn_1": first["pdb_fn"],
            "pdb_fn_2": second["pdb_fn"],
            "tm_norm_1": "",
            "tm_norm_2": "",
            "similarity_max": "",
            "distance": "",
            "status": "UNPARSEABLE_OUTPUT",
            "detail": completed.stdout[-1000:].replace("\n", " "),
        }

    tm_1, tm_2 = scores[:2]
    similarity = max(tm_1, tm_2)

    return {
        "pdb_fn_1": first["pdb_fn"],
        "pdb_fn_2": second["pdb_fn"],
        "tm_norm_1": f"{tm_1:.6f}",
        "tm_norm_2": f"{tm_2:.6f}",
        "similarity_max": f"{similarity:.6f}",
        "distance": f"{1.0 - similarity:.6f}",
        "status": "OK",
        "detail": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "data/global/metl6_reference_pdb_rcsb/reference_audit.tsv"
        ),
    )
    parser.add_argument(
        "--usalign",
        type=Path,
        default=Path("tools/usalign/USalign"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/esm2_metl6/structure_selection"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum simultaneous US-align subprocesses.",
    )
    parser.add_argument(
        "--min-ca",
        type=int,
        default=40,
        help="Exclude chains with fewer resolved CA residues.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any prior pairwise_usalign.tsv cache.",
    )
    args = parser.parse_args()

    if not args.audit.is_file():
        raise SystemExit(f"Missing audit TSV: {args.audit}")
    if not args.usalign.is_file():
        raise SystemExit(f"Missing US-align executable: {args.usalign}")

    args.out.mkdir(parents=True, exist_ok=True)
    chains_dir = args.out / "ca_chains"

    eligible = []
    excluded = []

    with args.audit.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["status"] != AUDIT_STATUS:
                excluded.append({
                    "pdb_fn": row["pdb_fn"],
                    "reason": f"audit_status={row['status']}",
                })
                continue

            chain = row["explicit_chain"]
            coordinate = Path(row["coordinate_path"])

            if not coordinate.is_file():
                excluded.append({
                    "pdb_fn": row["pdb_fn"],
                    "reason": f"missing_coordinate={coordinate}",
                })
                continue

            output_name = (
                row["pdb_fn"].removesuffix(".pdb")
                + f"__chain_{chain}.pdb"
            )
            chain_path = chains_dir / output_name

            try:
                ca_residues = extract_ca_chain(
                    coordinate,
                    chain,
                    chain_path,
                )
            except (OSError, ValueError) as exc:
                excluded.append({
                    "pdb_fn": row["pdb_fn"],
                    "reason": f"chain_extraction_failed={exc}",
                })
                continue

            if ca_residues < args.min_ca:
                excluded.append({
                    "pdb_fn": row["pdb_fn"],
                    "reason": (
                        f"ca_residues={ca_residues}"
                        f"<min_ca={args.min_ca}"
                    ),
                })
                continue

            eligible.append({
                "pdb_fn": row["pdb_fn"],
                "pdb_id": row["pdb_id"],
                "chain": chain,
                "variant_rows": int(row["variant_rows"]),
                "ca_residues": ca_residues,
                "chain_path": str(chain_path),
                "coordinate_sha256": row["coordinate_sha256"],
            })

    eligible.sort(key=lambda row: row["pdb_fn"])
    write_tsv(
        args.out / "excluded_candidates.tsv",
        ["pdb_fn", "reason"],
        excluded,
    )

    if len(eligible) < 6:
        raise SystemExit(
            f"Only {len(eligible)} eligible chains remain; need at least six."
        )

    print(
        f"Eligible explicit-chain candidates: {len(eligible)}; "
        f"excluded: {len(excluded)}"
    )

    pair_fields = [
        "pdb_fn_1",
        "pdb_fn_2",
        "tm_norm_1",
        "tm_norm_2",
        "similarity_max",
        "distance",
        "status",
        "detail",
    ]
    pair_path = args.out / "pairwise_usalign.tsv"

    cached = {}
    if pair_path.is_file() and not args.fresh:
        with pair_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["status"] == "OK":
                    cached[pair_key(row["pdb_fn_1"], row["pdb_fn_2"])] = row

    candidates_by_name = {row["pdb_fn"]: row for row in eligible}
    all_pairs = list(itertools.combinations(eligible, 2))
    missing = [
        (first, second)
        for first, second in all_pairs
        if pair_key(first["pdb_fn"], second["pdb_fn"]) not in cached
    ]

    print(
        f"Pairwise comparisons: {len(all_pairs)} total; "
        f"{len(cached)} cached; {len(missing)} to run."
    )

    if missing:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(run_alignment, args.usalign, first, second): (
                    first["pdb_fn"],
                    second["pdb_fn"],
                )
                for first, second in missing
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                cached[pair_key(row["pdb_fn_1"], row["pdb_fn_2"])] = row
                if index % 50 == 0 or index == len(missing):
                    print(f"Completed {index}/{len(missing)} alignments.", flush=True)

    pair_rows = [
        cached[pair_key(first["pdb_fn"], second["pdb_fn"])]
        for first, second in all_pairs
    ]
    write_tsv(pair_path, pair_fields, pair_rows)

    failed_names = set()
    for row in pair_rows:
        if row["status"] != "OK":
            failed_names.add(row["pdb_fn_1"])
            failed_names.add(row["pdb_fn_2"])

    usable = [
        row for row in eligible
        if row["pdb_fn"] not in failed_names
    ]

    if len(usable) < 6:
        raise SystemExit(
            f"Only {len(usable)} candidates have complete successful "
            "pairwise US-align results."
        )

    usable_names = [row["pdb_fn"] for row in usable]

    def similarity(first: str, second: str) -> float:
        row = cached[pair_key(first, second)]
        return float(row["similarity_max"])

    def distance(first: str, second: str) -> float:
        return 1.0 - similarity(first, second)

    def greedy_panel(seed: tuple[str, str]) -> list[str]:
        selected = list(seed)

        while len(selected) < 6:
            remaining = [
                name for name in usable_names
                if name not in selected
            ]

            def rank(name: str) -> tuple:
                distances = [distance(name, other) for other in selected]
                return (
                    -min(distances),
                    -(sum(distances) / len(distances)),
                    -candidates_by_name[name]["variant_rows"],
                    -candidates_by_name[name]["ca_residues"],
                    name,
                )

            selected.append(sorted(remaining, key=rank)[0])

        return selected

    def panel_score(panel: list[str]) -> tuple:
        similarities = [
            similarity(first, second)
            for first, second in itertools.combinations(panel, 2)
        ]
        return (
            max(similarities),                         # minimize worst redundancy
            sum(similarities) / len(similarities),    # then mean redundancy
            -min(candidates_by_name[x]["variant_rows"] for x in panel),
            tuple(sorted(panel)),
        )

    best_panel = None
    best_score = None

    for seed in itertools.combinations(usable_names, 2):
        panel = greedy_panel(seed)
        score = panel_score(panel)
        if best_score is None or score < best_score:
            best_panel, best_score = panel, score

    assert best_panel is not None

    panel_rows = []
    for rank, name in enumerate(best_panel, start=1):
        other_names = [x for x in best_panel if x != name]
        similarities = [similarity(name, other) for other in other_names]
        row = dict(candidates_by_name[name])
        row.update({
            "selection_rank": rank,
            "max_tm_to_other_panel_member": f"{max(similarities):.6f}",
            "mean_tm_to_other_panel_members": (
                f"{sum(similarities) / len(similarities):.6f}"
            ),
        })
        panel_rows.append(row)

    panel_fields = [
        "selection_rank",
        "pdb_fn",
        "pdb_id",
        "chain",
        "variant_rows",
        "ca_residues",
        "coordinate_sha256",
        "max_tm_to_other_panel_member",
        "mean_tm_to_other_panel_members",
        "chain_path",
    ]
    write_tsv(args.out / "provisional_panel.tsv", panel_fields, panel_rows)

    selected_pairs = [
        row for row in pair_rows
        if (
            row["pdb_fn_1"] in best_panel
            and row["pdb_fn_2"] in best_panel
        )
    ]
    write_tsv(
        args.out / "provisional_panel_pairwise.tsv",
        pair_fields,
        selected_pairs,
    )

    summary = {
        "audit_tsv": str(args.audit),
        "audit_sha256": sha256(args.audit),
        "audit_status_required": AUDIT_STATUS,
        "min_ca_residues": args.min_ca,
        "usalign_executable": str(args.usalign),
        "usalign_sha256": sha256(args.usalign),
        "eligible_explicit_chain_candidates": len(eligible),
        "candidates_with_complete_pairwise_results": len(usable),
        "pairwise_comparisons": len(all_pairs),
        "pairwise_failures": len(pair_rows) - sum(
            row["status"] == "OK" for row in pair_rows
        ),
        "selection_method": (
            "all-pair-seeded greedy max-min structural-distance selection; "
            "symmetric similarity=max(TM-score normalized by structure 1, "
            "TM-score normalized by structure 2); distance=1-similarity"
        ),
        "provisional_panel": best_panel,
        "max_pairwise_similarity_in_panel": best_score[0],
        "mean_pairwise_similarity_in_panel": best_score[1],
        "training_permitted": False,
        "remaining_gate": (
            "Validate every mutation token for each selected protein against "
            "the selected reference sequence before constructing sequences."
        ),
    }
    (args.out / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== Provisional structural panel ===")
    for row in panel_rows:
        print(
            f"{row['selection_rank']}. {row['pdb_fn']} "
            f"chain={row['chain']} CA={row['ca_residues']} "
            f"variants={row['variant_rows']} "
            f"worst_TM={row['max_tm_to_other_panel_member']}"
        )

    print(f"\nWrote {args.out / 'provisional_panel.tsv'}")
    print(f"Wrote {args.out / 'provisional_panel_pairwise.tsv'}")
    print(f"Wrote {args.out / 'selection_summary.json'}")


if __name__ == "__main__":
    main()