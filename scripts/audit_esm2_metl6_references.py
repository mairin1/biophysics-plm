#!/usr/bin/env python3
"""Download/audit candidate PDB references for standalone ESM-2 METL-6.

Writes only beneath data/global/metl6_reference_pdb_rcsb/.
Reads global.db in SQLite read-only mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "UNK": "X",
}

MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])$")
PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_candidate(name: str) -> tuple[str, str]:
    """Return literal RCSB lookup ID and explicit chain, if filename gives one.

    This does not rename pdb_fn or infer a replacement filename. The first
    four literal characters are recorded only as a provisional coordinate lookup
    key and are subsequently checked against the downloaded PDB HEADER record.
    """
    stem = name.removesuffix(".pdb")
    parts = stem.split("_")
    pdb_id = parts[0].lower()

    if not PDB_ID.fullmatch(pdb_id):
        raise ValueError(f"Cannot obtain a four-character PDB lookup ID from {name!r}")

    # Examples:
    #   1h2e_A_p.pdb       -> explicit chain A
    #   1a3a_remod_p.pdb   -> no explicit chain
    #   1bsg_p.pdb         -> no explicit chain
    explicit_chain = ""
    if len(parts) == 3 and parts[2] == "p" and parts[1] != "remod":
        explicit_chain = parts[1]

    return pdb_id, explicit_chain


def read_seqres(path: Path) -> dict[str, str]:
    chains: dict[str, list[str]] = defaultdict(list)

    for line in path.read_text(encoding="latin-1", errors="replace").splitlines():
        if not line.startswith("SEQRES"):
            continue
        chain = line[11].strip() or "_"
        residues = line[19:70].split()
        chains[chain].extend(AA3_TO_1.get(residue.upper(), "X") for residue in residues)

    return {chain: "".join(sequence) for chain, sequence in chains.items()}


def header_id(path: Path) -> str:
    for line in path.read_text(encoding="latin-1", errors="replace").splitlines():
        if line.startswith("HEADER"):
            return line[62:66].strip().lower()
    return ""


def fetch_pdb(pdb_id: str, out_path: Path) -> tuple[str, str]:
    """Return source URL and failure string; retain valid cached files."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"

    if out_path.is_file() and out_path.stat().st_size > 100:
        return url, ""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, 4):
        try:
            request = Request(url, headers={"User-Agent": "esm2-metl6-audit/1.0"})
            with urlopen(request, timeout=60) as response:
                payload = response.read()

            if not payload.startswith(b"HEADER"):
                raise RuntimeError("download did not begin with a PDB HEADER record")

            partial = out_path.with_suffix(".pdb.partial")
            partial.write_bytes(payload)
            partial.replace(out_path)
            return url, ""

        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            if attempt == 3:
                return url, f"{type(exc).__name__}: {exc}"
            time.sleep(attempt)

    raise AssertionError("unreachable")


def sampled_mutation_evidence(
    cursor: sqlite3.Cursor,
    pdb_fn: str,
    sample_limit: int,
) -> tuple[int, int, dict[int, set[str]], int]:
    """Read at most sample_limit indexed rows; never scan the whole dataset."""
    total = cursor.execute(
        """
        SELECT COUNT(*)
        FROM variant INDEXED BY pdb_fn_index
        WHERE pdb_fn = ?
        """,
        (pdb_fn,),
    ).fetchone()[0]

    rows = cursor.execute(
        """
        SELECT mutations
        FROM variant INDEXED BY pdb_fn_index
        WHERE pdb_fn = ?
        LIMIT ?
        """,
        (pdb_fn, sample_limit),
    )

    position_to_wt: dict[int, set[str]] = defaultdict(set)
    malformed = 0
    sampled = 0

    for (mutations,) in rows:
        sampled += 1
        for token in str(mutations).split(","):
            match = MUTATION.fullmatch(token.strip())
            if match is None:
                malformed += 1
                continue
            wt, position, _mut = match.groups()
            position_to_wt[int(position)].add(wt)

    return total, sampled, position_to_wt, malformed


def compare_sequence(sequence: str, expected: dict[int, set[str]]) -> dict[str, int]:
    matched = mismatched = out_of_range = conflicting_wt = 0

    for position, observed_wts in expected.items():
        if len(observed_wts) != 1:
            conflicting_wt += 1
            continue
        if position > len(sequence):
            out_of_range += 1
        elif sequence[position - 1] in observed_wts:
            matched += 1
        else:
            mismatched += 1

    return {
        "checked_positions": len(expected),
        "matched_positions": matched,
        "mismatched_positions": mismatched,
        "out_of_range_positions": out_of_range,
        "conflicting_wt_positions": conflicting_wt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-limit", type=int, default=512)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/global/metl6_reference_pdb_rcsb"),
    )
    args = parser.parse_args()

    data = Path("data/global")
    pdb_fns_path = data / "pdb_fns.txt"
    db_path = data / "global.db"

    if not pdb_fns_path.is_file() or not db_path.is_file():
        raise SystemExit("Expected data/global/pdb_fns.txt and data/global/global.db")

    candidates = sorted({
        Path(line.strip()).name
        for line in pdb_fns_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    })

    args.out.mkdir(parents=True, exist_ok=True)

    source_rows = {}
    for pdb_fn in candidates:
        pdb_id, explicit_chain = parse_candidate(pdb_fn)
        source_rows[pdb_fn] = (pdb_id, explicit_chain)

    unique_pdb_ids = sorted({pdb_id for pdb_id, _chain in source_rows.values()})
    downloads: dict[str, dict[str, str]] = {}

    print(f"Downloading/reusing {len(unique_pdb_ids)} distinct coordinate files.")
    for i, pdb_id in enumerate(unique_pdb_ids, start=1):
        path = args.out / f"{pdb_id}.pdb"
        url, error = fetch_pdb(pdb_id, path)
        downloads[pdb_id] = {
            "path": str(path),
            "url": url,
            "error": error,
            "sha256": sha256(path) if not error else "",
            "header_id": header_id(path) if not error else "",
        }
        print(f"[{i:3d}/{len(unique_pdb_ids)}] {pdb_id}: "
              f"{'OK' if not error else 'FAILED: ' + error}")
        time.sleep(0.10)

    uri = f"file:{db_path.resolve()}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=30)
    cur = con.cursor()
    cur.execute("PRAGMA query_only = ON")
    cur.execute("PRAGMA busy_timeout = 15000")

    fieldnames = [
        "pdb_fn", "pdb_id", "explicit_chain", "coordinate_path",
        "source_url", "coordinate_sha256", "header_id", "header_matches_lookup",
        "variant_rows", "sampled_variant_rows", "malformed_mutation_tokens",
        "sampled_unique_positions", "candidate_chains",
        "explicit_chain_metrics", "best_chain", "best_chain_metrics", "status",
    ]
    audit_rows = []

    for pdb_fn in candidates:
        pdb_id, explicit_chain = source_rows[pdb_fn]
        download = downloads[pdb_id]
        base = {
            "pdb_fn": pdb_fn,
            "pdb_id": pdb_id,
            "explicit_chain": explicit_chain,
            "coordinate_path": download["path"],
            "source_url": download["url"],
            "coordinate_sha256": download["sha256"],
            "header_id": download["header_id"],
            "header_matches_lookup": str(download["header_id"] == pdb_id).lower(),
        }

        total, sampled, expected, malformed = sampled_mutation_evidence(
            cur, pdb_fn, args.sample_limit
        )
        base.update({
            "variant_rows": total,
            "sampled_variant_rows": sampled,
            "malformed_mutation_tokens": malformed,
            "sampled_unique_positions": len(expected),
        })

        if download["error"]:
            base.update({
                "candidate_chains": "",
                "explicit_chain_metrics": "",
                "best_chain": "",
                "best_chain_metrics": "",
                "status": "DOWNLOAD_FAILED",
            })
            audit_rows.append(base)
            continue

        chains = read_seqres(Path(download["path"]))
        base["candidate_chains"] = ",".join(
            f"{chain}:{len(sequence)}" for chain, sequence in sorted(chains.items())
        )

        if not chains:
            base.update({
                "explicit_chain_metrics": "",
                "best_chain": "",
                "best_chain_metrics": "",
                "status": "NO_SEQRES",
            })
            audit_rows.append(base)
            continue

        metrics = {
            chain: compare_sequence(sequence, expected)
            for chain, sequence in chains.items()
        }

        def errors(item):
            _chain, metric = item
            return (
                metric["mismatched_positions"]
                + metric["out_of_range_positions"]
                + metric["conflicting_wt_positions"]
            )

        best_chain, best_metric = min(
            metrics.items(),
            key=lambda item: (errors(item), -item[1]["matched_positions"], item[0]),
        )

        explicit_metric = metrics.get(explicit_chain)
        base["explicit_chain_metrics"] = (
            repr(explicit_metric) if explicit_metric is not None else ""
        )
        base["best_chain"] = best_chain
        base["best_chain_metrics"] = repr(best_metric)

        if download["header_id"] != pdb_id:
            base["status"] = "HEADER_ID_MISMATCH"
        elif explicit_chain:
            if explicit_metric is None:
                base["status"] = "EXPLICIT_CHAIN_MISSING"
            elif errors((explicit_chain, explicit_metric)) == 0:
                base["status"] = "EXPLICIT_CHAIN_SAMPLE_MATCH"
            else:
                base["status"] = "EXPLICIT_CHAIN_SAMPLE_MISMATCH"
        else:
            perfect = [
                chain for chain, metric in metrics.items()
                if errors((chain, metric)) == 0 and metric["matched_positions"] > 0
            ]
            if len(perfect) == 1:
                # This remains provisional. Full validation happens only after
                # structural shortlisting, before any sequence is accepted.
                base["status"] = "PROVISIONAL_UNSPECIFIED_CHAIN"
            elif len(perfect) > 1:
                base["status"] = "AMBIGUOUS_UNSPECIFIED_CHAIN"
            else:
                base["status"] = "NO_SAMPLE_MATCHING_CHAIN"

        audit_rows.append(base)

    con.close()

    audit_path = args.out / "reference_audit.tsv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(audit_rows)

    manifest_path = args.out / "coordinate_manifest.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pdb_id", "coordinate_path", "source_url",
                        "coordinate_sha256", "header_id", "error"],
            delimiter="\t",
        )
        writer.writeheader()
        for pdb_id in unique_pdb_ids:
            source = downloads[pdb_id]
            writer.writerow({
                "pdb_id": pdb_id,
                "coordinate_path": source["path"],
                "source_url": source["url"],
                "coordinate_sha256": source["sha256"],
                "header_id": source["header_id"],
                "error": source["error"],
            })

    counts = Counter(row["status"] for row in audit_rows)
    print("\n=== Audit-status counts ===")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    print(f"\nAudit:    {audit_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()