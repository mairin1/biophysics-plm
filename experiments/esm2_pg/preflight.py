"""Verify the METL export before any ESM-2 job is submitted.

The point of this experiment is to compare ESM-2 against the EvoDiff Check-in 2
runs. That comparison is only meaningful if both were trained on the same data,
so this checks the export byte-for-byte against the SHA-256 hashes recorded in
results/evodiff/checkin2/preflight.json when the EvoDiff runs were made.

The export itself is gitignored and so was never carried by the repository. If
it came from a collaborator's filesystem, this is what confirms the copy is the
same one, and not a re-export that would silently differ.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE, SOURCE, digest, read_json, write_json  # noqa: E402

REFERENCE = Path('results/evodiff/checkin2/preflight.json')
REQUIRED = ['sequences.csv.gz', 'rosetta_numeric.csv.gz', 'sample_manifest.csv']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--allow-mismatch', action='store_true',
                        help='Proceed despite differing hashes; the ESM-2 and '
                             'EvoDiff results are then not directly comparable')
    args = parser.parse_args()

    if not REFERENCE.is_file():
        raise RuntimeError(f'Missing the EvoDiff reference hashes: {REFERENCE}')
    expected = read_json(REFERENCE)['source_hashes']

    missing = [name for name in REQUIRED if not (args.source / name).is_file()]
    if missing:
        raise RuntimeError(
            f'Export incomplete at {args.source}; missing {missing}.\n'
            'The METL export is gitignored and is not reproduced by cloning. '
            'Copy it from wherever the EvoDiff Check-in 2 runs were made, or '
            'regenerate it with scripts/export_metl.sbatch.')

    results, mismatched = {}, []
    for name in REQUIRED:
        actual = digest(args.source / name)
        want = expected.get(name)
        agrees = (actual == want)
        results[name] = {'sha256': actual, 'expected': want, 'matches': agrees}
        status = 'OK' if agrees else 'MISMATCH'
        print(f'{status:9} {name}  {actual[:16]}...', flush=True)
        if not agrees:
            mismatched.append(name)

    if mismatched and not args.allow_mismatch:
        raise RuntimeError(
            f'Export does not match the EvoDiff Check-in 2 data: {mismatched}.\n'
            'Training ESM-2 on a different export would make the comparison '
            'against the EvoDiff results invalid. Pass --allow-mismatch only '
            'if you intend that, and record the reason.')

    write_json(BASE / 'preflight.json', {
        'source': str(args.source),
        'files': results,
        'matches_evodiff_checkin2': not mismatched,
        'reference': str(REFERENCE),
        'status': 'PASS' if not mismatched else 'PASS_WITH_MISMATCH',
    })
    if mismatched:
        print('WARNING: proceeding with a different export than EvoDiff used',
              flush=True)
    else:
        print('Export is byte-identical to the EvoDiff Check-in 2 data', flush=True)


if __name__ == '__main__':
    main()
