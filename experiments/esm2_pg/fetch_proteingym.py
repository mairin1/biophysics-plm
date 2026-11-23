"""Download and verify the ProteinGym v1.3 DMS substitution benchmark.

Writes data/proteingym_v1.3/ with the 217 assay CSVs and the reference file,
plus a fetch_manifest.json recording sizes and checksums.

Run once on a login node (or any node with outbound network) before the GPU
jobs. The archive is ~1.0GB unzipped.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    EXPECTED_ASSAYS, PG_ARCHIVE_URL, PG_ASSAYS, PG_REFERENCE, PG_REFERENCE_URL,
    PG_ROOT, PG_VERSION, digest, read_csv, write_json,
)

REQUIRED_COLUMNS = {'mutant', 'mutated_sequence', 'DMS_score', 'DMS_score_bin'}


def download(url, destination):
    print(f'Downloading {url}', flush=True)
    subprocess.run(
        ['curl', '-fL', '--retry', '3', '--retry-delay', '5', '-o', str(destination), url],
        check=True,
    )
    if destination.stat().st_size == 0:
        raise RuntimeError(f'Empty download: {url}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if the data is already present')
    args = parser.parse_args()

    if PG_ASSAYS.is_dir() and PG_REFERENCE.is_file() and not args.force:
        existing = sorted(PG_ASSAYS.glob('*.csv'))
        if len(existing) == EXPECTED_ASSAYS:
            print(f'Already present: {len(existing)} assays in {PG_ASSAYS}', flush=True)
            return
        print(f'Incomplete download ({len(existing)} assays); refetching', flush=True)

    PG_ROOT.mkdir(parents=True, exist_ok=True)
    download(PG_REFERENCE_URL, PG_REFERENCE)

    reference = read_csv(PG_REFERENCE)
    if len(reference) != EXPECTED_ASSAYS:
        raise RuntimeError(
            f'Reference file lists {len(reference)} assays, expected {EXPECTED_ASSAYS}'
        )

    with tempfile.TemporaryDirectory(dir=PG_ROOT) as tmpdir:
        tmpdir = Path(tmpdir)
        archive = tmpdir / 'DMS_ProteinGym_substitutions.zip'
        download(PG_ARCHIVE_URL, archive)
        print('Extracting', flush=True)
        staging = tmpdir / 'unzipped'
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                # Refuse absolute paths and parent traversal before extracting.
                target = (staging / member).resolve()
                if not str(target).startswith(str(staging.resolve())):
                    raise RuntimeError(f'Unsafe archive member: {member}')
            zf.extractall(staging)

        found = sorted(staging.rglob('*.csv'))
        if len(found) != EXPECTED_ASSAYS:
            raise RuntimeError(
                f'Archive holds {len(found)} CSVs, expected {EXPECTED_ASSAYS}'
            )
        if PG_ASSAYS.exists():
            shutil.rmtree(PG_ASSAYS)
        PG_ASSAYS.mkdir(parents=True)
        for path in found:
            shutil.move(str(path), PG_ASSAYS / path.name)

    assays = sorted(PG_ASSAYS.glob('*.csv'))
    if len(assays) != EXPECTED_ASSAYS:
        raise RuntimeError(f'Expected {EXPECTED_ASSAYS} assay files, found {len(assays)}')

    # Every assay named by the reference file must exist, and must carry a
    # continuous DMS_score. The clinical benchmark in data/protein_gym has only
    # binarised labels, so this check also guards against pointing the pipeline
    # at the wrong dataset.
    by_name = {path.name: path for path in assays}
    total_rows = 0
    for row in reference:
        filename = row['DMS_filename']
        if filename not in by_name:
            raise RuntimeError(f'Reference names a missing assay file: {filename}')
        with open(by_name[filename], newline='') as handle:
            header = handle.readline().strip().split(',')
            missing = REQUIRED_COLUMNS - set(header)
            if missing:
                raise RuntimeError(f'{filename} lacks columns: {sorted(missing)}')
            total_rows += sum(1 for _ in handle)

    manifest = {
        'version': PG_VERSION,
        'archive_url': PG_ARCHIVE_URL,
        'reference_url': PG_REFERENCE_URL,
        'n_assays': len(assays),
        'total_variants': total_rows,
        'reference_sha256': digest(PG_REFERENCE),
        'assay_sha256': {path.name: digest(path) for path in assays},
    }
    write_json(PG_ROOT / 'fetch_manifest.json', manifest)
    print(f'OK: {len(assays)} assays, {total_rows} variants -> {PG_ASSAYS}', flush=True)


if __name__ == '__main__':
    main()
