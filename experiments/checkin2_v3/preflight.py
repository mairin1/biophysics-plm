import csv
import json
import sys
from pathlib import Path

from common import BASE, STAGE1, TRANSFER, SOURCE, SEEDS, TARGETS, digest, write_json


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    checks = [
        STAGE1 / 'AUDIT_OK',
        STAGE1 / 'report' / 'REPORT_READY',
        TRANSFER / 'PREFLIGHT_OK',
        TRANSFER / 'report' / 'TRANSFER_COMPLETE',
        TRANSFER / 'evaluation_manifest.json',
        Path('scripts/train_evodiff_stage1.py'),
        Path('scripts/transfer_common.py'),
        Path('scripts/evaluate_transfer.py'),
    ]
    for path in checks:
        if not path.is_file():
            raise RuntimeError(f'Required artifact is missing: {path}')
    a = json.loads((STAGE1 / 'audit_summary.json').read_text())
    m = json.loads((TRANSFER / 'evaluation_manifest.json').read_text())
    hashes = {}
    for name in ('sequences.csv.gz', 'rosetta_numeric.csv.gz', 'sample_manifest.csv'):
        hashes[name] = digest(SOURCE / name)
        if hashes[name] != a['source_file_sha256'][name]:
            raise RuntimeError(f'Source differs from Stage I: {name}')
        if hashes[name] != m['metadata']['source_sha256'][name]:
            raise RuntimeError(f'Source differs from transfer evaluation: {name}')
    if sorted(m['metadata']['holdouts']) != sorted(['1i1j_remod_p.pdb', '1xdz_remod_p.pdb']):
        raise RuntimeError('Held-out proteins do not match the original transfer evaluation')
    if len(m['metadata']['sources']) != 6:
        raise RuntimeError('Expected six source proteins')
    if a['families']['A'] + a['families']['B'] + a['families']['C'] != TARGETS:
        raise RuntimeError('ABC targets differ from the locked Stage I definition')
    status = []
    for seed in SEEDS:
        for condition in ('ABC',):
            p = STAGE1 / 'runs' / f'{condition}_seed{seed}'
            if not (p/'SUCCESS').is_file() or not (p/'best_model.pt').is_file():
                raise RuntimeError(f'Missing reference checkpoint: {p}')
            q = TRANSFER / 'runs' / f'{condition}_seed{seed}'
            if not (q/'SUCCESS').is_file() or not (q/'retention_metrics.csv').is_file():
                raise RuntimeError(f'Missing old transfer measurements: {q}')
            status.append(f'{condition}_seed{seed}')
    original = TRANSFER / 'runs' / f'BASELINE_seed{SEEDS[0]}'
    for name in ('SUCCESS','zero_shot_metrics.csv','few_shot_metrics.csv','retention_metrics.csv'):
        if not (original/name).is_file():
            raise RuntimeError(f'Missing pretrained reference file: {original/name}')
    with (TRANSFER / 'report' / 'run_status.csv').open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 19 or not all(r['transfer_complete']=='1' and r['retention_complete']=='1' for r in rows):
        raise RuntimeError('Previous transfer evaluation was not complete in all 19 conditions')
    write_json(BASE / 'preflight.json', {
        'status':'PASS', 'source_hashes':hashes, 'source_proteins':m['metadata']['sources'],
        'heldout_proteins':m['metadata']['holdouts'], 'reference_checkpoints':status,
        'existing_transfer_manifest':str(TRANSFER / 'evaluation_manifest.json'),
        'experimental_status':'Exploratory, heldout labels already examined; not independent confirmation'
    })
    (BASE/'PREFLIGHT_OK').touch()
    print('PASS: source hashes, Stage I reference, Stage II references, and fixed held-out evaluation manifest')

if __name__ == '__main__':
    main()
