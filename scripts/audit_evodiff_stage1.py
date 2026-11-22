from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import csv
import gzip
import hashlib
import json
import sys

import numpy as np

FAMILIES = {
    'A': ['fa_dun', 'omega', 'p_aa_pp', 'rama_prepro', 'pro_close', 'yhh_planarity'],
    'B': ['fa_elec', 'fa_rep', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb'],
    'C': ['fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'cbeta', 'cenpack', 'env'],
    'D_proposed': ['buried_all', 'buried_np', 'contact_all', 'degree', 'exposed_hydrophobics', 'ss_mis'],
}
RESERVED = ['fa_atr', 'fa_sol', 'total_sasa', 'rg', 'pack', 'unsat_hbond']
CONDITIONS = {'A': ['A'], 'B': ['B'], 'C': ['C'], 'AB': ['A', 'B'], 'BC': ['B', 'C'], 'ABC': ['A', 'B', 'C']}


def main():
    root = Path('metl-global')
    source = root / 'exports/run_29051193'
    out = root / 'analysis/evodiff_stage1_20261116'
    out.mkdir(parents=True, exist_ok=True)
    rows_by_protein = defaultdict(list)
    counts = Counter()
    positions = set()
    with gzip.open(source / 'sequences.csv.gz', 'rt') as sf, gzip.open(source / 'rosetta_numeric.csv.gz', 'rt') as lf:
        seqr, valr = csv.DictReader(sf), csv.DictReader(lf)
        available = [k for k in valr.fieldnames if k != 'row_position']
        necessary = set(RESERVED + sum((FAMILIES[k] for k in ('A', 'B', 'C')), []))
        missing = sorted(necessary - set(available))
        if missing:
            raise RuntimeError(f'Required attributes absent: {missing}')
        for seq, lab in zip_longest(seqr, valr):
            if seq is None or lab is None:
                raise RuntimeError('Mismatched CSV lengths')
            rid = int(seq['row_position'])
            if rid != int(lab['row_position']) or rid in positions:
                raise RuntimeError(f'Row mismatch or duplicate at {rid}')
            positions.add(rid)
            protein, role, split = seq['pdb_fn'], seq['role'], seq['split']
            counts[(protein, role, split)] += 1
            if role == 'train' and split == 'train':
                rows_by_protein[protein].append([float(lab[k]) for k in available])
    if len(positions) != 274000 or len(rows_by_protein) != 6:
        raise RuntimeError(f'Unexpected size or protein count: {len(positions)}, {len(rows_by_protein)}')
    if not all(counts[(p, 'train', 'train')] == 40000 and counts[(p, 'train', 'val')] == 2000
               and counts[(p, 'train', 'test')] == 2000 for p in rows_by_protein):
        raise RuntimeError('Existing training protein split counts differ from original manifest')
    holdouts = sorted({p for p, role, split in counts if role != 'train'})
    if len(holdouts) != 2 or not all(
            sum(n for (p2, role, split), n in counts.items() if p2 == p and role != 'train') == 5000
            for p in holdouts):
        raise RuntimeError(f'Unexpected holdout counts: {holdouts}')
    catalog = []
    standardized = []
    columns = {k: i for i, k in enumerate(available)}
    for protein in sorted(rows_by_protein):
        a = np.asarray(rows_by_protein[protein], dtype=np.float32)
        if not np.isfinite(a).all():
            raise RuntimeError(f'Nonfinite labels in {protein}')
        means, std = a.mean(0, dtype=np.float64), a.std(0, dtype=np.float64)
        for k in available:
            v = a[:, columns[k]]
            catalog.append([protein, k, len(v), float(means[columns[k]]), float(std[columns[k]]),
                            float((v == 0).mean()), float(np.quantile(v, .01)), float(np.median(v)),
                            float(np.quantile(v, .99)), int(std[columns[k]] > 1e-4)])
        target_names = sum((FAMILIES[k] for k in ('A', 'B', 'C')), [])
        ai = np.array([columns[x] for x in target_names])
        x = a[:, ai].astype(np.float64)
        x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-8)
        standardized.append(x)
    with open(out / 'target_audit.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['protein', 'target', 'n', 'mean', 'std', 'fraction_zero', 'q01', 'median', 'q99', 'active_std_gt_1e_4'])
        w.writerows(catalog)
    x = np.concatenate(standardized, axis=0)
    corr = np.nan_to_num(np.corrcoef(x, rowvar=False), nan=0.0)
    names = sum((FAMILIES[k] for k in ('A', 'B', 'C')), [])
    with open(out / 'within_protein_correlations.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['target'] + names)
        for n, row in zip(names, corr):
            w.writerow([n] + row.tolist())
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=70, ha='right', fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title('Rosetta attribute correlations within training proteins (z-scored per protein)')
        fig.colorbar(im, ax=ax, label='Pearson r')
        fig.tight_layout()
        fig.savefig(out / 'target_correlations.png', dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        values = np.array([[next(r[9] for r in catalog if r[0] == p and r[1] == name) for name in names]
                           for p in sorted(rows_by_protein)])
        im = ax.imshow(values, cmap='Greys', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(names)), names, rotation=70, ha='right', fontsize=8)
        ax.set_yticks(range(len(rows_by_protein)), sorted(rows_by_protein), fontsize=8)
        ax.set_title('Active target terms per training protein')
        fig.tight_layout()
        fig.savefig(out / 'active_targets.png', dpi=180)
        plt.close(fig)
        plot_error = None
    except Exception as e:
        plot_error = str(e)
    source_hashes = {}
    for filename in ['sequences.csv.gz', 'rosetta_numeric.csv.gz', 'sample_manifest.csv']:
        path = source / filename
        if path.exists():
            digest = hashlib.sha256()
            with open(path, 'rb') as inp:
                for chunk in iter(lambda: inp.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
            source_hashes[filename] = digest.hexdigest()
    summary = {
        'source_file_sha256': source_hashes,
        'source': str(source), 'rows_checked': len(positions),
        'train_proteins': sorted(rows_by_protein), 'holdout_proteins': holdouts,
        'counts': [{'protein': p, 'role': r, 'split': s, 'count': n} for (p,r,s),n in sorted(counts.items())],
        'numeric_columns': available, 'families': FAMILIES, 'conditions': CONDITIONS,
        'reserved_probes': RESERVED,
        'family_D_missing': sorted(set(FAMILIES['D_proposed']) - set(available)),
        'rotation_ready': False,
        'rotation_blocker': 'Current export has only 5k held-out-protein test rows and no training or validation rows for those proteins; construct an eight-protein master pool first',
        'audit_note': 'Correlations from training rows only. This gate validates existing Fold A labels, NOT the scientific optimality of each family.',
        'plot_error': plot_error,
    }
    (out / 'audit_summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'AUDIT_OK').touch()
    print(json.dumps({k: summary[k] for k in ['rows_checked', 'train_proteins', 'holdout_proteins', 'family_D_missing', 'rotation_ready', 'plot_error']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
