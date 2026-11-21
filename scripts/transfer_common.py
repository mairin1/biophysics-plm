from pathlib import Path
from collections import defaultdict
from itertools import zip_longest
import csv
import gzip
import hashlib
import json
import math
import numpy as np

ROOT = Path('metl-global/analysis/evodiff_transfer_20261119')
OLD = Path('metl-global/analysis/evodiff_stage1_20261116')
SOURCE = Path('metl-global/exports/run_29051193')
FAMILIES = {
    'A': ['fa_dun', 'omega', 'p_aa_pp', 'rama_prepro', 'pro_close', 'yhh_planarity'],
    'B': ['fa_elec', 'fa_rep', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb'],
    'C': ['fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'cbeta', 'cenpack', 'env'],
}
CONDITIONS = ['A', 'B', 'C', 'AB', 'BC', 'ABC']
SEEDS = [20261116, 20261117, 20261118]
TARGETS = ['fa_atr', 'fa_sol', 'total_sasa', 'rg', 'pack', 'unsat_hbond']
EXPECTED_HOLDOUTS = ['1i1j_remod_p.pdb', '1xdz_remod_p.pdb']
SUPPORT_SEEDS = [713, 719, 727]
SUPPORT_SIZES = [32, 128, 512]
ALPHAS = [10.0, 100.0, 1000.0]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def records():
    grouped = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)
    ids = set()
    with gzip.open(SOURCE/'sequences.csv.gz','rt') as sf, gzip.open(SOURCE/'rosetta_numeric.csv.gz','rt') as yf:
        xs, ys = csv.DictReader(sf), csv.DictReader(yf)
        if not set(TARGETS).issubset(ys.fieldnames):
            raise ValueError('Missing reserved target properties')
        for s,y in zip_longest(xs,ys):
            if s is None or y is None or s['row_position'] != y['row_position']:
                raise ValueError('Mismatched sequence/label CSV records')
            p, split, role, rid = s['pdb_fn'], s['split'], s['role'], int(s['row_position'])
            if rid in ids:
                raise ValueError('Duplicate physical row ID')
            ids.add(rid)
            vals = np.array([float(y[t]) for t in TARGETS], dtype=np.float32)
            if not np.isfinite(vals).all():
                raise ValueError(f'Nonfinite label at {rid}')
            if (role == 'train' and split in ('train','val')) or (role != 'train' and split == 'test'):
                grouped[p][split].append({'row_id':rid, 'sequence':s['sequence'], 'labels':vals.tolist(), 'role':role})
                counts[(p,role,split)] += 1
    if len(ids) != 274000:
        raise ValueError(f'Unexpected total file rows: {len(ids)}')
    if len(grouped) != 8:
        raise ValueError('Expected exactly eight proteins')
    for p, splits in grouped.items():
        if 'train' in splits:
            if len(splits['train']) != 40000 or len(splits['val']) != 2000:
                raise ValueError(f'Original train/val counts differ for {p}')
        else:
            if len(splits['test']) != 5000:
                raise ValueError(f'Original holdout counts differ for {p}')
    return grouped


def spearman(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(y) < 3 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        return float('nan')
    def rank(a):
        ix = np.argsort(a, kind='mergesort')
        b = a[ix]
        edges = np.r_[0, np.flatnonzero(b[1:] != b[:-1])+1, len(b)]
        result = np.empty(len(a), dtype=np.float64)
        for i,j in zip(edges[:-1],edges[1:]):
            result[ix[i:j]] = (i+j+1)/2
        return result
    a, b = rank(y), rank(pred)
    if a.std() < 1e-10 or b.std() < 1e-10:
        return float('nan')
    return float(np.corrcoef(a,b)[0,1])


def as_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, entries, fields):
    with open(path,'w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(entries)
