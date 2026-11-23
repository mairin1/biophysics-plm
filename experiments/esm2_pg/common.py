"""Shared constants and helpers for the ESM-2 biophysical-supervision experiment.

This experiment ports the Check-in 2 fine-tuning methods (GENTLE, L2SP, PARTIAL)
from EvoDiff OA_DM_38M to ESM-2 650M, then evaluates the resulting frozen
encoders on the ProteinGym v1.3 DMS substitution assays with a per-assay
supervised ridge probe.

Deliberate inheritances from the EvoDiff protocol, none of them re-tuned for
ESM-2: the 18 Rosetta targets (groups A+B+C), the encoder/head learning rates,
the L2-SP coefficient, and the PARTIAL freezing fraction. Treat cross-model
comparisons as exploratory for that reason.
"""

from pathlib import Path
import csv
import hashlib
import json
import os

import numpy as np

# ---------------------------------------------------------------- model

# Non-gated checkpoint: no license acknowledgement or HF token required.
ESM2_MODEL = 'facebook/esm2_t33_650M_UR50D'
EMBED_DIM = 1280
N_LAYERS = 33
# ESM-2 positional limit is 1024 tokens including <cls> and <eos>.
MAX_RESIDUES = 1022

# ---------------------------------------------------------------- paths

BASE = Path('metl-global/analysis/esm2_pg_20261122')
SOURCE = Path('metl-global/exports/run_29051193')
PG_ROOT = Path('data/proteingym_v1.3')
PG_ASSAYS = PG_ROOT / 'DMS_ProteinGym_substitutions'
PG_REFERENCE = PG_ROOT / 'DMS_substitutions.csv'

PG_VERSION = 'v1.3'
PG_ARCHIVE_URL = (
    'https://marks.hms.harvard.edu/proteingym/'
    f'ProteinGym_{PG_VERSION}/DMS_ProteinGym_substitutions.zip'
)
PG_REFERENCE_URL = (
    'https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/'
    'main/reference_files/DMS_substitutions.csv'
)
EXPECTED_ASSAYS = 217

# ---------------------------------------------------------------- training

SEEDS = [20261116, 20261117, 20261118]
METHODS = ['GENTLE', 'L2SP', 'PARTIAL']
CONDITIONS = ['BASELINE'] + METHODS

# Rosetta groups A + B + C, in the Check-in 2 order. fa_atr / fa_sol are
# deliberately excluded: they stay reserved as probe targets.
TARGETS = [
    'fa_dun', 'omega', 'p_aa_pp', 'rama_prepro', 'pro_close', 'yhh_planarity',
    'fa_elec', 'fa_rep', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb',
    'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'cbeta', 'cenpack', 'env',
]
RESERVED = ['fa_atr', 'fa_sol']

L2SP_COEFFICIENT = 50.0
ENCODER_LR_STANDARD = 2e-5
ENCODER_LR_GENTLE = 2e-6
HEAD_LR = 2e-4
WEIGHT_DECAY = 0.01
PARTIAL_TRAINABLE_FRACTION = 0.25  # last quarter of the transformer stack

# ---------------------------------------------------------------- evaluation

# Per-assay variant cap. 217 assays hold ~2.47M variants in total, which is too
# many to embed once per model, let alone ten times. The subsample is drawn once
# and shared by every model so per-assay comparisons stay paired.
MAX_VARIANTS_PER_ASSAY = 2000
SUBSAMPLE_SEED = 20261122
MIN_VARIANTS_PER_ASSAY = 100
N_FOLDS = 5
SPLIT_SCHEMES = ['random', 'modulo']
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def run_dir(condition, seed=None):
    """Directory for one trained model. BASELINE is seed-independent."""
    if condition == 'BASELINE':
        return BASE / 'runs' / 'BASELINE'
    return BASE / 'runs' / f'{condition}_seed{seed}'


def eval_dir(condition, seed=None):
    if condition == 'BASELINE':
        return BASE / 'proteingym' / 'BASELINE'
    return BASE / 'proteingym' / f'{condition}_seed{seed}'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with open(tmp, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def ranks(a):
    """Average ranks, matching the tie handling used in the EvoDiff scripts."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind='mergesort')
    sorted_a = a[order]
    edges = np.r_[0, np.flatnonzero(sorted_a[1:] != sorted_a[:-1]) + 1, len(sorted_a)]
    out = np.empty(len(a), dtype=np.float64)
    for i, j in zip(edges[:-1], edges[1:]):
        out[order[i:j]] = (i + j + 1) / 2
    return out


def spearman(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(y) < 3 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        return float('nan')
    a, b = ranks(y), ranks(pred)
    if a.std() < 1e-10 or b.std() < 1e-10:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def parse_mutant_positions(mutant):
    """Positions touched by a ProteinGym `mutant` string such as 'A1P:D2N'.

    Returned 0-indexed. ProteinGym numbers positions from 1 relative to the
    reference `target_seq`.
    """
    positions = []
    for token in str(mutant).split(':'):
        token = token.strip()
        if not token:
            continue
        digits = ''.join(c for c in token[1:-1] if c.isdigit())
        if not digits:
            raise ValueError(f'Cannot parse mutation token: {token!r}')
        positions.append(int(digits) - 1)
    if not positions:
        raise ValueError(f'No mutations parsed from {mutant!r}')
    return positions
